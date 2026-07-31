import numpy as np
import torch
import os
import cv2
import h5py
import pickle
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T

# T2/T4 image augmentation, gated by env ACT_AUG=1 (T1/T3 leave it off). Color jitter targets
# demo_randomized's random_light / background distribution shift -- matches the T2 hint menu.
# Applied per camera on the [0,1] float image; qpos/action are untouched.
_ACT_AUG = os.environ.get("ACT_AUG", "0") == "1"
_AUG_TF = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05) if _ACT_AUG else None
_GRIPPER_EVENT_SAMPLE_PROB = float(os.environ.get("TRON2_ACT_GRIPPER_EVENT_SAMPLE_PROB", "0"))
_GRIPPER_EVENT_PRE_WINDOW = int(os.environ.get("TRON2_ACT_GRIPPER_EVENT_PRE_WINDOW", "50"))
_GRIPPER_EVENT_THRESHOLD = float(os.environ.get("TRON2_ACT_GRIPPER_EVENT_THRESHOLD", "0.5"))
_RECOVERY_START_EPISODE = int(os.environ.get("TRON2_ACT_RECOVERY_START_EPISODE", "-1"))
_RECOVERY_SAMPLE_PROB = float(os.environ.get("TRON2_ACT_RECOVERY_SAMPLE_PROB", "0"))
_RECOVERY_SPLIT_SEED = int(os.environ.get("TRON2_ACT_RECOVERY_SPLIT_SEED", "0"))
_FOCUS_START_EPISODE = int(os.environ.get("TRON2_ACT_FOCUS_START_EPISODE", "-1"))
_FOCUS_SAMPLE_PROB = float(os.environ.get("TRON2_ACT_FOCUS_SAMPLE_PROB", "0"))
_FIXED_STATS_PATH = os.environ.get("TRON2_ACT_FIXED_STATS_PATH", "").strip()
_PHASE_SAMPLE_PROBS_TEXT = os.environ.get("TRON2_ACT_PHASE_SAMPLE_PROBS", "").strip()
_PHASE_SAMPLE_PROBS = None
if _PHASE_SAMPLE_PROBS_TEXT:
    _PHASE_SAMPLE_PROBS = np.asarray(
        [float(value) for value in _PHASE_SAMPLE_PROBS_TEXT.split(",")],
        dtype=np.float64,
    )
    if _PHASE_SAMPLE_PROBS.shape != (5,):
        raise ValueError(
            "TRON2_ACT_PHASE_SAMPLE_PROBS must contain five comma-separated values: "
            "approach,grasp,transport,place,release"
        )
    if np.any(_PHASE_SAMPLE_PROBS < 0) or not np.isclose(_PHASE_SAMPLE_PROBS.sum(), 1.0):
        raise ValueError("TRON2_ACT_PHASE_SAMPLE_PROBS must be non-negative and sum to 1")
if not 0.0 <= _GRIPPER_EVENT_SAMPLE_PROB <= 1.0:
    raise ValueError("TRON2_ACT_GRIPPER_EVENT_SAMPLE_PROB must be in [0, 1]")
if _GRIPPER_EVENT_PRE_WINDOW < 0:
    raise ValueError("TRON2_ACT_GRIPPER_EVENT_PRE_WINDOW must be non-negative")
if not 0.0 <= _RECOVERY_SAMPLE_PROB < 1.0:
    raise ValueError("TRON2_ACT_RECOVERY_SAMPLE_PROB must be in [0, 1)")
if not 0.0 <= _FOCUS_SAMPLE_PROB < 1.0:
    raise ValueError("TRON2_ACT_FOCUS_SAMPLE_PROB must be in [0, 1)")
if _RECOVERY_SAMPLE_PROB + _FOCUS_SAMPLE_PROB >= 1.0:
    raise ValueError("recovery and focus sampling probabilities must sum to less than 1")
if (_RECOVERY_SAMPLE_PROB > 0 or _FOCUS_SAMPLE_PROB > 0) and _RECOVERY_START_EPISODE < 0:
    raise ValueError("TRON2_ACT_RECOVERY_START_EPISODE is required when recovery sampling is enabled")
if _FOCUS_SAMPLE_PROB > 0 and _FOCUS_START_EPISODE < 0:
    raise ValueError("TRON2_ACT_FOCUS_START_EPISODE is required when focus sampling is enabled")

import IPython

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_action_len,
                 gripper_event_sample_prob=0.0):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_action_len = max_action_len
        self.gripper_event_sample_prob = gripper_event_sample_prob
        self.is_sim = None
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def _sample_start_ts(self, root, episode_len):
        if _PHASE_SAMPLE_PROBS is not None:
            return self._sample_phase_start_ts(root, episode_len)
        if (self.gripper_event_sample_prob > 0
                and np.random.random() < self.gripper_event_sample_prob):
            gripper = np.asarray(root["/action"][:, [7, 15]])
            closing = ((gripper[:-1] >= _GRIPPER_EVENT_THRESHOLD)
                       & (gripper[1:] < _GRIPPER_EVENT_THRESHOLD))
            events = np.flatnonzero(np.any(closing, axis=1)) + 1
            if events.size:
                event_ts = int(np.random.choice(events))
                lookback = min(_GRIPPER_EVENT_PRE_WINDOW, event_ts)
                return event_ts - int(np.random.randint(lookback + 1))
        return int(np.random.choice(episode_len))

    @staticmethod
    def _sample_phase_start_ts(root, episode_len):
        """Sample one of five manipulation phases using gripper transitions.

        The split is deliberately training-only and task-agnostic. A trajectory
        that starts with a closed gripper (for example an object-lift recovery
        suffix) starts in the transport phase. The final reopen event anchors
        placement/release, which are short under uniform timestep sampling.
        """
        gripper = np.asarray(root["/action"][:, [7, 15]])
        closed = np.any(gripper < _GRIPPER_EVENT_THRESHOLD, axis=1)
        close_events = np.flatnonzero(~closed[:-1] & closed[1:]) + 1
        open_events = np.flatnonzero(closed[:-1] & ~closed[1:]) + 1

        close_ts = 0 if closed[0] else (int(close_events[0]) if close_events.size else episode_len // 5)
        later_open = open_events[open_events > close_ts]
        open_ts = int(later_open[-1]) if later_open.size else max(close_ts + 1, int(0.9 * episode_len))

        grasp_end = min(open_ts, close_ts + max(20, int(0.08 * episode_len)))
        place_start = max(grasp_end, open_ts - max(35, int(0.12 * episode_len)))
        boundaries = (
            (0, max(1, close_ts)),
            (close_ts, max(close_ts + 1, grasp_end)),
            (grasp_end, max(grasp_end + 1, place_start)),
            (place_start, max(place_start + 1, open_ts)),
            (open_ts, episode_len),
        )
        phase = int(np.random.choice(len(boundaries), p=_PHASE_SAMPLE_PROBS))
        start, stop = boundaries[phase]
        start = min(max(0, start), episode_len - 1)
        stop = min(max(start + 1, stop), episode_len)
        return int(np.random.randint(start, stop))

    def __getitem__(self, index):
        sample_full_episode = False

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = None
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = self._sample_start_ts(root, episode_len)
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                # JPEG-encoded 存储 → 解码回 (H,W,3) uint8(与旧 raw 存储对训练透明)
                image_dict[cam_name] = cv2.imdecode(
                    np.frombuffer(root[f"/observations/images/{cam_name}"][start_ts], np.uint8),
                    cv2.IMREAD_COLOR)
            # get all actions after and including start_ts
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                if os.environ.get("TRON2_ACT_ACTION_SHIFT", "0") == "1":
                    action_start = max(0, start_ts - 1)
                else:
                    action_start = start_ts
                action = root["/action"][action_start:]
                action_len = episode_len - action_start

        self.is_sim = is_sim
        padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)  # 根据max_action_len初始化
        padded_action[:action_len] = action
        is_pad = np.ones(self.max_action_len, dtype=bool)  # 初始化为全1（True）
        is_pad[:action_len] = 0  # 前action_len个位置设置为0（False），表示非填充部分

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        if _AUG_TF is not None:
            # per-camera color jitter (T2/T4); independent random params per view
            image_data = torch.stack([_AUG_TF(image_data[k]) for k in range(image_data.shape[0])])
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]  # Assuming this is a numpy array
            action = root["/action"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    # Pad all tensors to the maximum size
    max_qpos_len = max(q.size(0) for q in all_qpos_data)
    max_action_len = max(a.size(0) for a in all_action_data)

    padded_qpos = []
    for qpos in all_qpos_data:
        current_len = qpos.size(0)
        if current_len < max_qpos_len:
            # Pad with the last element
            pad = qpos[-1:].repeat(max_qpos_len - current_len, 1)
            qpos = torch.cat([qpos, pad], dim=0)
        padded_qpos.append(qpos)

    padded_action = []
    for action in all_action_data:
        current_len = action.size(0)
        if current_len < max_action_len:
            pad = action[-1:].repeat(max_action_len - current_len, 1)
            action = torch.cat([action, pad], dim=0)
        padded_action.append(action)

    all_qpos_data = torch.stack(padded_qpos)
    all_action_data = torch.stack(padded_action)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }

    return stats, max_action_len


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, distributed=False):
    print(f"\nData from: {dataset_dir}\n")
    # obtain train test split
    train_ratio = 0.8
    weighted_sampling = _RECOVERY_SAMPLE_PROB > 0 or _FOCUS_SAMPLE_PROB > 0
    if weighted_sampling:
        if _RECOVERY_START_EPISODE <= 0 or _RECOVERY_START_EPISODE >= num_episodes:
            raise ValueError(
                f"invalid recovery boundary {_RECOVERY_START_EPISODE} for {num_episodes} episodes"
            )
        # Preserve the exact base-only permutation/split. Recovery uses an independent
        # RNG so adding it does not perturb the baseline's NumPy state or base split.
        base_indices = np.random.permutation(_RECOVERY_START_EPISODE)
        recovery_stop = _FOCUS_START_EPISODE if _FOCUS_SAMPLE_PROB > 0 else num_episodes
        invalid_recovery_stop = recovery_stop <= _RECOVERY_START_EPISODE
        if _FOCUS_SAMPLE_PROB > 0:
            invalid_recovery_stop = invalid_recovery_stop or recovery_stop >= num_episodes
        else:
            invalid_recovery_stop = invalid_recovery_stop or recovery_stop > num_episodes
        if invalid_recovery_stop:
            raise ValueError(
                f"invalid focus boundary {_FOCUS_START_EPISODE} for recovery boundary "
                f"{_RECOVERY_START_EPISODE} and {num_episodes} episodes"
            )
        recovery_indices = np.arange(_RECOVERY_START_EPISODE, recovery_stop)
        recovery_indices = np.random.RandomState(_RECOVERY_SPLIT_SEED).permutation(recovery_indices)
        base_split = int(train_ratio * len(base_indices))
        recovery_split = int(train_ratio * len(recovery_indices))
        train_groups = [base_indices[:base_split], recovery_indices[:recovery_split]]
        val_groups = [base_indices[base_split:], recovery_indices[recovery_split:]]
        if _FOCUS_SAMPLE_PROB > 0:
            focus_indices = np.arange(_FOCUS_START_EPISODE, num_episodes)
            focus_indices = np.random.RandomState(_RECOVERY_SPLIT_SEED + 1).permutation(focus_indices)
            focus_split = int(train_ratio * len(focus_indices))
            train_groups.append(focus_indices[:focus_split])
            val_groups.append(focus_indices[focus_split:])
        train_indices = np.concatenate(train_groups)
        val_indices = np.concatenate(val_groups)
    else:
        shuffled_indices = np.random.permutation(num_episodes)
        train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
        val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats, max_action_len = get_norm_stats(dataset_dir, num_episodes)
    if _FIXED_STATS_PATH:
        with open(_FIXED_STATS_PATH, "rb") as handle:
            fixed_stats = pickle.load(handle)
        expected_dim = norm_stats["qpos_mean"].shape
        for key in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
            if key not in fixed_stats:
                raise KeyError(f"fixed normalization stats missing {key}: {_FIXED_STATS_PATH}")
            if np.asarray(fixed_stats[key]).shape != expected_dim:
                raise ValueError(
                    f"fixed normalization stats {key} shape "
                    f"{np.asarray(fixed_stats[key]).shape}, expected {expected_dim}"
                )
        norm_stats = fixed_stats
        print(f"Using fixed normalization stats from {_FIXED_STATS_PATH}")

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats, max_action_len,
        gripper_event_sample_prob=_GRIPPER_EVENT_SAMPLE_PROB,
    )
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, max_action_len)
    if _GRIPPER_EVENT_SAMPLE_PROB > 0:
        print("Gripper event sampling "
              f"prob={_GRIPPER_EVENT_SAMPLE_PROB} pre_window={_GRIPPER_EVENT_PRE_WINDOW} "
              f"threshold={_GRIPPER_EVENT_THRESHOLD}")
    if _PHASE_SAMPLE_PROBS is not None:
        print(
            "Phase timestep sampling approach,grasp,transport,place,release="
            + ",".join(f"{value:.3f}" for value in _PHASE_SAMPLE_PROBS)
        )
    # Under DDP each rank gets a disjoint shard via DistributedSampler (which owns the shuffling) --
    # this is what makes N ranks ~Nx throughput. Non-distributed keeps plain shuffle=True.
    if distributed and weighted_sampling:
        raise ValueError("recovery-weighted sampling is only implemented for single-GPU training")
    if weighted_sampling:
        episode_ids = np.asarray(train_indices)
        focus_mask = ((episode_ids >= _FOCUS_START_EPISODE)
                      if _FOCUS_SAMPLE_PROB > 0 else np.zeros(len(episode_ids), dtype=bool))
        recovery_mask = ((episode_ids >= _RECOVERY_START_EPISODE) & ~focus_mask)
        base_mask = ~(recovery_mask | focus_mask)
        recovery_count = int(recovery_mask.sum())
        focus_count = int(focus_mask.sum())
        base_count = int(base_mask.sum())
        if recovery_count == 0 or base_count == 0:
            raise ValueError(
                "recovery-weighted split must contain both groups: "
                f"base={base_count} recovery={recovery_count} "
                f"start={_RECOVERY_START_EPISODE}"
            )
        if _FOCUS_SAMPLE_PROB > 0 and focus_count == 0:
            raise ValueError(
                f"focus-weighted split has no focus episodes: start={_FOCUS_START_EPISODE}"
            )
        sample_weights = np.empty(len(train_indices), dtype=np.float64)
        sample_weights[base_mask] = (
            1.0 - _RECOVERY_SAMPLE_PROB - _FOCUS_SAMPLE_PROB
        ) / base_count
        sample_weights[recovery_mask] = _RECOVERY_SAMPLE_PROB / recovery_count
        if _FOCUS_SAMPLE_PROB > 0:
            sample_weights[focus_mask] = _FOCUS_SAMPLE_PROB / focus_count
        train_sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_indices),
            replacement=True,
        )
        print(
            "Recovery episode sampling "
            f"prob={_RECOVERY_SAMPLE_PROB} start={_RECOVERY_START_EPISODE} "
            f"focus_prob={_FOCUS_SAMPLE_PROB} focus_start={_FOCUS_START_EPISODE} "
            f"train_base={base_count} train_recovery={recovery_count} "
            f"train_focus={focus_count}"
        )
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=4,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=4,
        persistent_workers=True,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
