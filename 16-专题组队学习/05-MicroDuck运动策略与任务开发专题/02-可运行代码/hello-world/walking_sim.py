"""A small, inspectable ONNX-to-MuJoCo velocity-control example."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.request

import numpy as np


MODEL_REVISION = "ad9d41a763f30dc770817c33d92dcfec0205db1e"
MODEL_SHA256 = "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c"
MODEL_URL = (
    "https://huggingface.co/spaces/pollen-robotics/microduck-simulator/resolve/"
    f"{MODEL_REVISION}/app/public/policies/BEST_alpha_walking.onnx"
)
OBSERVATION_NAMES = (
    "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
    "command", "head_command", "body_command",
)
CONTROL_DECIMATION = 4
TORQUE_LIMIT_NM = 0.6405236195572268


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_policy(cache=None, proxy=None):
    cache = Path(cache or Path.home() / ".cache/microduck-learning").expanduser()
    target = cache / "BEST_alpha_walking.onnx"
    if target.is_file() and sha256(target) == MODEL_SHA256:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    proxy = proxy or os.getenv("MICRODUCK_DOWNLOAD_PROXY")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(
        {"http": proxy, "https": proxy} if proxy else None))
    temporary = None
    try:
        with opener.open(MODEL_URL, timeout=90) as response:
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
                temporary = Path(stream.name)
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        if sha256(temporary) != MODEL_SHA256:
            raise ValueError("Walking model SHA-256 mismatch")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def bounded_command(vx=0.0, vy=0.0, yaw_rate=0.0):
    values = np.asarray([vx, vy, yaw_rate], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Velocity commands must be finite")
    return np.clip(values, [-0.3, -0.2, -1.5], [0.3, 0.2, 1.5]).astype(np.float32)


class WalkingSimulation:
    """Keep one physical scene alive across notebook command cells."""

    def __init__(self, playground, policy_path, *, width=640, height=360):
        import mujoco
        import onnxruntime as ort

        self.mujoco = mujoco
        scene = Path(playground) / "src/mjlab_microduck/robot/microduck/scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
        self.model.opt.timestep = 0.005
        self.model.actuator_forcerange[:, 0] = -TORQUE_LIMIT_NM
        self.model.actuator_forcerange[:, 1] = TORQUE_LIMIT_NM
        self.model.actuator_forcelimited[:] = 1
        self.data = mujoco.MjData(self.model)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(policy_path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 61] or inputs[0].type != "tensor(float)":
            raise ValueError("This example requires the walking actor with float32 obs[1,61]")
        if len(outputs) != 1 or outputs[0].shape != [1, 14] or outputs[0].type != "tensor(float)":
            raise ValueError("This example requires float32 action[1,14]")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        if tuple(meta.get("observation_names", "").split(",")) != OBSERVATION_NAMES:
            raise ValueError("Unsupported observation layout; use the default walking model")
        if meta.get("command_names") != "twist,head_pose,body_pose":
            raise ValueError("Unsupported command layout")
        self.default_pose = np.asarray([float(x) for x in meta["default_joint_pos"].split(",")],
                                       dtype=np.float32)
        self.action_scale = float(meta["action_scale"])
        if self.default_pose.shape != (14,) or not np.isfinite(self.default_pose).all():
            raise ValueError("Invalid default joint positions")
        if not math.isfinite(self.action_scale) or self.action_scale <= 0:
            raise ValueError("Invalid action scale")
        joints = self.model.actuator_trnid[:, 0]
        names = [self.model.joint(int(j)).name for j in joints]
        if self.model.nu != 14 or names != meta["joint_names"].split(","):
            raise ValueError("Policy joint order differs from the simulator actuator order")
        self.joint_qpos = self.model.jnt_qposadr[joints]
        self.joint_qvel = self.model.jnt_dofadr[joints]
        self.trunk = self.model.body("trunk_base").id
        self.sensor = self.model.sensor("imu_ang_vel").id
        if self.model.sensor_dim[self.sensor] != 3:
            raise ValueError("Invalid gyro sensor")
        base_joint = self.model.joint("trunk_base_freejoint").id
        self.base_qpos = int(self.model.jnt_qposadr[base_joint])
        self.base_qvel = int(self.model.jnt_dofadr[base_joint])
        self.command = np.zeros(13, dtype=np.float32)
        self.last_action = np.zeros(14, dtype=np.float32)
        self.control_dt = CONTROL_DECIMATION * self.model.opt.timestep
        self.renderer = None
        self.width, self.height = width, height
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 0.8, 125, -22
        self.reset()

    def reset(self):
        self.mujoco.mj_resetData(self.model, self.data)
        self.command.fill(0)
        self.last_action.fill(0)
        q = self.base_qpos
        self.data.qpos[q:q + 3] = [0, 0, 0.125]
        self.data.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        self.data.qpos[self.joint_qpos] = self.default_pose
        self.data.ctrl[:] = self.default_pose
        self.mujoco.mj_forward(self.model, self.data)

    def set_command(self, vx=0.0, vy=0.0, yaw_rate=0.0):
        self.command[:3] = bounded_command(vx, vy, yaw_rate)
        return self.command[:3].copy()

    def forward(self, speed=0.30):
        return self.set_command(vx=speed)

    def turn(self, yaw_rate=0.50):
        return self.set_command(yaw_rate=yaw_rate)

    def stop(self):
        return self.set_command()

    def observation(self):
        address = int(self.model.sensor_adr[self.sensor])
        rotation = self.data.xmat[self.trunk].reshape(3, 3)
        obs = np.concatenate((
            self.data.sensordata[address:address + 3],
            rotation.T @ np.asarray([0, 0, -1]),
            self.data.qpos[self.joint_qpos] - self.default_pose,
            self.data.qvel[self.joint_qvel], self.last_action, self.command,
        )).astype(np.float32).reshape(1, 61)
        if not np.isfinite(obs).all():
            raise RuntimeError("Non-finite simulator observation")
        return obs

    def apply_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (1, 14) or not np.isfinite(action).all():
            raise ValueError("Expected finite action[1,14]")
        target = self.default_pose + self.action_scale * action[0]
        self.data.ctrl[:] = target
        self.last_action[:] = action[0]
        return target.copy()

    def physics_step(self):
        for _ in range(CONTROL_DECIMATION):
            self.mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Non-finite physical state")

    def state(self):
        rotation = self.data.xmat[self.trunk].reshape(3, 3)
        velocity_body = rotation.T @ self.data.qvel[self.base_qvel:self.base_qvel + 3]
        return {
            "time_s": float(self.data.time),
            "x_m": float(self.data.qpos[self.base_qpos]),
            "y_m": float(self.data.qpos[self.base_qpos + 1]),
            "height_m": float(self.data.qpos[self.base_qpos + 2]),
            "yaw_rad": float(math.atan2(rotation[1, 0], rotation[0, 0])),
            "vx_m_s": float(velocity_body[0]),
            "upright_cos": float(rotation[2, 2]),
        }

    def render(self):
        if self.renderer is None:
            self.renderer = self.mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.camera.lookat[:] = self.data.xpos[self.trunk]
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def save_gif(video, output):
    import imageio_ffmpeg

    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(video),
        "-filter_complex",
        "fps=8,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=3",
        "-loop", "0", str(output),
    ], check=True, timeout=120)
    return Path(output)


def save_report(path, trace, policy_path):
    by_phase = {}
    for phase in dict.fromkeys(row["phase"] for row in trace):
        rows = [row for row in trace if row["phase"] == phase]
        yaw = np.unwrap([row["yaw_rad"] for row in rows])
        by_phase[phase] = {
            "steps": len(rows), "command": rows[0]["command"],
            "dx_m": rows[-1]["x_m"] - rows[0]["x_m"],
            "dy_m": rows[-1]["y_m"] - rows[0]["y_m"],
            "yaw_change_rad": float(yaw[-1] - yaw[0]),
            "last_second_mean_abs_vx_m_s": float(np.mean([abs(row["vx_m_s"]) for row in rows[-50:]])),
        }
    report = {
        "backend": "ONNX Runtime CPU -> MuJoCo position actuators",
        "model_revision": MODEL_REVISION, "model_sha256": sha256(policy_path),
        "control_frequency_hz": 50, "physics_frequency_hz": 200,
        "steps": len(trace), "phases": by_phase,
        "min_height_m": min(row["height_m"] for row in trace),
        "min_upright_cos": min(row["upright_cos"] for row in trace),
        "final_state": trace[-1], "trace": trace,
    }
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
