# Robot Data Conversion Guide

## Overview

This guide demonstrates how to convert robot data into the [ LeRobot dataset V2.0 format ](https://github.com/huggingface/lerobot?tab=readme-ov-file#the-lerobotdataset-format)——`GR00T LeRobot` that suits us. Although we have added additional structures, our schema remains fully compatible with the upstream LeRobot 2.0. These additional metadata and structures enable more detailed specification and linguistic annotation of the robot data.

## Requirements

### Core Requirements

The folder should follow a structure similar to the following, and include these core folders and files:

```
.
├─meta
│ ├─episodes.jsonl
│ ├─modality.json # -> GR00T LeRobot特有
│ ├─info.json
│ └─tasks.jsonl
├─videos
│ └─chunk-000
│   └─observation.images.ego_view
│     └─episode_000001.mp4
│     └─episode_000000.mp4
└─data
  └─chunk-000
    ├─episode_000001.parquet
    └─episode_000000.parquet
```

### Video observation (video/chunk-*)
The video folder will contain mp4 files related to each episode, following the naming format episode_00000X.mp4, where X represents the episode number.
**Requirements**:
- It must be stored in MP4 format.
- The name should use `observation.images.<video_name>`.


### Data (data/chunk-*)
The data folder will contain all parquet files related to each episode, following the naming format episode_00000X.parquet, where X represents the episode number.
Each parquet file will contain:
- State information: stored as observation.state, which is a one-dimensional connected array for all state modalities.
- Actions: stored as action, which is a one-dimensional connected array for all action modalities.
- Timestamps: stored as timestamp, which are floating-point numbers representing start times.
- Annotations: stored as annotation.<annotation_source>.<annotation_type>(.<annotation_name>) (see the annotation field in the example configuration for naming examples). Other columns should not use the annotation prefix. If you are interested in adding multiple annotations, please refer to (multiple-annotation-support).

#### Example Parquet File
The following is an example of the `robot_sim.PickNPlace` dataset in the [demo data directory](../../../../10-具身智能其他仿真工具及仿真前沿/Isaac-GR00T/demo_data/robot_sim.PickNPlace).
```
{
    "observation.state":[-0.01147082911843003,...,0], // 基于modality.json文件的连接状态数组
    "action":[-0.010770668025204974,...0], // 基于modality.json文件的连接动作数组
    "timestamp":0.04999995231628418, // 观察的时间戳
    "annotation.human.action.task_description":0, // meta/tasks.jsonl文件中任务描述的索引
    "task_index":0, // meta/tasks.jsonl文件中任务的索引
    "annotation.human.validity":1, // meta/tasks.jsonl文件中任务的索引
    "episode_index":0, // episode的索引
    "index":0, // 观察的索引。这是整个数据集中所有观察的全局索引。
    "next.reward":0, // 下一个观察的奖励
    "next.done":false // episode是否完成
}
```

### Metadata

- `episodes.jsonl` contains a list of all episodes in the entire dataset. Each episode includes a series of tasks and the length of the episode.
- `tasks.jsonl` contains a list of all tasks in the entire dataset.
- `modality.json` contains modal configurations.
- `info.json` contains dataset information.

#### meta/tasks.jsonl
Here is an example of a `meta/tasks.jsonl` file containing task descriptions.
```
{"task_index": 0, "task": "pick the squash from the counter and place it in the plate"}
{"task_index": 1, "task": "valid"}
```

The task index can be referenced in a parquet file to obtain the task description. In this case, the first observed `annotation.human.action.task_description` is "pick the squash from the counter and place it in the plate", and `annotation.human.validity` is "valid".

`tasks.json` contains a list of all tasks in the entire dataset.

#### meta/episodes.jsonl

Here is an example of a `meta/episodes.jsonl` file containing episode information.

```
{"episode_index": 0, "tasks": [...], "length": 416}
{"episode_index": 1, "tasks": [...], "length": 470}
```

`episodes.json` contains a list of all episodes in the entire dataset. Each episode includes a series of tasks and the length of the episode.


#### `meta/modality.json` Configuration

This file provides detailed metadata on status and action modes, enabling the following functions:

- **Separation of data storage and interpretation:**
  - **State and actions:** Stored as a connected float32 array. The `modality.json` file provides the metadata required to interpret these arrays into different, fine-grained fields with additional training information.
  - **Video:** Stored as separate files, and the configuration file allows them to be renamed to a standardized format.
  - **Comments:** All comment fields are tracked. If there are no comments, do not include the `annotation` field in the configuration file.
- **Fine-grained segmentation:** Divide the state and actions arrays into more semantically meaningful fields.
- **Clear mapping:** Clear mapping of data dimensions.
- **Complex data transformation:** Support for normalization and rotation transformations of specific fields during training.

##### Mode

```json
{
    "state": {
        "<state_key>": {
            "start": <int>,         // 状态数组中的起始索引
            "end": <int>,           // 状态数组中的结束索引
            "rotation_type": <str>,  // 可选：指定旋转格式
            "dtype": <str>,         // 可选：指定数据类型
            "range": <tuple[float, float]>, // 可选：指定模态的范围
        }
    },
    "action": {
        "<action_key>": {
            "start": <int>,         // 动作数组中的起始索引
            "end": <int>,           // 动作数组中的结束索引
            "absolute": <bool>,      // 可选：true表示绝对值，false表示相对/增量值
            "rotation_type": <str>,  // 可选：指定旋转格式
            "dtype": <str>,         // 可选：指定数据类型
            "range": <tuple[float, float]>, // 可选：指定模态的范围
        }
    },
    "video": {
        "<new_key>": {
            "original_key": "<original_video_key>"
        }
    },
    "annotation": {
        "<annotation_key>": {}  // 空字典，保持与其他模态的一致性
    }
}
```

**Supported rotation types:**

- `axis_angle`
- `quaternion`
- `rotation_6d`
- `matrix`
- `euler_angles_rpy`
- `euler_angles_ryp`
- `euler_angles_pry`
- `euler_angles_pyr`
- `euler_angles_yrp`
- `euler_angles_ypr`

##### Example Configuration

```json
{
    "state": {
        "left_arm": { // parquet文件中observation.state数组的前7个元素是左臂
            "start": 0,
            "end": 7
        },
        "left_hand": { // parquet文件中observation.state数组的接下来6个元素是左手
            "start": 7,
            "end": 13
        },
        "left_leg": {
            "start": 13,
            "end": 19
        },
        "neck": {
            "start": 19,
            "end": 22
        },
        "right_arm": {
            "start": 22,
            "end": 29
        },
        "right_hand": {
            "start": 29,
            "end": 35
        },
        "right_leg": {
            "start": 35,
            "end": 41
        },
        "waist": {
            "start": 41,
            "end": 44
        }
    },
    "action": {
        "left_arm": {
            "start": 0,
            "end": 7
        },
        "left_hand": {
            "start": 7,
            "end": 13
        },
        "left_leg": {
            "start": 13,
            "end": 19
        },
        "neck": {
            "start": 19,
            "end": 22
        },
        "right_arm": {
            "start": 22,
            "end": 29
        },
        "right_hand": {
            "start": 29,
            "end": 35
        },
        "right_leg": {
            "start": 35,
            "end": 41
        },
        "waist": {
            "start": 41,
            "end": 44
        }
    },
    "video": {
        "ego_view": { // 视频存储在videos/chunk-*/observation.images.ego_view/episode_00000X.mp4中
            "original_key": "observation.images.ego_view"
        }
    },
    "annotation": {
        "human.action.task_description": {}, // 任务描述存储在meta/tasks.jsonl文件中
        "human.validity": {}
    }
}
```

### Multiple annotation support

To support multiple annotations in a single parquet file, users can add additional columns to the parquet file. Users should handle these columns similarly to how they handled the `task_index` column in the original LeRobot V2 dataset:

In LeRobot V2, the actual language descriptions are stored in a line of `meta/tasks.jsonl` files, while the parquet file only stores the corresponding index in `task_index` column. We follow the same convention and store the corresponding index of each annotation in `annotation.<annotation_source>.<annotation_type>` column. Although `task_index` column can still be used for default annotations, a dedicated column `annotation.<annotation_source>.<annotation_type>` is required to ensure it can be loaded by our custom data loader.

### GR00T LeRobot: An Extension of Standard LeRobot
GR00T LeRobot is a variant of the standard LeRobot format with additional fixed requirements:
- The standard LeRobot format uses meta/stats.json, but our data loader does not require it. If computation is too time-consuming, this file can be safely ignored.
- The "observation.state" key must always contain the proprioceptive state.
- We support multi-channel annotation formats (e.g., coarse-grained, fine-grained) allowing users to add any number of annotation channels via the `annotation.<annotation_source>.<annotation_type>` key as needed.
- We require an additional metadata file `meta/modality.json` that does not exist in the standard LeRobot format.

#### Note

- Specify optional fields only when they differ from the default value.
- The video key mapping standardizes the camera names across the entire dataset.
- All indices start at zero and follow Python array slicing conventions `[start:end]`.

## Example

Refer to the [dataset example](../../../../10-具身智能其他仿真工具及仿真前沿/Isaac-GR00T/demo_data/robot_sim.PickNPlace) for a complete reference.
