# 公开素材说明

GitHub 只保留素材说明；脱敏后的仿真预览视频和截图放在对应的 Datawhale Hugging Face 数据集 `TaskB-B2Piper/videos/` 与 `TaskB-B2Piper/frames/` 中。

推荐公开视频至少包含头部视角、末端视角和第三方环境视角，并用同一 `episode_id` 或仿真时间戳同步。环境视角必须在视频说明中标注为调试视角，不得让读者误以为它是策略输入。

当前媒体均为本地 debug/evaluation preview，不是官方排行榜证明；其中 top-down/global 视角使用了 privileged trace，只用于解释失败原因。
