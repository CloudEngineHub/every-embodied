# FCloud OmniBot 2026 Reproducibility Archive

这是 Datawhale［数据鲸］组织发布的 FCloud OmniBot 2026 赛后公开归档。内容包括教程对应的轻量代码、脱敏实验记录、代表性视频和文件校验清单。

## 证据边界

- SIMPLE［合成数据与状态机框架］视频：G1 全身弯腰抓取放置的物理仿真教师回合，作为状态机、慢速运输和多视角记录参考。
- MuJoCo［物理仿真器］视频：KUKA iiwa14 + Robotiq 2F85［两指夹爪］抓取橙子并放到盘子的真实接触回合，作为最小接触验收案例。
- FCloud［比赛平台］、CAPX/B1K［家庭环境基线］和 Unitree［宇树官方环境］材料：保留可复现的接口、日志摘要和失败边界，不声称存在已核验的完整比赛高分回合。

`oracle`［真值教师］、`bootstrap`［启动验证］、`smoke`［冒烟测试］和诊断数据不应被当作正式比赛成绩或视觉策略数据。

## 目录

```text
README.md
MANIFEST.sha256
source/                         # 脱敏后的轻量代码和配置
notes/                          # 公开事实、路线和接口笔记
evidence/mujoco_fruits/         # 真实接触水果抓取视频和摘要
evidence/simple/                # SIMPLE G1 多视角成功视频和摘要
```

下载前请阅读 GitHub 教程中的 [发布与清理说明](https://github.com/datawhalechina/every-embodied/tree/main/15-Challenge%E7%AB%9E%E8%B5%9B/FCloud_OmniBot_2026)。完整 Isaac Sim［机器人仿真器］资产、GPU［图形处理器］驱动、环境缓存、模型权重和第三方受限资源不包含在本归档中。

## 许可与引用

教程和原创轻量代码遵循 `datawhalechina/every-embodied` 的许可证；第三方项目、机器人资产、模型和数据集继续遵循各自上游许可证。复用时请同时引用比赛官网、SIMPLE、Unitree、CAPX/B1K 和 MuJoCo 水果样例的上游项目。

