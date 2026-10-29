# BladeMaster 与 DexLab：柔体切割仿真入门

切一块豆腐状软体，再撤刀并推开切下的部分，看似简单，却要求仿真器同时处理材料变形、拓扑分离和切面接触。本章从 BladeMaster 的切割机制出发，介绍 DexLab 的开源 CPU 实现及香蕉、豆腐实验入口。

## 1. 从视频看到什么

[小红书演示](https://www.xiaohongshu.com/explore/6ad0dd96000000001c02fb40)展示了刀具切割香蕉与豆腐状材料。[DexLab 香蕉实验页面](https://huangkiki.github.io/Dexlab/zh-cn/latest/banana-cutting.html)提供表面与粒子视图，连续三刀将材料分为四段；[豆腐实验页面](https://huangkiki.github.io/Dexlab/zh-cn/latest/cutting.html)展示下刀、撤刀，以及用单侧诊断冲量检验切后材料能否独立运动。颜色用于区分切割历史，判断分离还要结合连接关系和运动记录。

香蕉的几何和材料参数是合成设置，尚未针对真实香蕉标定；该实验不包含机械臂或学习策略。小红书作者介绍其实现参考了论文中的 TLMPM 和持久切割历史，属于独立编写的 CPU 简化实现。原论文 BladeMaster 使用 GPU 加速，两者的性能和实现范围不能直接等同。

## 2. 切口为什么需要记忆

传统粒子与网格之间的传递若跨越切面，撤刀后两侧材料可能再次被数值耦合。BladeMaster 使用全拉格朗日物质点法（TLMPM），根据刀具几何在线为材料点记录持久侧别标签。这些标签控制粒子与网格间的耦合：仍相连的材料保留传递，切面两侧避免重新连接。切面之间还需要接触处理，使它们能够再次接触和滑动；刀具与材料的双向耦合则让材料反力影响刀具运动。

```mermaid
flowchart LR
    A[刀具轨迹与材料粒子] --> B[更新切割侧别与历史]
    B --> C[按相容关系传递粒子与网格动量]
    C --> D[材料接触与刀具反力]
    D --> E[保存状态并检查切后运动]
```

原论文支持渐进和交叉切割；具体复现项目的支持范围应以其对应版本实验页面为准。DexLab 的豆腐基线用有限直刀和持久标签，结合动态刀具、接触反力和独立评分器；其重建表面是显示近似。

## 3. 阅读和运行实验

先打开 [DexLab 仓库](https://github.com/huangkiki/Dexlab)及其[香蕉案例说明](https://github.com/huangkiki/Dexlab/tree/main/demos/banana-cut)，对照粒子视图观察三次切割后的分段。豆腐案例的[完整实验页](https://huangkiki.github.io/Dexlab/zh-cn/latest/cutting.html)给出了环境依赖、资源限额、运行命令和验收阈值。运行前按该页面配置 DexLab 环境、共享锁和数据目录；输出目录需要是新目录。

实验会保存物理状态、轨迹及评分结果。判断是否完成切割时，查看 `score.json` 和评分器退出码，而不只看动画或运行器是否退出成功。豆腐基线还提供部分切割、未切割和时间步敏感性案例；它们帮助区分真实的材料分离和视觉上的切缝。项目报告的时间步敏感性结果显示反力峰值发生明显变化，尚未建立切割力收敛，因此这些结果适合学习机制和数值检查，而不能直接当作真实食材的力标定。

## 4. 延伸阅读

- [BladeMaster 项目页](https://jango6324.github.io/blademaster/)和[论文](https://arxiv.org/abs/2609.27342)：原始方法、切割和操作演示。
- [DexLab 豆腐实验](https://huangkiki.github.io/Dexlab/zh-cn/latest/cutting.html)：CPU 实现、参数、对照和复现命令。
- [DexLab 香蕉实验](https://huangkiki.github.io/Dexlab/zh-cn/latest/banana-cutting.html)：连续三刀及可切换的表面、粒子回放。
