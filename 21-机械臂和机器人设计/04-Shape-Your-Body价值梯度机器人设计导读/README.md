# Shape Your Body：用价值梯度优化机器人身体设计

> 本文导读 **Shape Your Body: Value Gradients for Multi-Embodiment Robot Design**。这篇工作来自 Technical University of Darmstadt、Robotics Institute Germany、DFKI 和 hessian.AI，作者为 Nico Bohlinger 与 Jan Peters。论文 arXiv 编号为 `2606.00702v1`，2026 年 5 月 30 日提交。项目页地址是 [nico-bohlinger.github.io/shape-your-body](https://nico-bohlinger.github.io/shape-your-body/)。

## 1. 如何用价值梯度优化机器人身体

Shape Your Body 研究的是一个很有想象力、也很工程的问题：

```text
能不能像优化神经网络参数一样，用梯度优化机器人身体参数？
```

传统机器人设计通常是人工设计机械结构，再为这个结构训练控制策略。学习式 co-design 会把身体和控制器一起优化，但常见做法代价很高：外层搜索机器人身体，内层为每个候选身体重新训练或适配控制器。只要设计空间稍微大一点，比如质量、惯量、几何尺寸、关节限位、执行器力矩、速度上限、PD 增益都可调，这个循环就会非常慢。

Shape Your Body 的思路是把设计成本摊薄：

```text
先训练一个多具身 policy + value function
    ↓
冻结 policy 和 critic
    ↓
把 critic 当成可微分的设计代理模型
    ↓
对机器人身体参数求 value gradient
    ↓
用梯度更新新的机器人设计
```

论文把这个设计搜索方法叫 **Value-Gradient Design Search (VGDS)**。一句话概括：

> 先用多机器人强化学习学会“什么身体更容易被当前策略控制好”，再用价值函数对身体参数的梯度来优化新的机器人设计。

这篇论文研究的对象不是感知、动作生成或视频预测，而是 **robot embodiment / morphology / continuous design parameters**。Build123d、Text-to-CAD 与 ForgeCAD 更关注如何生成或编辑几何，Shape Your Body 则研究如何用学习到的控制价值反向指导身体参数。

## 2. 论文、项目页和开源状态

| 条目 | 链接 | 当前状态 |
| :--- | :--- | :--- |
| 论文 | [arXiv:2606.00702](https://arxiv.org/abs/2606.00702) / [HTML](https://arxiv.org/html/2606.00702v1) | 2026-05-30 提交 |
| 项目页 | [Shape Your Body](https://nico-bohlinger.github.io/shape-your-body/) | 官方项目页，包含交互 demo、方法图、机器人集合和结果图 |
| PDF | [TU Darmstadt PDF](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/NicoBohlinger/shape_your_body.pdf) | 项目页 Paper 指向的官方 PDF |
| 视频 | [YouTube](https://www.youtube.com/watch?v=_SHSWUSQZvg) | 项目页 Video 指向的视频 |
| 代码 | 项目页当前显示 `Code (soon)` | 还没有看到官方公开代码仓库入口 |

截至 2026 年 7 月 18 日，项目页、论文和交互式网页演示已经公开，可以体验不同机器人和设计搜索轨迹；官方代码入口仍标注为 `Code (soon)`。本章据此提供方法导读、图解和复现准备说明。官方代码公开后，可继续补充环境配置和最小复现实验。

## 3. 总览图：先训练价值函数，再用价值梯度塑形

![Shape Your Body method overview](assets/hero.png)

**图 1 Shape Your Body 方法总览。** 先在大量机器人身体变化上训练 embodiment-aware policy 和 value function；训练完成后冻结网络，对连续设计向量求价值梯度，并在信任域约束下更新机器人身体参数。  
来源：[Shape Your Body 官方项目页](https://nico-bohlinger.github.io/shape-your-body/)。

这张图可以拆成两段。

第一段是 **multi-embodiment RL training**。每个环境在 episode reset 时采样一个机器人设计，并在整个 episode 内保持这个设计不变。控制策略要学会在不同身体参数下完成速度跟踪任务，critic 要学会评估“某个状态 + 某个身体设计”未来能得到多少回报。

第二段是 **design search**。训练结束后，不再为每个新设计重新训练 policy。系统冻结 policy 和 critic，把 robot design 表示成归一化设计向量 `f`，通过可微分映射 `Φ(f)` 转成真实身体参数，再计算 frozen critic 对设计向量的梯度：

```text
design vector f
    -> physical embodiment Φ(f)
    -> critic value V(s, Φ(f))
    -> ∇f V
    -> update f
```

这就是“用梯度设计机器人”的核心。梯度不是来自真实硬件，也不是来自 CAD 软件的几何损失，而是来自训练好的 value function。它告诉优化器：在当前状态分布和当前冻结策略下，哪些身体参数变化预计会提高任务回报。

## 4. 训练集：50 个机器人，最高 1177 个连续设计参数

![Multi-embodiment training robots](assets/main_robots.png)

**图 2 多具身训练集。** 官方项目页展示最大训练集包含 50 个机器人：15 个四足、31 个双足/人形、4 个六足。每个机器人有 190 到 1177 个连续设计参数。  
来源：[Shape Your Body 官方项目页](https://nico-bohlinger.github.io/shape-your-body/)。

论文的训练设置不是只在一个小玩具机器人上调腿长，而是覆盖较大的 morphology 分布。最大训练集包含：

- `15` 个 quadrupeds；
- `31` 个 bipeds / humanoids；
- `4` 个 hexapods；
- 每个机器人有 `190` 到 `1177` 个 continuous design parameters。

这些参数不只包括几何尺寸，还包括更贴近控制性能的物理和控制参数：

| 参数类别 | 例子 | 为什么影响控制 |
| :--- | :--- | :--- |
| 几何参数 | link geometry、foot size、joint origins | 改变质心、支撑面、腿部摆动范围和碰撞关系 |
| 质量与惯量 | body mass、inertia、center-of-mass offset | 影响动态稳定性、加速度响应和能耗 |
| 关节属性 | joint axis、joint range、damping、friction | 决定运动自由度、阻尼和可达姿态 |
| 执行器属性 | actuator torque limits、velocity limits | 决定可输出力矩和速度上限 |
| 控制参数 | PD gains、action scale | 影响低层跟踪、稳定性和动作放大 |

这也是为什么它适合和机器人设计教程放在一起。CAD 只告诉我们“形状是什么”，URDF/MJCF 只告诉仿真器“身体参数是什么”，而 Shape Your Body 试图回答的是：这些身体参数如何影响一个已训练策略的长期回报。

## 5. URMA policy 和 direct-design critic

Shape Your Body 建立在 URMA 这类 multi-embodiment policy architecture 上。URMA 的目标是处理不同机器人拓扑：有的机器人 12 个关节，有的 20 多个关节，有的是四足，有的是人形，有的是六足。它不能简单把所有关节状态拼成固定长度向量。

URMA 的做法是把 observation 拆成两部分：

```text
general observations
    + per-joint observations
    + per-joint description vectors
```

每个 joint 都有自己的 observation 和 description。description 里编码静态身体信息，例如关节轴、相对位置、关节限位、PD 增益、相邻 body 质量、惯量和几何参数。网络对每个 joint 单独编码，再用 attention / aggregation 把可变数量的 joint latent 汇聚成固定维度表示。

Shape Your Body 的关键改动在 critic。论文把标准 URMA critic 改成 **direct-design URMA critic**，让设计参数更直接影响 value prediction。直观理解：

```text
普通 embodiment-aware critic:
    身体描述主要参与 attention key，设计信息影响不够直接。

direct-design critic:
    observation 和 description 一起进入 value-side encoder，
    让 critic 对设计参数更敏感，
    从而给出更有用的 ∇design V。
```

这一步非常重要。VGDS 依赖 critic 对设计参数的梯度。如果 critic 只会粗略区分机器人类型，不能对 foot size、joint range、actuator limit、PD gains 的微小变化做出有意义响应，那么 value gradient 就会噪声很大，设计搜索会失效。

论文还使用 ensemble value heads，通过多个 critic head 的 mean value 做优化，降低单个 value approximation error 导致灾难性设计更新的风险。

## 6. VGDS：带软信任域的价值梯度搜索

项目页给出的 VGDS 目标可以写成：

```text
J_hat_lambda(f)
  = mean_m V_bar(s_m, Phi(f))
    - lambda * ||f - f_ref||^2 / d_design
```

其中：

- `f` 是 normalized design vector；
- `Φ(f)` 是把归一化设计向量映射到真实物理参数的 design map；
- `s_m` 来自固定 state bank；
- `V_bar` 是 frozen critic ensemble 的平均价值预测；
- `f_ref` 是 reference design，可以是 nominal URDF，也可以是一个初始设计；
- `lambda` 控制离参考设计多远；
- `d_design` 用来归一化不同维度设计空间的惩罚强度。

为什么需要 trust region？因为 critic 不是物理真理，它只在训练分布附近相对可靠。如果无约束最大化 value，优化器可能把设计推到边界，比如极端质量、极端几何尺寸、极端 actuator limit，让 critic 误以为回报很高，但真实 rollout 崩掉。论文也提到，无约束 critic maximization 容易走向设计空间边界，导致真实性能下降。

因此 VGDS 每步做三件事：

1. 在 state bank minibatch 上计算 critic mean value。
2. 对 `f` 求梯度，同时加上 reference design 的软信任域惩罚。
3. 用 Adam 上升一步，并裁剪单步更新和最终设计范围。

用伪代码表示：

```text
initialize f from a reference or perturbed design
for iter in 1..N:
    sample states from state bank
    objective = critic_value(states, Phi(f)) - trust_region_penalty(f, f_ref)
    grad = d objective / d f
    f = f + clipped_adam_step(grad)
    f = clip(f, valid_design_bounds)
return f
```

这就是它和 CMA-ES、PSO、Bayesian Optimization 等黑盒优化方法的根本区别。黑盒方法只看“某个设计得分是多少”，VGDS 还利用“往哪个设计方向移动会更好”的梯度信息，因此在高维连续设计空间里更有优势。

## 7. 单机器人设计：和强黑盒优化器对比

![Single-robot design results](assets/e1_ranking_return.png)

**图 3 单机器人设计结果。** 在 Unitree Go2、MIT Humanoid 和 Golem 等目标机器人上，VGDS 从 perturbed design 出发，优化后的 return improvement 与 BO、CMA-ES、PSO、CEM、DE、ARS、TuRBO、GC-PFO 等强黑盒优化器对比。  
来源：[Shape Your Body 官方项目页](https://nico-bohlinger.github.io/shape-your-body/)。

第一组实验是 single-robot design。做法是：先针对某个目标机器人训练一个 policy 和 critic，再从这个机器人被扰动的设计出发，用不同优化器恢复或改进设计。这里的 nominal URDF 只是 trust region 的参考锚点，不是最终必须回到的设计。

它对比的不是弱 baseline，而是一批常见高维优化方法：

- Bayesian Optimization；
- CMA-ES；
- Particle Swarm Optimization；
- Cross-Entropy Method；
- Differential Evolution；
- Augmented Random Search；
- TuRBO；
- Gradient-Covariance Particle Filter Optimization。

项目页总结是：VGDS matches or outperforms strong surrogate-search baselines。这个结论的含义不是“梯度设计永远比进化算法好”，而是在这类任务里，trained critic 的解析梯度确实能为高维机器人参数搜索提供有效方向。

## 8. 和 RL-based co-design 的成本差异

![VGDS vs RL co-design baselines](assets/rl_vs_frozen_combined.png)

**图 4 VGDS 与 RL-based co-design 对比。** RL co-design baseline 需要为每个初始设计重新训练或适配策略；VGDS 在一次 7-9 小时训练后，每个新增设计只需约 1-2 分钟搜索。  
来源：[Shape Your Body 官方项目页](https://nico-bohlinger.github.io/shape-your-body/)。

这张图是 Shape Your Body 最值得强调的工程点。很多 co-design 方法也能找到不错设计，但代价是“每来一个初始设计，就重新跑一轮策略训练”。如果目标是给几十个或几百个机器人变体做设计搜索，这个成本很难接受。

VGDS 的计算模式不同：

```text
一次多具身 RL 训练：7-9 小时
    ↓
冻结 policy + critic
    ↓
每个新增设计：1-2 分钟 value-gradient search
```

这就是 amortized robot design。前期训练贵，但训练出的 value function 可以反复复用。对于探索很多候选机器人、批量扰动设计、做设计敏感性分析，这种方式比“每个身体重新训练策略”更适合扩展。

当然，这并不意味着训练成本消失了。它只是把训练成本从“每个设计都付一次”变成“先付一次大成本，然后为很多设计摊薄”。如果目标机器人分布和训练分布差得太远，frozen critic 的梯度可能不可信，仍然需要重新训练或扩展多具身训练集。

## 9. 跨机器人泛化：held-out robot 和 50-robot training

Shape Your Body 还测试了更难的设置：不是只在同一个机器人上做扰动恢复，而是把目标机器人从训练集里 held out，再看多具身 critic 是否能指导它的设计优化。

项目页总结是：使用 full 50-robot set 训练的模型，可以让 Unitree Go2、MIT Humanoid 和 Golem 等目标机器人超过 nominal URDF；但对 Golem 这类六足机器人，hexapod-only model 表现最好，原因可能是 full 50-robot set 主要由 quadruped 和 biped/humanoid 组成，六足样本较少。

这说明多具身价值函数的泛化不是无条件的。它需要训练分布覆盖目标 morphology。若训练集里大多是四足和人形，用它去优化六足、蛇形、轮腿或机械臂结构，critic 可能会外推过远。实际应用时要问：

- 目标机器人的拓扑是否在训练分布附近？
- 设计参数范围是否在训练 curriculum 覆盖内？
- critic 的 high-value design 是否真的能通过 rollout 验证？
- 信任域是否足够限制不可信外推？

## 10. 价值梯度还能做设计诊断

![Design analysis heatmaps](assets/design_analysis_heatmaps.png)

**图 5 价值梯度设计分析。** 论文把 optimized design 与 reference design 的差异按 body part 和 parameter type 分组，形成热力图，帮助工程师定位哪些身体参数最影响回报。  
来源：[Shape Your Body 官方项目页](https://nico-bohlinger.github.io/shape-your-body/)。

Shape Your Body 还有一个很实用的副产品：value gradient 不只能产生一个优化后的设计，也能解释“哪些参数值得工程师重点看”。

项目页给了几个例子：

- MIT Humanoid 的强变化集中在 nominal joint positions、PD gains 和 reduced foot size。
- Golem 的变化集中在 reduced action scale、lower P gain 和 higher D gain。
- Unitree Go2 的变化更物理：rear-leg joint axes、foot geometry、front hip/calf actuator velocity limits。

这类分析对机器人设计很有价值。因为高维优化器给出的最终设计可能包含上千个参数变化，工程师很难直接理解。把 `f* - f_ref` 按 body part 和 parameter type 分组，可以从“优化结果”回到“设计 insight”：

```text
不是只问：最终设计是什么？
还要问：优化器认为哪个身体部位、哪类参数限制了性能？
```

这使 VGDS 不只是自动搜索工具，也可以作为机械设计和控制调参的诊断工具。

## 11. 复现边界：现在能做什么，暂时不能做什么

当前项目页没有公开代码仓库，因此本节把学习路径分为三个层级：先理解价值梯度设计，再复现通用优化接口，最后在官方代码发布后核对论文结果。

第一档是 **方法阅读与交互 demo**。项目页提供浏览器 demo，可以选择 Unitree Go2、MIT Humanoid、Golem、ANYmal C、Booster T1、Mini PI、Fourier GR1-T2 等机器人，切换 Reference / Co-Design，查看 VGDS 迭代如何改变身体，并用速度命令控制 policy。这个 demo 适合理解“optimized design 不是静态图片，而是仍要配合 frozen policy 运动”。

第二档是 **复现思路设计**。在代码公开前，可以先用自己熟悉的 MJX / MuJoCo / Isaac Lab 任务搭一个缩小版：

```text
选择一个机器人
定义少量连续设计参数：腿长、脚尺寸、质量、PD gain、actuator limit
在参数随机化下训练 embodiment-aware policy + critic
冻结 critic
对设计参数求 ∇design V
用 rollout 验证 critic 推荐设计是否真的更好
```

这个版本不能证明 50-robot 泛化能力，但能帮助理解 value-gradient design 的基本闭环。

第三档是 **完整复现**。等官方代码公开后，需要重点关注这些模块：

- 多具身机器人模型集合和设计参数定义；
- normalized design vector 到 MJCF/URDF/MJX 参数的 differentiable design map；
- URMA policy / direct-design critic 实现；
- PPO 多具身训练 curriculum；
- state bank 的收集方式；
- VGDS optimizer 的 trust-region penalty、Adam step、clip 策略；
- rollout 验证和 black-box baseline 对齐方式。

最关键的坑会在 differentiable design map。几何、质量、惯量、关节限位、执行器限制、PD 增益等参数必须能从归一化向量稳定映射到仿真器可接受的身体参数，同时要避免负质量、不可制造尺寸、重叠 body、关节限位不合理等问题。

## 12. 和现有机器人设计教程的关系

| 教程 | 关注点 | Shape Your Body 的补充 |
| :--- | :--- | :--- |
| Build123d | 用 Python 参数化生成 CAD 几何 | 解决“形状如何写成参数化源码” |
| Text-to-CAD | 用代码智能体生成和修改 CAD | 解决“自然语言/参考图如何进入 CAD 工程流” |
| ForgeCAD | 用 code-first CAD 复刻复杂装配和灵巧手 | 解决“复杂装配如何可视化、导出、检查” |
| Shape Your Body | 用 value gradient 优化身体参数 | 解决“哪些身体参数让策略更容易控制” |

如果把机器人设计链路串起来，可以理解成：

```text
CAD / robot description 给出候选身体
    -> 仿真器和控制器评估身体能否运动
    -> 多具身 critic 学会估计身体价值
    -> value gradient 反向指出身体应该怎么改
    -> 工程师再把可行设计落回 CAD / URDF / MJCF
```

Shape Your Body 不会替代机械工程师，也不会自动生成可制造机器人。它更像一个学习型设计建议器：告诉你在当前任务、当前控制策略和当前仿真分布下，哪些连续身体参数值得沿哪个方向调整。

## 13. 推荐组队学习题目

在组队学习里可以把它放到 **感知建图 / 机器人设计** 方向的 Task04，题目可以写成：

> 对比 Text-to-CAD、ForgeCAD 和 Shape Your Body：前两者解决如何生成可编辑几何，Shape Your Body 解决如何用控制价值反向指导身体参数。请说明 CAD 参数、URDF/MJCF 参数、仿真控制参数和 value-gradient design parameter 之间的关系。

交付可以不跑实验，但要回答：

- 为什么传统 robot co-design 通常很贵？
- multi-embodiment policy 和 value function 分别起什么作用？
- direct-design critic 为什么比普通 critic 更适合求设计梯度？
- VGDS 为什么需要 soft trust region？
- 价值梯度给出的设计变化怎样转化成工程师可理解的 body-part / parameter-type 诊断？
- 在官方代码尚未公开时，完整复现需要先补齐哪些模块？

## 14. 资料来源

本文图片来自 [Shape Your Body 官方项目页](https://nico-bohlinger.github.io/shape-your-body/)，为了教程稳定访问已保存到本地 `assets/`。文字内容参考 [arXiv:2606.00702](https://arxiv.org/abs/2606.00702)、[arXiv HTML](https://arxiv.org/html/2606.00702v1)、[官方 PDF](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/NicoBohlinger/shape_your_body.pdf) 和项目页。
