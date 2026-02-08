# Eureka (VisFly-Eureka) vs LaRes 对比分析

## 1. 概述

| 项目 | **我们的 Eureka (VisFly-Eureka)** | **LaRes** (NeurIPS 2025) |
|------|-----------------------------------|---------------------------|
| 定位 | LLM 驱动的奖励优化 + VisFly 四旋翼 | 进化强化学习 + LLM 自适应奖励搜索 |
| 环境 | VisFly（hover, navigation, landing 等） | MetaWorld（机械臂：关窗、按按钮等） |
| RL 算法 | BPTT / SHAC / PPO（依任务） | SAC（off-policy） |
| 奖励接口 | `get_reward(self) -> Tensor`（注入到 env） | `compute_reward(...) -> (reward, reward_component_dict)`（传参调用） |

---

## 2. 核心流程对比

### 2.1 Eureka（我们）：迭代式「生成 → 全量训练 → 排序 → 反馈」

```
每轮 (iteration):
  1. LLM 根据任务描述 + 环境上下文 + 上一轮反馈 → 生成 N 个奖励函数（如 5 个）
  2. 对每个奖励：在独立子进程中注入到环境 → 从头训练策略（如 BPTT 到收敛）
  3. 评估：每个样本得到 success_rate、episode_length、final_reward 等
  4. Agent Voter 或按 success_rate + episode_length 选 best
  5. 将 best 奖励代码 + TensorBoard 等反馈写入下一轮 prompt → 下一轮
```

- **数据**：每轮、每个样本独立训练，**不跨轮复用** transition；同一轮内多样本并行（多子进程）。
- **策略**：每个样本一个独立训练 run，**不保留种群**；只保留「当前 best 奖励代码」用于 prompt。

### 2.2 LaRes：进化式「种群 + 共享 buffer + 按步推进」

```
初始化: LLM 生成 pop_size 个奖励函数，每个对应一个 SAC agent（策略），共享一个 replay buffer

每步 (按环境步数推进):
  1. Thompson Sampling 选择「哪个策略」与环境交互 → 该策略跑一局，得到 (s, a, r_1, r_2, ..., r_pop, s', done)
     - 同一 transition 用当前所有奖励函数算 r_i，存入 buffer（reward_list）
  2. Workers 从 buffer 里按自己的 index 取 reward_list[index] 做 SAC 更新
  3. 每隔 LLM_freq 步（如 200k）：
     - 评估所有个体（成功率/回报），排序，保留 elite（如 top-3）
     - 非 elite 的奖励函数由 LLM 重新生成（基于 best 的反馈）
     - Reward scaling：新奖励的均值/方差对齐到 elite
     - Buffer 里所有 transition 用新奖励函数重新算 reward，更新 reward_list（reward relabeling）
     - 可选：用 best 策略的参数初始化其他 agent
```

- **数据**：**共享 buffer**；同一 transition 存多份 reward（每个奖励一个），换奖励后**重算并 scale**，实现 **reward relabeling**。
- **策略**：多种策略（每奖励一个 SAC），通过 Thompson Sampling 决定谁与环境交互；策略参数可周期性同步 best。

---

## 3. 差异对照表

| 维度 | **Eureka (我们)** | **LaRes** |
|------|-------------------|-----------|
| **优化粒度** | 按「轮」：每轮 N 个奖励各自完整训练一次 | 按「环境步数」：边交互边训练，周期性换奖励 |
| **样本效率** | 较低：每样本从头训练，无跨轮数据复用 | 较高：buffer 复用 + reward relabeling，同一批 transition 服务多奖励 |
| **策略结构** | 每样本独立策略，不保留种群；只保留 best 代码 | 种群：pop_size 个策略 + pop_size 个奖励，协同进化 |
| **探索-利用** | 无显式机制；依赖每轮多样本多样性 | Thompson Sampling 选策略交互，平衡探索/利用 |
| **奖励稳定性** | 无显式 scale；依赖 LLM 生成合理尺度 | Reward scaling：新奖励对齐 elite 的均值/方差，训练更稳 |
| **训练并行** | 子进程级：每样本一进程，多样本并行 | 多 Worker 进程：每个 agent 一 worker，异步更新 |
| **反馈给 LLM** | TensorBoard 曲线、success_rate、episode_length、轨迹等；Agent Voter 可选 | 成功率、回报、当前 best 的 output 等 |
| **初始化** | 无 human reward 必选；可选 temptuner（只调系数） | 可选 LaRes_with_init：用人类设计的奖励初始化 |

---

## 4. 适用场景

- **Eureka（我们）**  
  - 适合：VisFly 这类**需要梯度/特定算法（BPTT/SHAC）**的环境，**任务相对独立、每轮可接受完整训练**。  
  - 强调：**奖励函数结构随迭代进化**（如从简单四项到 proximity + acc），与「成功率/步数」提升一致；实现简单、与现有 VisFly 流程一致。

- **LaRes**  
  - 适合：MetaWorld 等 **off-policy（SAC）+ 高样本效率** 需求；强调 **sample efficiency** 和 **ERL 中的 SOTA**。  
  - 依赖：共享 buffer、reward relabeling、Thompson Sampling、reward scaling 等一整套设计。

---

## 5. 若要在我们框架里借鉴 LaRes 的思路

- **Reward relabeling**：我们当前是「一个奖励 → 一次独立训练」，没有共享 buffer。若未来在 VisFly 里用 off-policy 算法（如 SAC），可考虑：  
  - 存 (s, a, s')，对多份奖励分别算 r 再更新，或换奖励时重算 r，以提升样本效率。  
- **Reward scaling**：在切换/生成新奖励时，对新奖励做均值方差对齐，减少训练不稳定。  
- **种群 + 选择**：我们已有「每轮多样本 + 选 best」，但策略不保留；若要做「多策略协同进化」，需要引入多策略共享 buffer 与选择机制，改动较大。

---

## 6. 参考文献

- **LaRes**: Li et al., "LaRes: Evolutionary Reinforcement Learning with LLM-based Adaptive Reward Search", NeurIPS 2025.  
- **Eureka**: 我们实现基于 Eureka 风格的 LLM 奖励优化，与 VisFly 深度集成。
