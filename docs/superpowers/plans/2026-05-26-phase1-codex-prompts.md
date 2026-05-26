# Phase 1 Codex 执行提示词

> 使用方法：按顺序将下方提示词逐个发给 Codex。每个 Task 完成并提交后，再发下一个。
> Task 间存在代码依赖，**严禁并行执行**。

---

## Task 0 提示词：双积分器 UAV 编队环境

```
### 任务上下文

项目：SafeComm-VoI，Python 3.10 + PyTorch，UAV 编队 MARL
当前阶段：Phase 1 — Task 0（从零构建）
工作目录：/root/autodl-tmp/safecomm-marl/
已完成的前置任务：无（这是第一个 Task）

架构约束（必须遵守）：
- 观测向量格式固定为 [p_x,p_y,p_z, v_x,v_y,v_z]（6维）+ K_obs个邻居各6维
- obs_dim = 6 + K_obs * 6；此公式不得改动
- step() 返回 (obs_list, rewards, costs, done, info)；costs 为 0.0 或 1.0（碰撞指示器）
- get_physical_graph() 返回真实位置图，不受 k 裁剪
- compute_cbf_values() 使用真实位置，不使用噪声观测

### 当前任务

完整内容见：docs/superpowers/plans/2026-05-26-safecomm-voi-phase1.md，Task 0 节

摘要：
1. 创建 envs/__init__.py、algorithms/__init__.py、tests/__init__.py（空文件）
2. 写 tests/test_env.py（11 个测试，先运行确认 FAIL）
3. 实现 envs/uav_formation_env.py（双积分器动力学，圆形编队目标）
4. 运行 pytest tests/test_env.py -v 确认全部 PASS
5. git commit

### 成功标准

```bash
cd /root/autodl-tmp/safecomm-marl
python -m pytest tests/test_env.py -v
# 预期：11 passed
git log --oneline -1
# 预期包含：feat: add UAVFormationEnv
```
```

---

## Task 1 提示词：PPO 工具函数

```
### 任务上下文

项目：SafeComm-VoI，Python 3.10 + PyTorch，UAV 编队 MARL
当前阶段：Phase 1 — Task 1
工作目录：/root/autodl-tmp/safecomm-marl/

已完成的前置任务：
- Task 0：envs/uav_formation_env.py 已创建，tests/test_env.py 全部通过

架构约束（必须遵守）：
- RolloutBuffer.get_training_data() 必须返回包含 obs/actions/log_probs_old/advantages/returns/cost_advantages/cost_returns/global_states/adj_matrices 的字典
- GAE 在 done=True 时必须截断（mask = 1 - done）
- mgda_solve 保留供 Phase 2 使用，Phase 1 不调用但必须实现

### 当前任务

完整内容见：docs/superpowers/plans/2026-05-26-safecomm-voi-phase1.md，Task 1 节

摘要：
1. 写 tests/test_ppo_utils.py（先运行确认 FAIL）
2. 实现 algorithms/ppo_utils.py（RolloutBuffer、compute_gae、ppo_clip_loss、value_loss、explained_variance、mgda_solve）
3. pytest tests/test_ppo_utils.py -v 全部 PASS
4. git commit

### 成功标准

```bash
cd /root/autodl-tmp/safecomm-marl
python -m pytest tests/test_ppo_utils.py -v
# 预期：全部 passed
python -m pytest tests/ -v 2>&1 | tail -5
# 预期：test_env + test_ppo_utils 全部通过，无 FAILED
```
```

---

## Task 2 提示词：神经网络模块

```
### 任务上下文

项目：SafeComm-VoI，Python 3.10 + PyTorch，UAV 编队 MARL
当前阶段：Phase 1 — Task 2
工作目录：/root/autodl-tmp/safecomm-marl/

已完成的前置任务：
- Task 0：envs/uav_formation_env.py
- Task 1：algorithms/ppo_utils.py

架构约束（必须遵守）：
- 参数共享：所有 agent 共用同一个 Actor、MessageEncoder、GATAggregator（CTDE）
- log_sigma 为共享可训练标量参数（nn.Parameter），不得为 agent 独立参数
- act() 返回 4-tuple：(actions[N,3], log_probs[N], values[...], cost_values[...])
- actions 经过 tanh 压缩，范围 (-1, 1)
- evaluate_actions_batch() 返回 5-tuple：(log_probs[T,N], entropy[T,N], values[T], cost_values[T], agg_msgs[T,N,msg_dim])
- Critic 输入：global_state = concat(obs_all) = [N*obs_dim]，自动 pad/trim

### 当前任务

完整内容见：docs/superpowers/plans/2026-05-26-safecomm-voi-phase1.md，Task 2 节

摘要：
1. 写 tests/test_networks.py（先运行确认 FAIL）
2. 实现 algorithms/networks.py（MessageEncoder、GATAggregator、Actor、Critic、CostCritic、SafeCommNetworks）
3. pytest tests/test_networks.py -v 全部 PASS
4. git commit

### 成功标准

```bash
cd /root/autodl-tmp/safecomm-marl
python -m pytest tests/test_networks.py -v
# 预期：全部 passed（含 act 形状、tanh 范围、梯度流动测试）
python -m pytest tests/ -v 2>&1 | tail -5
# 预期：全部 passed，无 FAILED
```
```

---

## Task 3 提示词：VoI 感知调度器

```
### 任务上下文

项目：SafeComm-VoI，Python 3.10 + PyTorch，UAV 编队 MARL
当前阶段：Phase 1 — Task 3
工作目录：/root/autodl-tmp/safecomm-marl/

已完成的前置任务：
- Task 0、1、2 全部完成

架构约束（必须遵守，来自 master-coordination.md 第4节）：
- 约束3：VoI 公式不得修改：priority_ij = (1 + β·min(Δt/τ_ref, τ_max)) / max(h_sched_ij, ε)
- 约束3：h_sched_ij = max(||p_i - p̂_j|| - d_min - d_buf_ij, 0)^2，clip到0
- 约束4：仅被选入 E_t 的 pair 更新通信历史，未选入的只递增 Δt_comm
- 约束1（Proposition 1）：|E_t| <= k 由 TopK 严格保证
- schedule_mode="full" 返回所有边（无限通信预算基线用）
- schedule_mode="random" 返回随机 k 条边（DGN baseline 用）

### 当前任务

完整内容见：docs/superpowers/plans/2026-05-26-safecomm-voi-phase1.md，Task 3 节

摘要：
1. 写 tests/test_scheduler.py（8 个测试，含 TestBudgetConstraint 和 TestVoIPriority）
2. 实现 algorithms/scheduler.py（SafetyPriorityScheduler）
3. pytest tests/test_scheduler.py -v 全部 PASS
4. git commit

### 成功标准

```bash
cd /root/autodl-tmp/safecomm-marl
python -m pytest tests/test_scheduler.py -v
# 预期：8 passed（含 test_voi_formula_beta_zero_numerically 等精确数值测试）
python -m pytest tests/ -v 2>&1 | tail -5
# 预期：全部 passed
```
```

---

## Task 4 提示词：SafeCommVoI 主算法

```
### 任务上下文

项目：SafeComm-VoI，Python 3.10 + PyTorch，UAV 编队 MARL
当前阶段：Phase 1 — Task 4（核心算法）
工作目录：/root/autodl-tmp/safecomm-marl/

已完成的前置任务：
- Task 0：envs/uav_formation_env.py
- Task 1：algorithms/ppo_utils.py
- Task 2：algorithms/networks.py
- Task 3：algorithms/scheduler.py

架构约束（必须遵守）：
- Phase 1 无 HOCBF-QP：collect_rollout() 直接执行策略动作，不做 QP 过滤
- Phase 1 无 MGDA：update() 只使用 total_actor_loss.backward()，不调用 mgda_solve
- 双动作区分：Phase 1 不存在（Phase 2 才引入），此处 exec_actions = policy_actions * action_scale
- Lagrangian 更新：λ_s ← clip(λ_s + α·(J_safe - d_safe), 0, λ_max)，其中 J_safe = mean_step_cost * max_steps
- collect_rollout 必须调用 scheduler.schedule() + scheduler.update_history()
- RolloutBuffer.add() 的 adj_matrix 参数：从 active_edges 构造 (N,N) 邻接矩阵
- save/load 必须持久化 lambda_s、n_iterations、total_steps

### 当前任务

完整内容见：docs/superpowers/plans/2026-05-26-safecomm-voi-phase1.md，Task 4 节

摘要：
1. 写 tests/test_algorithm.py（6 个测试，先运行确认 FAIL）
2. 实现 algorithms/safecomm_psched.py（SafeCommVoI 类：__init__、collect_rollout、update、train、evaluate、save、load）
3. pytest tests/test_algorithm.py -v 全部 PASS
4. pytest tests/ -v 全套通过（无回归）
5. git commit

### 成功标准

```bash
cd /root/autodl-tmp/safecomm-marl
python -m pytest tests/test_algorithm.py -v
# 预期：6 passed（含 lambda_s 更新方向、save/load roundtrip）
python -m pytest tests/ -v 2>&1 | tail -8
# 预期：全部 passed（test_env + test_ppo_utils + test_networks + test_scheduler + test_algorithm）
```
```

---

## Task 5 提示词：配置文件与训练入口

```
### 任务上下文

项目：SafeComm-VoI，Python 3.10 + PyTorch，UAV 编队 MARL
当前阶段：Phase 1 — Task 5
工作目录：/root/autodl-tmp/safecomm-marl/

已完成的前置任务：
- Task 0–4 全部完成，pytest tests/ 全部通过

架构约束（必须遵守）：
- configs/default.yaml 必须包含 comm.beta（VoI 权重）、comm.tau_ref、comm.tau_max、comm.v_max_cbf
- flatten_config() 将嵌套 YAML 展平为 SafeCommVoI 所需的扁平字典
- train.py 必须支持 --beta 参数（消融实验用）
- train.py 不得直接执行训练（只定义 main()，由 if __name__ == "__main__": main() 调用）

### 当前任务

完整内容见：docs/superpowers/plans/2026-05-26-safecomm-voi-phase1.md，Task 5 节

摘要：
1. mkdir -p configs
2. 创建 configs/default.yaml（含 env/comm/network/ppo/lr/safety/training 全部字段）
3. 创建 train.py（含 load_config、flatten_config、main）
4. 验证 yaml 可加载，train.py --help 不报错
5. git commit

### 成功标准

```bash
cd /root/autodl-tmp/safecomm-marl
python -c "import yaml; c=yaml.safe_load(open('configs/default.yaml')); assert c['comm']['beta']==1.0; print('OK')"
python train.py --help
# 预期：显示帮助信息，无 import error
python -m pytest tests/ -v 2>&1 | tail -5
# 预期：全部 passed（Task 5 不修改算法代码，无回归）
```
```

---

## Task 6 提示词：Phase 1 成功标准验证

```
### 任务上下文

项目：SafeComm-VoI，Python 3.10 + PyTorch，UAV 编队 MARL
当前阶段：Phase 1 — Task 6（验证，不写新代码）
工作目录：/root/autodl-tmp/safecomm-marl/

已完成的前置任务：
- Task 0–5 全部完成

### 当前任务

运行以下 4 项验证，确认 Phase 1 成功标准全部满足。完整脚本见：
docs/superpowers/plans/2026-05-26-safecomm-voi-phase1.md，Task 6 节

1. pytest tests/ -v → 全部 PASSED
2. 验证 |E_t| <= k：运行 20 步调度，每步 assert len(active_edges) <= 4
3. 验证 λ_s 更新方向：safety_budget=0.001 时训练后 λ_s > 初值
4. 验证 β=0 vs β>0 调度差异可见

### 成功标准（4 项均须通过）

```bash
cd /root/autodl-tmp/safecomm-marl

# 1. 全量测试
python -m pytest tests/ -v 2>&1 | tail -5
# 预期：全部 PASSED

# 2. |E_t| <= k
python -c "
from envs.uav_formation_env import UAVFormationEnv
from algorithms.safecomm_psched import SafeCommVoI
env = UAVFormationEnv(n_agents=4, d_min=0.5, dt=0.1, max_steps=10, R_comm=5.0)
agent = SafeCommVoI(env, config={'k':4,'n_steps':40,'n_epochs':1,'batch_size':20,'msg_dim':32,'hidden_dims':[64,64],'H':2,'safety_budget':0.5,'beta':1.0,'tau_ref':1.0,'tau_max':5.0}, device='cpu')
env.reset(); agent.scheduler.reset_episode(env.positions)
for _ in range(20):
    phys=env.get_physical_graph(); ae=agent.scheduler.schedule(env.positions,phys)
    assert len(ae)<=4
print('OK: |E_t|<=k')
"

# 3. λ_s 方向
python -c "
from envs.uav_formation_env import UAVFormationEnv
from algorithms.safecomm_psched import SafeCommVoI
env = UAVFormationEnv(n_agents=4,d_min=0.3,dt=0.1,max_steps=10,R_comm=5.0)
agent = SafeCommVoI(env,config={'k':4,'n_steps':40,'n_epochs':2,'batch_size':20,'msg_dim':32,'hidden_dims':[64,64],'H':2,'safety_budget':0.001,'beta':1.0,'tau_ref':1.0,'tau_max':5.0},device='cpu')
lb=float(agent.lambda_s); agent.collect_rollout(); agent.update(); la=float(agent.lambda_s)
assert la>=lb; print(f'OK: λ_s {lb:.4f} -> {la:.4f}')
"

# 4. β 差异
python -c "
import numpy as np
from algorithms.scheduler import SafetyPriorityScheduler
pos=np.array([[0,0,0],[3,0,0],[10,0,0]],dtype=np.float64); edges=[(0,1),(0,2),(1,2)]
sv=SafetyPriorityScheduler(3,1,0.5,v_max=0.0,dt=0.1,beta=2.0,tau_ref=0.5,tau_max=10.0)
ss=SafetyPriorityScheduler(3,1,0.5,v_max=0.0,dt=0.1,beta=0.0,tau_ref=0.5,tau_max=10.0)
sv.reset_episode(pos); ss.reset_episode(pos)
for _ in range(20): sv.update_history([(0,1)],pos); ss.update_history([(0,1)],pos)
pv=sv.compute_voi_priority(pos,edges); ps=ss.compute_voi_priority(pos,edges)
assert abs(pv[(0,2)]/pv[(0,1)] - ps[(0,2)]/ps[(0,1)]) > 1e-3
print('OK: β差异可见')
"
```

全部通过后，将结果发回 Claude 进行 Phase 1 → Phase 2 门控审查。
```

---

## 给 Claude 的 Phase 1 → Phase 2 门控审查请求模板

当 Task 6 所有验证通过后，将以下内容发给 Claude：

```
Phase 1 已完成，请进行 Phase 1 → Phase 2 门控审查。

pytest 输出：
[粘贴 pytest tests/ -v 的完整输出]

4 项验证结果：
[粘贴 Task 6 四段脚本的输出]

请按照 docs/superpowers/plans/2026-05-26-safecomm-voi-master-coordination.md
第3节"Phase 1 → Phase 2 门控"逐条核查，并出具 ✅ 或 ❌ 结论。
```
