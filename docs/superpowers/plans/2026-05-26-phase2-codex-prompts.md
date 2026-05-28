# Phase 2 Codex 执行提示词

> 使用方法：将下方「综合执行提示词」一次性发给 Codex。
> Codex 收到后使用 `superpowers:subagent-driven-development` 技能，
> 自主调度子 agent 按顺序完成 Task 1–7。
> Task 间存在代码依赖，**子 agent 必须顺序执行，严禁并行**。

---

## Phase 2 综合执行提示词

```
### 任务上下文

项目：SafeComm-VoI，Python 3.10 + PyTorch，UAV 编队 MARL
当前阶段：Phase 2 — 完整 SafeComm-VoI（HOCBF-QP + 双约束 + 层级 MGDA + Baselines）
工作目录：/root/autodl-tmp/safecomm-marl/
Phase 1 状态：✅ 已完成（80 tests passing，commit 4f1e825）

已完成的前置文件（Phase 1 遗留，不得修改）：
- envs/uav_formation_env.py
- algorithms/scheduler.py
- algorithms/networks.py（Phase 2 会在此添加 IntCostCritic）
- algorithms/ppo_utils.py（Phase 2 会在此扩展 RolloutBuffer）
- algorithms/safecomm_psched.py（Phase 2 会在此集成 QP / MGDA / λ_int）
- train.py（Phase 2 会在此添加 WandB 集成）
- configs/default.yaml
- tests/test_env.py、test_ppo_utils.py、test_networks.py、test_scheduler.py、test_algorithm.py

---

### 执行指令

使用 superpowers:subagent-driven-development 技能，按顺序调度子 agent 完成以下 Task 1–7。
每个 Task 完成后必须经过 spec compliance 和 code quality 双轮审查再进入下一个。
**Task 间存在代码依赖，必须串行执行。**

完整任务规格见：docs/superpowers/plans/2026-05-26-safecomm-voi-phase2.md
架构约束见：docs/superpowers/plans/2026-05-26-safecomm-voi-master-coordination.md 第4节

---

### 架构约束摘要（子 agent 必须遵守）

**约束 1（双动作不变项）**
collect_rollout 中环境执行 a_star（QP 过滤后），PPO log probability 使用 a_pi（策略原始输出）。
两者必须分开存储于 RolloutBuffer，不可合并或混用。

**约束 2（CBF 缓冲区单调性）**
d_buf_ij = v_max × dt_comm_ij，随通信缺失时间严格单调增。
d_eff = d_min + d_buf；调度器 h_sched 和 QP 约束必须使用完全相同的 d_buf 公式。

**约束 5（QP slack 变量）**
ξ_i ≥ 0 必须保留。优化变量固定为 4 维 [a_x, a_y, a_z, ξ]。禁止改为硬约束（无 slack）。

**约束 6（Safety gate 用 EMA，不用单 rollout）**
J_safe_ema ← (1-α) × J_safe_ema + α × J_safe_current
safe_gate = (J_safe_ema ≤ d_safe + δ_s)
λ_int 更新条件必须基于 EMA 值，禁止直接用单次 rollout 的 J_safe 触发。

**约束 7（MGDA 两阶段顺序）**
第一级：(g_task, λ_s·g_safe)；第二级：(g_ts, λ_int·g_int)。
顺序不可颠倒。每级仅在对应 λ > threshold 时激活 MGDA，否则退化为普通梯度。

**约束 10（测试文件只增不减）**
test_env.py、test_ppo_utils.py、test_networks.py、test_scheduler.py、test_algorithm.py
在 Phase 2 中不得修改或删除，只允许追加。

**约束 11（evaluate.py 双模式）**
evaluate.py 必须同时支持 --no_qp（关 Shield）和默认开启 Shield 两种模式。

**约束 13（WandB 训练追踪）**
train.py 中所有正式训练必须初始化 WandB 并记录 metrics。
wandb.init(project="safecomm-marl", config=cfg, name=f"SafeCommVoI-N{n}-k{k}-{timestamp}")
API Key 通过 WANDB_API_KEY 环境变量注入（已在 .env 文件中），禁止硬编码。
测试文件顶部必须设置 os.environ.setdefault("WANDB_MODE", "disabled")，不得影响 pytest 通过。

---

### Task 1：C+ HOCBF-QP Shield

**新建文件：**
- algorithms/hocbf_qp.py
- tests/test_hocbf_qp.py

**实现要点：**
HOCBFQPShield 类，per-agent SLSQP 求解，4 维优化变量 [a_x, a_y, a_z, ξ]：
- _build_constraint()：
  d_buf = v_max × dt_comm；d_eff = d_min + d_buf
  h = ||r_ij||^2 - d_eff^2
  dh_dt = 2 * r_ij^T * v_ij
  psi1 = dh_dt + alpha1 * h
  m_acc = 2 * a_max * ||r_ij||
  rhs_j = -(2*||v_ij||^2 + alpha1*dh_dt + alpha2*psi1 - m_acc)
  g_j = [2*r_ij; 1]，约束：g_j @ x - rhs_j >= 0
- filter_action()：无邻居时原样返回 a_pi；返回 (a_star, xi, intervention)
- filter_all_agents()：对所有 agent 并行调用 filter_action

tests/test_hocbf_qp.py 必须包含以下测试类（完整代码见 phase2.md Task 1 Step 1）：
- TestNoNeighbors::test_no_neighbors_returns_a_pi
- TestSafeRegion::test_safe_pair_no_intervention
- TestDangerousApproach::test_head_on_collision_intervenes
- TestDangerousApproach::test_qp_output_bounded_by_a_max
- TestSlackVariable::test_slack_nonnegative
- TestConservativeBuffer::test_larger_dt_comm_tightens_constraint
- TestFilterAllAgents::test_filter_all_agents_shapes

**成功标准：**
```bash
cd /root/autodl-tmp/safecomm-marl
python -m pytest tests/test_hocbf_qp.py -v
# 预期：7 个测试全部 PASSED
python -m pytest tests/ -v 2>&1 | tail -5
# 预期：全部 PASSED（Phase 1 无回归）
```

---

### Task 2：网络和缓冲区扩展

**修改文件：**
- algorithms/networks.py
- algorithms/ppo_utils.py

**networks.py 修改要点：**

1. 在 CostCritic 之后新增 IntCostCritic（结构与 CostCritic 完全相同）：
```python
class IntCostCritic(nn.Module):
    def __init__(self, global_state_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None: hidden_dims = [256, 256]
        self.int_cost_value_net = build_mlp(global_state_dim, 1, hidden_dims, "tanh")
    def forward(self, global_state):
        return self.int_cost_value_net(global_state).squeeze(-1)
```

2. SafeCommNetworks.__init__ 中添加：
```python
self.int_cost_critic = IntCostCritic(global_state_dim, hidden_dims)
```

3. get_values() 扩展为返回 3-tuple：(values, cost_values, int_cost_values)

4. act() 扩展返回 5-tuple（末尾增加 int_cost_values_pred: ndarray[N]）：
```python
int_cost_value = self.int_cost_critic(global_state.unsqueeze(0)).squeeze().cpu().numpy()
return actions, log_probs, values, cost_values, np.full(self.n_agents, float(int_cost_value))
```

5. evaluate_actions_batch() 扩展返回 6-tuple（末尾增加 int_cost_values）：
```python
values, cost_values, int_cost_values = self.get_values(global_states)
return log_probs, entropy, values, cost_values, int_cost_values, agg_msgs
```

**ppo_utils.py 修改要点：**

1. RolloutBuffer.reset() 中添加：
```python
self.int_costs = np.zeros((T, N), dtype=np.float32)
self.int_cost_values = np.zeros(T + 1, dtype=np.float32)
```

2. add() 参数列表末尾添加（有默认值，向后兼容）：
```python
int_cost: Optional[np.ndarray] = None,
int_cost_value: float = 0.0,
```
并在方法体中存储：
```python
if int_cost is not None: self.int_costs[t] = int_cost
self.int_cost_values[t] = int_cost_value
```

3. finish_rollout() 添加参数 last_int_cost_value: float = 0.0，存储到 self.int_cost_values[self.ptr]

4. compute_advantages_and_returns() 扩展：对 int_costs 计算 GAE，返回 6-tuple：
(advantages, returns, cost_advantages, cost_returns, int_cost_advantages, int_cost_returns)

5. get_training_data() 返回字典新增键：
int_cost_advantages, int_cost_returns, int_cost_values_old

**⚠️ 兼容性注意：**
test_networks.py 中如有测试 act() 返回值 tuple 长度为 4 的断言，需更新为 5；
evaluate_actions_batch() 返回 tuple 长度 5 的断言需更新为 6。

**成功标准：**
```bash
cd /root/autodl-tmp/safecomm-marl
python -m pytest tests/test_networks.py tests/test_ppo_utils.py tests/ -v 2>&1 | tail -10
# 预期：全部 PASSED
```

---

### Task 3：双动作 collect_rollout（集成 HOCBF-QP）

**修改文件：**
- algorithms/safecomm_psched.py

**修改要点：**

Step 1：添加 import 和 __init__ 中的新参数：
```python
from algorithms.hocbf_qp import HOCBFQPShield
```
default_config 新增（见 phase2.md Task 3 Step 1）：
use_hocbf_qp, alpha1, alpha2, qp_rho, lambda_int_init, lambda_int_max,
lr_lambda_int, safety_budget_int, safety_gate_ema_window, safety_gate_tolerance,
use_mgda, lambda_threshold, lambda_int_threshold

__init__ 新增：
```python
self.lambda_int = self.config["lambda_int_init"]
self.lambda_int_max = self.config["lambda_int_max"]
self._j_safe_ema = None
self._ema_alpha = 1.0 / max(self.config["safety_gate_ema_window"], 1)
self.safe_gate_active = False

if self.config["use_hocbf_qp"]:
    self.shield = HOCBFQPShield(
        d_min=self.config["d_min"],
        a_max=self.config["action_scale"],
        alpha1=self.config["alpha1"],
        alpha2=self.config["alpha2"],
        rho=self.config["qp_rho"],
        v_max=self.config.get("v_max_cbf", env.v_max),
        dt=env.dt,
    )
else:
    self.shield = None
```

critic_params 扩展包含 int_cost_critic.parameters()

Step 2：更新 collect_rollout() 循环体：
- networks.act() 返回 5-tuple（新增 int_cost_values_pred）
- 调用 shield.filter_all_agents() 得到 a_star_scaled、xis、int_costs_step
  - 位置/速度/通信时间来自 self.scheduler._last_known_pos / _last_known_vel / _last_comm_time
- 环境执行 exec_actions（QP 过滤后）
- buffer.add() 中传入 actions=actions_pi（策略动作，非 QP 动作），int_cost=int_costs_step
- bootstrap 调用 finish_rollout(last_int_cost_value=...)

**⚠️ 关键不变项（约束 1）：**
buffer 中存储的 actions 必须是 a_pi（策略原始动作），环境 step 用 a_star（QP 输出），两者严禁混淆。

**成功标准：**
```bash
cd /root/autodl-tmp/safecomm-marl
python -m pytest tests/ -v 2>&1 | tail -10
# 预期：全部 PASSED（含 Phase 1 所有测试无回归）
```

---

### Task 4：λ_int + Safety Gate EMA + 层级 MGDA

**修改文件：**
- algorithms/safecomm_psched.py

**update() 修改要点：**

Step 1：前向计算使用 6-tuple：
```python
log_probs_b, entropy_b, values_b, cost_values_b, int_cost_values_b, _ = \
    self.networks.evaluate_actions_batch(...)
```

Step 2：计算 int_cost 相关损失：
```python
int_cost_adv_b = data["int_cost_advantages"][idx]
int_penalty = self.lambda_int * int_cost_adv_b.mean()
int_cost_vf_loss = value_loss(int_cost_values_b, data["int_cost_values_old"][idx].detach(),
                               data["int_cost_returns"][idx], clip_eps)
total_critic_loss += self.config["cost_value_coef"] * int_cost_vf_loss
```

Step 3：层级 Two-Stage MGDA（严格遵守约束 7）：
```
actor_params = encoder.params + aggregator.params + actor.params
若 use_mgda AND λ_s > lambda_threshold:
    第一级：分别对 (task+entropy loss) 和 safety_penalty 做 backward，得 g_task, g_safe
    调用 mgda_solve(g_task, g_safe) → w_task, w_safe
    g_ts = w_task*g_task + w_safe*g_safe
否则：
    整体 backward，得 g_ts

若 safe_gate_active AND λ_int > lambda_int_threshold:
    对 int_penalty 做 backward，得 g_int
    调用 mgda_solve(g_ts, g_int) → w_ts, w_int
    g_final = w_ts*g_ts + w_int*g_int
否则（safe_gate 未激活）：
    g_final = g_ts（不加入 int_penalty 梯度）

将 g_final 写回各参数的 .grad，clip_grad_norm，optimizer.step()
```

Step 4：Lagrangian 乘子更新（在 update() 循环外，每次迭代调用一次）：
```python
# λ_s 更新（无条件）
self.lambda_s = clip(λ_s + lr_lambda * (J_safe - d_safe), 0, λ_max)

# Safety gate EMA（约束 6）
if self._j_safe_ema is None: self._j_safe_ema = J_safe
else: self._j_safe_ema = (1-α)*self._j_safe_ema + α*J_safe
self.safe_gate_active = (self._j_safe_ema <= d_safe + delta_s)

# λ_int 更新（仅在 safe_gate 激活后，约束 6）
if self.safe_gate_active:
    self.lambda_int = clip(λ_int + lr_lambda_int*(J_int - d_int), 0, λ_int_max)
```

Step 5：update_info 字典新增：lambda_int, safe_gate_active, j_safe_ema, J_int, mean_xi

Step 6：save/load 持久化 lambda_int, _j_safe_ema, safe_gate_active

Step 7：train() 日志打印新增 λ_int、gate 状态

**成功标准：**
```bash
cd /root/autodl-tmp/safecomm-marl
python -m pytest tests/ -v 2>&1 | tail -10
# 预期：全部 PASSED
```

---

### Task 5：Baselines（配置包装器类）

**新建文件：**
- baselines/__init__.py（暂不含 MACPO，Task 6 补充）
- baselines/mappo.py
- baselines/mappo_lag.py
- baselines/dgn_mappo.py
- baselines/hocbf_ppo.py
- baselines/random_topk.py

所有 baseline 均继承 SafeCommVoI，通过配置覆盖实现差异：

| 类 | schedule_mode | k | use_hocbf_qp | safety_budget | lambda_int_init | use_mgda |
|----|---------------|---|--------------|---------------|-----------------|----------|
| MAPPOAgent | full | 9999 | False | 1e9 | 0.0 | False |
| MAPPOLagAgent | full | 9999 | False | 默认 | 0.0 | False |
| DGNMAPPOAgent | random | 默认 | False | 1e9 | 0.0 | False |
| HOCBFPPOAgent | full | 9999 | True | 默认 | 0.0 | False |
| SafeCommRandomTopK | random | 默认 | True | 默认 | 默认 | True |

完整代码见 phase2.md Task 5。

**成功标准：**
```bash
cd /root/autodl-tmp/safecomm-marl
python -c "from baselines import MAPPOAgent, MAPPOLagAgent, DGNMAPPOAgent, HOCBFPPOAgent, SafeCommRandomTopK; print('OK')"
python -m pytest tests/ -v 2>&1 | tail -5
# 预期：全部 PASSED
```

---

### Task 6：MACPO Baseline + test_baselines.py

**新建文件：**
- baselines/macpo.py
- tests/test_baselines.py

**macpo.py 实现要点：**
MACPOAgent 继承 SafeCommVoI（全通信 + 无 QP），重写 update()：
- 若 J_safe > d_safe（constraint_active=True）：
  - 计算任务梯度 g_task（backward on task+entropy loss）
  - 计算安全梯度 g_safe（backward on cost_adv.mean()）
  - 投影：g_proj = g_task - (dot(g_task,g_safe) / ||g_safe||^2) * g_safe
  - 将 g_proj 写回参数 .grad
- 否则：直接 backward task loss

完整代码见 phase2.md Task 6 Step 1（含 6-tuple 解包 evaluate_actions_batch）。

**更新 baselines/__init__.py：**
```python
from baselines.macpo import MACPOAgent
__all__ = ["MAPPOAgent", "MAPPOLagAgent", "DGNMAPPOAgent",
           "HOCBFPPOAgent", "SafeCommRandomTopK", "MACPOAgent"]
```

**tests/test_baselines.py 要求：**
- 文件顶部添加 os.environ.setdefault("WANDB_MODE", "disabled")
- @pytest.mark.parametrize 6 个 baseline 各跑一次 collect_rollout + update
- FAST_CONFIG 中设置 use_hocbf_qp=False 加速 smoke test

完整测试代码见 phase2.md Task 6 Step 4。

**成功标准：**
```bash
cd /root/autodl-tmp/safecomm-marl
python -m pytest tests/test_baselines.py -v
# 预期：6 个参数化测试全部 PASSED
python -m pytest tests/ -v 2>&1 | tail -10
# 预期：全部 PASSED
```

---

### Task 7：evaluate.py + 消融配置 + WandB 集成 + 集成冒烟测试

**新建/修改文件：**
- evaluate.py
- configs/ablation/beta_zero.yaml
- configs/ablation/no_cbf_buffer.yaml
- configs/ablation/no_hocbf_qp.yaml
- configs/ablation/no_lambda_int.yaml
- configs/ablation/no_mgda.yaml
- configs/ablation/random_topk.yaml
- train.py（修改：WandB 集成 + 消融参数支持）

**evaluate.py 要求（约束 11）：**
- 支持 --no_qp 参数（关闭 QP Shield，shield=None）
- 支持 --checkpoint、--config、--episodes、--n_agents 参数
- run_evaluation() 返回：FE, FE_std, Success, Collision, CVR, MPS, InterventionRate, lambda_s, lambda_int
- 完整代码见 phase2.md Task 7 Step 1

**消融 YAML 见 phase2.md Task 7 Step 2：**
- beta_zero.yaml：comm.beta=0.0
- no_cbf_buffer.yaml：comm.v_max_cbf=0.0
- no_hocbf_qp.yaml：safety.use_hocbf_qp=false
- no_lambda_int.yaml：safety.lambda_int_init=0.0, safety_budget_int=1e9
- no_mgda.yaml：safety.use_mgda=false
- random_topk.yaml：comm.schedule_mode=random, comm.beta=0.0

**train.py 修改要点（约束 13）：**

顶部添加：
```python
import os, wandb
from datetime import datetime
```

flatten_config() 返回字典中添加：
```python
"use_hocbf_qp": safety_cfg.get("use_hocbf_qp", True),
"use_mgda": safety_cfg.get("use_mgda", True),
"lambda_int_init": safety_cfg.get("lambda_int_init", 0.0),
"safety_budget_int": safety_cfg.get("safety_budget_int", 0.5),
```

main() 中 agent 初始化后立即添加：
```python
run_name = (f"SafeCommVoI-N{env_cfg.get('n_agents',4)}-k{cfg_flat.get('k',4)}"
            f"-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
wandb.init(project="safecomm-marl", name=run_name, config=cfg_flat,
           tags=[f"phase2", f"N{env_cfg.get('n_agents',4)}"])
```

每次迭代日志后添加 wandb.log({...所有 Phase 2 metrics...}, step=agent.total_steps)
（必须记录：actor_loss, critic_loss, lambda_s, lambda_int, J_safe, J_int,
safe_gate_active, approx_kl, mean_episode_cost, mean_formation_error,
mean_bandwidth_util, fps）

训练结束后（agent.save 之后）添加 wandb.finish()

**注意：所有测试文件（包括新建的 test_baselines.py）顶部必须有：**
```python
import os
os.environ.setdefault("WANDB_MODE", "disabled")
```

**集成冒烟测试（Task 7 Step 3c + Step 4）：**

dry-run 验证（WANDB_MODE=disabled 模式）：
```bash
WANDB_MODE=disabled python train.py --config configs/default.yaml --max_iterations 3
# 预期：3 次迭代正常完成，无 WandB 相关错误
```

Phase 2 集成冒烟测试（见 phase2.md Task 7 Step 4）：
```bash
python -c "
import numpy as np
from envs.uav_formation_env import UAVFormationEnv
from algorithms.safecomm_psched import SafeCommVoI
env = UAVFormationEnv(n_agents=4, d_min=0.3, dt=0.1, max_steps=10, R_comm=5.0)
cfg = {'k':4,'n_steps':40,'n_epochs':2,'batch_size':20,'msg_dim':32,'hidden_dims':[64,64],
       'H':2,'safety_budget':0.5,'beta':1.0,'tau_ref':1.0,'tau_max':5.0,
       'use_hocbf_qp':True,'qp_rho':100.0,'alpha1':1.0,'alpha2':1.0,
       'lambda_int_init':0.0,'safety_budget_int':0.5,'use_mgda':True,
       'safety_gate_ema_window':3,'safety_gate_tolerance':0.1}
agent = SafeCommVoI(env, config=cfg, device='cpu')
rollout_info = agent.collect_rollout()
update_info = agent.update()
assert 'lambda_int' in update_info
assert 'safe_gate_active' in update_info
assert 'J_int' in update_info
print('OK: collect_rollout + update + Phase 2 指标')

obs_list, _ = env.reset()
agent.scheduler.reset_episode(env.positions)
for _ in range(20):
    phys = env.get_physical_graph()
    result = agent.scheduler.schedule_step(env.positions, phys)
    assert len(result['active_edges']) <= 4
print('OK: |E_t| <= k 仍然满足')

import tempfile, os
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f: tmppath = f.name
agent.save(tmppath)
agent2 = SafeCommVoI(env, config=cfg, device='cpu')
agent2.load(tmppath)
assert abs(agent2.lambda_int - agent.lambda_int) < 1e-6
os.unlink(tmppath)
print('OK: save/load 正常（含 lambda_int, j_safe_ema）')
print('=== Phase 2 集成冒烟测试全部通过 ===')
"
```

**最终全套测试：**
```bash
cd /root/autodl-tmp/safecomm-marl
python -m pytest tests/ -v
# 预期：全部 PASSED（含 test_hocbf_qp.py、test_baselines.py、Phase 1 所有测试）
```

---

### Phase 2 总体成功标准

所有以下条件全部满足，Phase 2 才算完成：

1. pytest tests/ -v → 全部 PASSED（含 test_hocbf_qp.py、test_baselines.py）
2. HOCBF-QP：头对头碰撞场景中 intervention > 0（test_head_on_collision_intervenes）
3. dt_comm 越大 → intervention >= fresh（test_larger_dt_comm_tightens_constraint）
4. 6 个 baseline 各自完成 1 次 collect_rollout + update
5. evaluate.py --no_qp 和默认模式均可执行（不要求有 checkpoint，只要无 import error）
6. WANDB_MODE=disabled python train.py --max_iterations 3 正常完成（无错误）
7. Phase 2 集成冒烟：lambda_int / safe_gate_active / J_int 字段存在，save/load 完整

完成后，将 pytest tests/ -v 的完整输出和集成冒烟测试输出发给 Claude，请求 Phase 2 → Phase 3 门控审查。

---

### 注意事项

1. 每个 Task 完成后运行 pytest tests/ -v 确认无回归，才开始下一个 Task
2. test_algorithm.py 含有 test_phase1_smoke，Phase 2 改动 safecomm_psched.py 后必须确保该测试依然通过
3. networks.py 的 act() 从 4-tuple 扩展为 5-tuple，所有调用方（safecomm_psched.py、test_networks.py）必须同步更新
4. evaluate_actions_batch() 从 5-tuple 扩展为 6-tuple，同理
5. MACPO 的 update() 内 evaluate_actions_batch 调用需解包 6 个返回值（第5个 _ 为 int_cost_values）
6. WandB API Key 已在 /root/autodl-tmp/safecomm-marl/.env 中配置，train.py 通过 load_dotenv() 读取
```
