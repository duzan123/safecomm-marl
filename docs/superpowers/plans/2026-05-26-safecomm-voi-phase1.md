# SafeComm-VoI Phase 1 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 SafeComm-PSched v2.0 代码库升级为 SafeComm-VoI Phase 1（Scheduler + MAPPO-Lag Prototype），满足规格 v3.0 Phase 1 成功标准。

**Architecture:** 三处核心修改——(1) 在调度器中加入 VoI 优先级公式（AoI 分子 + 保守 h_sched 分母）；(2) 将主算法类重命名为 SafeCommVoI 并移除 Phase 1 不需要的 MGDA；(3) 更新配置与入口脚本。网络、环境、PPO 工具函数无需改动。

**Tech Stack:** Python 3.x, NumPy, PyTorch, pytest, PyYAML

**外部代码复用边界（Phase 1）：**
- `external/on-policy`（MIT）只参考 R-MAPPO 的 GAE、PPO clip、actor/critic 更新和 CTDE rollout 组织；本阶段保持轻量 `ppo_utils.py`，不迁移 on-policy runner、env wrapper 或训练脚本体系。
- `external/MAGIC`（MIT）可参考 masked attention 的邻接 mask 写法；本阶段自写 `MessageEncoder`/attention 聚合接口，保证只聚合 `E_t` 内消息。
- `external/InforMARL`（MIT）只参考局部图观测和邻接矩阵组织，不引入 torch-geometric 依赖。
- `external/sched_net` 无明确许可证且为 TF1 风格，只作为“共享信道 + k 个通信名额”的问题设定参考；`SafetyPriorityScheduler` 和 VoI 公式必须完整自写。
- Phase 1 不复制任何外部仓库源码片段；若后续借鉴 MIT 片段，需要在文件注释或文档中标明来源和 commit。

---

## 文件结构

| 文件 | 操作 | 说明 |
|------|------|------|
| `algorithms/scheduler.py` | 修改 | 新增 VoI 优先级公式（β, τ_ref, τ_max 参数 + `compute_voi_priority`），更新 `_safety_priority_topk` |
| `algorithms/safecomm_psched.py` | 修改 | 重命名类为 `SafeCommVoI`，移除 MGDA 分支，传递 VoI 调度器参数 |
| `configs/default.yaml` | 修改 | 新增 VoI 超参数（beta、tau_ref、tau_max、H、v_max_cbf），K_obs 改为 4 |
| `train.py` | 修改 | 更新 import 和 `flatten_config` 以传递 VoI 参数 |
| `tests/test_scheduler.py` | 修改 | 新增 `TestVoIPriority` 测试类（VoI 公式、β=0 退化、AoI 单调性） |
| `tests/test_algorithm.py` | 修改 | 更新 import 和测试函数为 `SafeCommVoI` |

**不修改的文件：** `algorithms/networks.py`（H=2 已正确），`algorithms/ppo_utils.py`（mgda_solve 留给 Phase 2），`envs/uav_formation_env.py`（动力学已正确），`tests/test_env.py`、`tests/test_ppo_utils.py`、`tests/test_networks.py`

---

## Task 1：VoI 优先级公式（scheduler.py）

**Files:**
- Modify: `algorithms/scheduler.py`
- Test: `tests/test_scheduler.py`

### 步骤

- [ ] **Step 1: 在 TestVoIPriority 中写失败测试**

在 `tests/test_scheduler.py` 文件末尾（`if __name__ == "__main__"` 之前）添加：

```python
class TestVoIPriority:
    """VoI 优先级公式正确性测试（规格 §2.1）"""

    def test_voi_formula_beta_zero_numerically(self):
        """β=0, v_max=0: priority = 1/h_sched = 1/(dist-d_min)^2"""
        sched = SafetyPriorityScheduler(
            n_agents=2, k=1, d_min=0.5, v_max=0.0, dt=0.1,
            beta=0.0, tau_ref=1.0, tau_max=5.0,
        )
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        sched.reset_episode(positions)
        prios = sched.compute_voi_priority(positions, [(0, 1)])
        # h = (1.0 - 0.5)^2 = 0.25; priority = 1.0 / 0.25 = 4.0
        assert abs(prios[(0, 1)] - 4.0) < 1e-6

    def test_voi_formula_with_aoi_numerically(self):
        """β=1, 5 steps gap: AoI=0.5, priority = (1+0.5)/0.25 = 6.0"""
        sched = SafetyPriorityScheduler(
            n_agents=2, k=1, d_min=0.5, v_max=0.0, dt=0.1,
            beta=1.0, tau_ref=1.0, tau_max=5.0,
        )
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        sched.reset_episode(positions)
        for _ in range(5):
            sched.update_history([], positions)  # 5 步未通信
        prios = sched.compute_voi_priority(positions, [(0, 1)])
        # dt_comm=5*0.1=0.5s; AoI=min(0.5/1.0,5.0)=0.5; h=(0.5)^2=0.25
        # priority=(1+1.0*0.5)/0.25=6.0
        assert abs(prios[(0, 1)] - 6.0) < 1e-6

    def test_beta_zero_selects_dangerous_pair(self):
        """β=0 应选最危险对（h 最小），不受 AoI 影响"""
        sched = SafetyPriorityScheduler(
            n_agents=3, k=1, d_min=0.5, v_max=0.0, dt=0.1,
            beta=0.0, tau_ref=1.0, tau_max=5.0,
        )
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.51, 0.0, 0.0],   # (0,1): h≈0.0001，极危险
            [5.0, 0.0, 0.0],    # (0,2): h=20.25，安全
        ], dtype=np.float64)
        sched.reset_episode(positions)
        # 让 (0,1) 保持新鲜，(0,2) AoI 积累 20 步——β=0 时 (0,1) 仍应被选中
        for _ in range(20):
            sched.update_history([(0, 1)], positions)
        active = sched.schedule(positions, [(0, 1), (0, 2), (1, 2)])
        assert (0, 1) in active, f"β=0 应选最危险对 (0,1)，实际={active}"

    def test_proposition2_monotone_priority(self):
        """Proposition 2：通信缺失时间越长 → VoI 优先级单调不降"""
        sched = SafetyPriorityScheduler(
            n_agents=2, k=1, d_min=0.5, v_max=0.0, dt=0.1,
            beta=1.0, tau_ref=1.0, tau_max=10.0,
        )
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        sched.reset_episode(positions)
        prev_prio = sched.compute_voi_priority(positions, [(0, 1)])[(0, 1)]
        for _ in range(8):
            sched.update_history([], positions)
            curr_prio = sched.compute_voi_priority(positions, [(0, 1)])[(0, 1)]
            assert curr_prio >= prev_prio - 1e-9, (
                f"VoI 优先级应单调不降：{prev_prio} -> {curr_prio}"
            )
            prev_prio = curr_prio

    def test_aoi_boost_stale_pair_over_safe_pair(self):
        """β>0: 长期未通信对的 VoI 优先级高于 β=0 时"""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],   # (0,1): h=(2.5)^2=6.25
            [1.5, 0.0, 0.0],   # (0,2): h=(1.0)^2=1.0
        ], dtype=np.float64)
        edges = [(0, 1), (0, 2), (1, 2)]

        sched_voi = SafetyPriorityScheduler(
            n_agents=3, k=1, d_min=0.5, v_max=0.0, dt=0.1,
            beta=2.0, tau_ref=0.5, tau_max=10.0,
        )
        sched_safe = SafetyPriorityScheduler(
            n_agents=3, k=1, d_min=0.5, v_max=0.0, dt=0.1,
            beta=0.0, tau_ref=0.5, tau_max=10.0,
        )
        sched_voi.reset_episode(positions)
        sched_safe.reset_episode(positions)

        # (0,1) 已 20 步未通信，(0,2) 保持新鲜
        for _ in range(20):
            sched_voi.update_history([(0, 2)], positions)
            sched_safe.update_history([(0, 2)], positions)

        prios_voi = sched_voi.compute_voi_priority(positions, edges)
        prios_safe = sched_safe.compute_voi_priority(positions, edges)

        # β>0 对 (0,1) 的优先级应 > β=0 时（AoI 带来提升）
        assert prios_voi[(0, 1)] > prios_safe[(0, 1)], (
            "β>0 应通过 AoI 提升长期未通信对的优先级"
        )

    def test_tau_max_clamps_aoi(self):
        """AoI 超过 tau_max * tau_ref 后优先级不再增长"""
        sched = SafetyPriorityScheduler(
            n_agents=2, k=1, d_min=0.5, v_max=0.0, dt=0.1,
            beta=1.0, tau_ref=1.0, tau_max=2.0,  # AoI 上限 = 2.0
        )
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        sched.reset_episode(positions)
        # 30 步 → dt_comm=3.0s; AoI=min(3.0/1.0,2.0)=2.0（已截断）
        for _ in range(30):
            sched.update_history([], positions)
        prio_at_30 = sched.compute_voi_priority(positions, [(0, 1)])[(0, 1)]
        # 再过 10 步，AoI 仍为 2.0（截断），优先级不变
        for _ in range(10):
            sched.update_history([], positions)
        prio_at_40 = sched.compute_voi_priority(positions, [(0, 1)])[(0, 1)]
        assert abs(prio_at_40 - prio_at_30) < 1e-6, "tau_max 应截断 AoI，优先级不再增长"
```

- [ ] **Step 2: 运行失败测试**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_scheduler.py::TestVoIPriority -v 2>&1 | head -30
```

预期：`ERROR` 或 `AttributeError: __init__() got unexpected keyword argument 'beta'`（`compute_voi_priority` 和参数尚未存在）

- [ ] **Step 3: 在 scheduler.py 中添加 VoI 参数和 compute_voi_priority 方法**

在 `SafetyPriorityScheduler.__init__` 的参数列表中添加三个新参数（在 `dt: float = 0.1` 之后，`):` 之前）：

```python
    def __init__(
        self,
        n_agents: int,
        k: int,
        d_min: float = 0.5,
        eps: float = 1e-6,
        schedule_mode: str = "safety_priority",
        v_max: float = 0.0,
        dt: float = 0.1,
        beta: float = 1.0,       # AoI 权重（β=0 退化为纯安全优先）
        tau_ref: float = 1.0,    # AoI 参考时间（秒），默认取通信周期
        tau_max: float = 5.0,    # AoI 放大上限
    ):
```

在 `__init__` 方法体中，在 `self._current_step: int = 0` 之后添加：

```python
        self.beta = beta
        self.tau_ref = tau_ref
        self.tau_max = tau_max
```

在 `compute_conservative_cbf_values` 方法之后（约第 255 行），添加新方法 `compute_voi_priority`：

```python
    def compute_voi_priority(
        self,
        positions: np.ndarray,
        edges: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[Tuple[int, int], float]:
        """
        VoI 优先级评分（规格 §2.1）：
          priority_ij = (1 + β·min(Δt_comm/τ_ref, τ_max)) / max(h_sched_ij, ε)
          h_sched_ij = max(||p_i - p_hat_j|| - d_min - d_buf_ij, 0)^2
          d_buf_ij = v_max · Δt_comm_ij

        β=0 退化为纯安全优先（消融基线）。
        注意：h_sched 仅用于调度排序，不用于 QP 约束。
        """
        if edges is None:
            edges = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]

        priorities: Dict[Tuple[int, int], float] = {}
        for raw_i, raw_j in edges:
            i, j = min(raw_i, raw_j), max(raw_i, raw_j)

            # --- h_sched（保守估计，使用最近通信已知位置）---
            if self._last_known_pos is not None and self.v_max > 0:
                p_hat_j = self._last_known_pos[j]
                dt_comm = float(self._last_comm_time[i, j]) * self.dt
                d_buf = self.v_max * dt_comm
            else:
                p_hat_j = positions[j]
                dt_comm = 0.0
                d_buf = 0.0

            dist = float(np.linalg.norm(positions[i] - p_hat_j))
            h_sched = max(dist - self.d_min - d_buf, 0.0) ** 2

            # --- AoI 分子（urgency）---
            aoi = min(dt_comm / max(self.tau_ref, 1e-9), self.tau_max)
            numerator = 1.0 + self.beta * aoi

            priorities[(i, j)] = numerator / max(h_sched, self.eps)

        return priorities
```

- [ ] **Step 4: 更新 `_safety_priority_topk` 使用 VoI 公式**

将 `_safety_priority_topk` 方法完整替换为：

```python
    def _safety_priority_topk(
        self,
        positions: np.ndarray,
        phys_edges: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """VoI 优先级 TopK 调度（规格 §2.1）。β=0 退化为纯安全优先。"""
        priorities = self.compute_voi_priority(positions, phys_edges)
        sorted_edges = sorted(
            phys_edges,
            key=lambda e: priorities.get((min(e), max(e)), 0.0),
            reverse=True,
        )
        return sorted_edges[:self.k]
```

- [ ] **Step 5: 运行 VoI 测试确认通过**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_scheduler.py::TestVoIPriority -v
```

预期：6 个测试全部 `PASSED`

- [ ] **Step 6: 运行全量调度器测试（确认无回归）**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_scheduler.py -v
```

预期：所有测试通过（原有测试 + 6 个新测试）

- [ ] **Step 7: 提交**

```bash
cd /root/autodl-tmp/safecomm-marl && git add algorithms/scheduler.py tests/test_scheduler.py
git commit -m "feat: add VoI priority formula to SafetyPriorityScheduler (Phase 1)"
```

---

## Task 2：重命名 SafeCommPSched → SafeCommVoI，移除 MGDA

**Files:**
- Modify: `algorithms/safecomm_psched.py`
- Test: `tests/test_algorithm.py`

### 背景

现有 `safecomm_psched.py` 包含 MGDA 分支（Phase 2 特性），Phase 1 规格明确不包含 MGDA。需要：
1. 将类名从 `SafeCommPSched` 改为 `SafeCommVoI`（保留文件名以避免大范围重构）
2. 移除 `update()` 中的 MGDA 条件分支，始终使用简单 backward
3. 将 VoI 调度器新参数（beta、tau_ref、tau_max）从 config 传入调度器

### 步骤

- [ ] **Step 1: 更新 test_algorithm.py 的 import**

将 `tests/test_algorithm.py` 第 9 行的 import 替换为：

```python
from algorithms.safecomm_psched import SafeCommVoI
```

并将所有 `SafeCommPSched` 引用改为 `SafeCommVoI`：

```python
FAST_CONFIG = {
    "k": 4,
    "n_steps": 40,
    "n_epochs": 2,
    "batch_size": 20,
    "msg_dim": 32,
    "hidden_dims": [64, 64],
    "H": 2,
    "safety_budget": 0.5,
    # VoI 参数
    "beta": 1.0,
    "tau_ref": 1.0,
    "tau_max": 5.0,
}


def make_env_and_agent():
    env = UAVFormationEnv(n_agents=4, d_min=0.3, dt=0.1, max_steps=50, R_comm=5.0)
    agent = SafeCommVoI(env, config=FAST_CONFIG, device="cpu")
    return env, agent
```

同时删除 FAST_CONFIG 中的 `"lambda_threshold": 0.1`（该参数随 MGDA 一起移除）。

- [ ] **Step 2: 运行失败测试**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_algorithm.py -v 2>&1 | head -20
```

预期：`ImportError: cannot import name 'SafeCommVoI'`

- [ ] **Step 3: 重命名类并移除 MGDA**

**3a. 类名和 import 注释：** 将 `safecomm_psched.py` 第 38 行 `class SafeCommPSched:` 改为：

```python
class SafeCommVoI:
    """
    SafeComm-VoI Phase 1 算法主类（Scheduler + MAPPO-Lag Prototype）。

    Phase 1 包含：Safety-Aware VoI Scheduler + MAPPO(H=2) + 单约束 Lagrangian(λ_s)
    Phase 1 不含：C+ HOCBF-QP、λ_int、MGDA、baseline 对比（见 Phase 2）
    """
```

**3b. default_config 更新：** 在 `__init__` 的 `default_config` 字典中，替换相关条目：

```python
        default_config = {
            # 环境参数
            "n_agents": env.n,
            "obs_dim": env.obs_dim,
            "act_dim": env.act_dim,
            "action_scale": env.a_max,
            # 通信与调度器参数
            "k": 4,
            "R_comm": env.R_comm,
            "d_min": env.d_min,
            "schedule_mode": "safety_priority",
            "beta": 1.0,          # AoI 权重
            "tau_ref": 1.0,       # AoI 参考时间（秒）
            "tau_max": 5.0,       # AoI 上限
            # 网络参数
            "msg_dim": 64,
            "hidden_dims": [256, 256],
            "H": 2,
            # PPO 超参数
            "n_steps": 2048,
            "n_epochs": 10,
            "batch_size": 256,
            "gamma": 0.99,
            "lam": 0.95,
            "clip_eps": 0.2,
            "value_coef": 0.5,
            "cost_value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            # 学习率
            "lr_actor": 3e-4,
            "lr_critic": 1e-3,
            "lr_lambda": 0.01,
            # 安全约束
            "safety_budget": 0.1,
            "lambda_init": 0.0,
            "lambda_max": 10.0,
            # 日志
            "log_interval": 10,
            "eval_interval": 50,
        }
```

**3c. 调度器初始化** — 将 `self.scheduler = SafetyPriorityScheduler(...)` 调用更新为：

```python
        self.scheduler = SafetyPriorityScheduler(
            n_agents=n,
            k=self.config["k"],
            d_min=self.config["d_min"],
            schedule_mode=self.config["schedule_mode"],
            v_max=self.config.get("v_max_cbf", env.v_max),
            dt=env.dt,
            beta=self.config.get("beta", 1.0),
            tau_ref=self.config.get("tau_ref", 1.0),
            tau_max=self.config.get("tau_max", 5.0),
        )
```

**3d. 移除 MGDA 分支** — 在 `update()` 方法中，找到以下代码段（约第 402–436 行）：

```python
                # 反向传播（Actor）—— 含 MGDA 梯度调和（v2.0）
                actor_params = (
                    list(self.networks.encoder.parameters())
                    + list(self.networks.aggregator.parameters())
                    + list(self.networks.actor.parameters())
                )
                if lambda_s > self.config.get("lambda_threshold", 0.1):
                    # MGDA：分别计算任务梯度和安全梯度
                    loss_task_only = actor_loss + entropy_loss
                    loss_safe_only = safety_penalty

                    self.actor_optimizer.zero_grad()
                    loss_task_only.backward(retain_graph=True)
                    g_task = torch.cat([
                        p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                        for p in actor_params
                    ])

                    self.actor_optimizer.zero_grad()
                    loss_safe_only.backward(retain_graph=True)
                    g_safe = torch.cat([
                        p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                        for p in actor_params
                    ])

                    w_task, w_safe = mgda_solve(g_task, g_safe)
                    combined_loss = w_task * loss_task_only + w_safe * loss_safe_only

                    self.actor_optimizer.zero_grad()
                    combined_loss.backward()
                else:
                    self.actor_optimizer.zero_grad()
                    total_actor_loss.backward(retain_graph=True)
```

替换为（Phase 1 不用 MGDA，始终使用直接 backward）：

```python
                # 反向传播（Actor）—— Phase 1 无 MGDA，Phase 2 加入
                actor_params = (
                    list(self.networks.encoder.parameters())
                    + list(self.networks.aggregator.parameters())
                    + list(self.networks.actor.parameters())
                )
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
```

**3e. 更新 train/log 打印文本：** 将第 506 行的打印语句改为：

```python
        print(f"[SafeComm-VoI Phase 1] 开始训练，总步数={total_steps}，设备={self.device}")
```

**3f. 更新 save/load 打印文本：** 将第 655 行和第 666 行的 `SafeCommPSched` 改为 `SafeComm-VoI`：

```python
        print(f"[SafeComm-VoI] 模型已保存至 {path}")
```

```python
        print(f"[SafeComm-VoI] 模型已从 {path} 加载，"
              f"迭代={self.n_iterations}，步数={self.total_steps}")
```

- [ ] **Step 4: 移除 safecomm_psched.py 中不再使用的 mgda_solve import**

将第 33 行的 import 中移除 `mgda_solve`（`ppo_utils.py` 仍保留该函数供 Phase 2 使用）：

```python
from algorithms.ppo_utils import (
    RolloutBuffer,
    ppo_clip_loss,
    value_loss,
    normalize_advantages,
    explained_variance,
)
```

- [ ] **Step 5: 运行 test_algorithm.py 确认通过**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_algorithm.py -v
```

预期：所有测试通过

- [ ] **Step 6: 提交**

```bash
cd /root/autodl-tmp/safecomm-marl && git add algorithms/safecomm_psched.py tests/test_algorithm.py
git commit -m "refactor: rename SafeCommPSched -> SafeCommVoI, remove MGDA (Phase 1)"
```

---

## Task 3：更新 configs/default.yaml 和 train.py

**Files:**
- Modify: `configs/default.yaml`
- Modify: `train.py`

### 步骤

- [ ] **Step 1: 更新 configs/default.yaml**

将 `configs/default.yaml` 完整替换为以下内容（新增 VoI 超参数，K_obs 改为 4，删除 PSched 注释）：

```yaml
# SafeComm-VoI Phase 1 默认超参数配置
# 规格参考：docs/superpowers/specs/2026-05-26-safecomm-voi-design.md §2.1

# ============================================================
# 环境参数
# ============================================================
env:
  n_agents: 4              # UAV 数量（Phase 1: 4）
  dt: 0.1                  # 仿真步长（秒）
  max_steps: 200           # 每集最大步数
  d_min: 0.5               # 最小安全距离（米）
  R_comm: 5.0              # 通信感知半径（米）
  v_max: 3.0               # 最大速度（米/秒）
  a_max: 2.0               # 最大加速度（米/秒²）
  arena_size: 10.0         # 飞行区域边长（米）
  formation_type: "circle" # 目标编队形状（circle/line/grid/v_shape）
  w_formation: 1.0         # 编队误差奖励权重
  w_success: 5.0           # 成功奖励
  success_threshold: 0.3   # 成功判定阈值（米）
  K_obs_neighbors: 4       # 观测中包含的最大邻居数（规格默认 K_obs=4）
  seed: 42

# ============================================================
# 通信与 VoI 调度器参数
# ============================================================
comm:
  k: 4                     # 通信预算（每步最多活跃链路数）
  schedule_mode: "safety_priority"  # safety_priority | random | full
  beta: 1.0                # AoI 权重（β=0 退化为纯安全优先，消融用）
  tau_ref: 1.0             # AoI 参考时间（秒），默认取通信周期
  tau_max: 5.0             # AoI 放大上限（截断防止 AoI 无限增长）
  v_max_cbf: 3.0           # CBF 保守缓冲用速度上界（应等于 env.v_max）

# ============================================================
# 网络架构
# ============================================================
network:
  msg_dim: 64              # 消息向量维度
  hidden_dims: [256, 256]  # Actor/Critic MLP 隐藏层
  H: 2                     # 注意力头数（规格默认 H=2）

# ============================================================
# PPO 超参数
# ============================================================
ppo:
  n_steps: 2048            # 每次迭代收集的时间步数
  n_epochs: 10             # 每次迭代 PPO 更新轮数
  batch_size: 256          # mini-batch 大小
  gamma: 0.99              # 折扣因子
  lam: 0.95                # GAE λ
  clip_eps: 0.2            # PPO clip 参数 ε
  value_coef: 0.5          # 价值损失系数
  cost_value_coef: 0.5     # 安全成本价值损失系数
  entropy_coef: 0.01       # 熵奖励系数
  max_grad_norm: 0.5       # 梯度裁剪阈值

# ============================================================
# 学习率
# ============================================================
lr:
  actor: 3.0e-4            # Actor 学习率
  critic: 1.0e-3           # Critic 学习率
  lambda: 0.01             # Lagrangian 乘子学习率（双时间尺度，慢于 π）

# ============================================================
# 安全约束（Lagrangian）
# ============================================================
safety:
  budget: 0.1              # d_safe：每集期望安全成本上限
  lambda_init: 0.0         # Lagrangian 乘子初值
  lambda_max: 10.0         # λ 上界（防止过度惩罚）

# ============================================================
# 训练设置
# ============================================================
training:
  total_steps: 1_000_000   # 总训练步数
  log_interval: 10         # 打印间隔（迭代数）
  eval_interval: 50        # 评估间隔（迭代数）
  eval_episodes: 10        # 评估集数
  save_interval: 100       # 保存间隔（迭代数）
  checkpoint_dir: "checkpoints"
  device: "auto"
  seed: 42

# ============================================================
# 消融配置说明（ablation/ 目录下覆盖字段）
# ============================================================
# ablation/random_scheduler.yaml:  comm.schedule_mode = "random"
# ablation/full_comm.yaml:         comm.k = 9999
# ablation/beta_zero.yaml:         comm.beta = 0.0   （纯安全优先消融）
```

- [ ] **Step 2: 更新 train.py**

将 `train.py` 完整替换为：

```python
"""
SafeComm-VoI Phase 1 训练入口

用法：
  python train.py                              # 使用默认配置
  python train.py --config configs/default.yaml
  python train.py --n_agents 4 --k 4 --total_steps 500000
  python train.py --beta 0.0                  # 消融：纯安全优先
"""

import argparse
import yaml
import os

from envs.uav_formation_env import UAVFormationEnv
from algorithms.safecomm_psched import SafeCommVoI


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def flatten_config(cfg: dict) -> dict:
    """将嵌套 YAML 配置展平为 SafeCommVoI 所需的扁平字典。"""
    comm_cfg = cfg.get("comm", {})
    net_cfg = cfg.get("network", {})
    ppo_cfg = cfg.get("ppo", {})
    lr_cfg = cfg.get("lr", {})
    safety_cfg = cfg.get("safety", {})
    train_cfg = cfg.get("training", {})

    return {
        # 通信与 VoI 调度器
        "k": comm_cfg.get("k", 4),
        "schedule_mode": comm_cfg.get("schedule_mode", "safety_priority"),
        "beta": comm_cfg.get("beta", 1.0),
        "tau_ref": comm_cfg.get("tau_ref", 1.0),
        "tau_max": comm_cfg.get("tau_max", 5.0),
        "v_max_cbf": comm_cfg.get("v_max_cbf", 3.0),
        # 网络
        "msg_dim": net_cfg.get("msg_dim", 64),
        "hidden_dims": net_cfg.get("hidden_dims", [256, 256]),
        "H": net_cfg.get("H", 2),
        # PPO
        "n_steps": ppo_cfg.get("n_steps", 2048),
        "n_epochs": ppo_cfg.get("n_epochs", 10),
        "batch_size": ppo_cfg.get("batch_size", 256),
        "gamma": ppo_cfg.get("gamma", 0.99),
        "lam": ppo_cfg.get("lam", 0.95),
        "clip_eps": ppo_cfg.get("clip_eps", 0.2),
        "value_coef": ppo_cfg.get("value_coef", 0.5),
        "cost_value_coef": ppo_cfg.get("cost_value_coef", 0.5),
        "entropy_coef": ppo_cfg.get("entropy_coef", 0.01),
        "max_grad_norm": ppo_cfg.get("max_grad_norm", 0.5),
        # 学习率
        "lr_actor": lr_cfg.get("actor", 3e-4),
        "lr_critic": lr_cfg.get("critic", 1e-3),
        "lr_lambda": lr_cfg.get("lambda", 0.01),
        # 安全
        "safety_budget": safety_cfg.get("budget", 0.1),
        "lambda_init": safety_cfg.get("lambda_init", 0.0),
        "lambda_max": safety_cfg.get("lambda_max", 10.0),
        # 日志
        "log_interval": train_cfg.get("log_interval", 10),
        "eval_interval": train_cfg.get("eval_interval", 50),
    }


def main():
    parser = argparse.ArgumentParser(description="SafeComm-VoI Phase 1 训练")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n_agents", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None, help="AoI 权重（0=纯安全优先）")
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.n_agents is not None:
        cfg["env"]["n_agents"] = args.n_agents
    if args.k is not None:
        cfg["comm"]["k"] = args.k
    if args.beta is not None:
        cfg["comm"]["beta"] = args.beta
    if args.total_steps is not None:
        cfg["training"]["total_steps"] = args.total_steps
    if args.seed is not None:
        cfg["env"]["seed"] = args.seed
        cfg["training"]["seed"] = args.seed

    env_cfg = cfg.get("env", {})
    train_cfg = cfg.get("training", {})

    env = UAVFormationEnv(
        n_agents=env_cfg.get("n_agents", 4),
        dt=env_cfg.get("dt", 0.1),
        max_steps=env_cfg.get("max_steps", 200),
        d_min=env_cfg.get("d_min", 0.5),
        R_comm=env_cfg.get("R_comm", 5.0),
        v_max=env_cfg.get("v_max", 3.0),
        a_max=env_cfg.get("a_max", 2.0),
        arena_size=env_cfg.get("arena_size", 10.0),
        formation_type=env_cfg.get("formation_type", "circle"),
        w_formation=env_cfg.get("w_formation", 1.0),
        w_success=env_cfg.get("w_success", 5.0),
        success_threshold=env_cfg.get("success_threshold", 0.3),
        K_obs_neighbors=env_cfg.get("K_obs_neighbors", 4),
        seed=env_cfg.get("seed", 42),
    )

    agent_config = flatten_config(cfg)

    agent = SafeCommVoI(
        env=env,
        config=agent_config,
        device=args.device if args.device else train_cfg.get("device", "auto"),
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)

    total_steps = train_cfg.get("total_steps", 1_000_000)
    metrics = agent.train(total_steps=total_steps)

    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    agent.save(os.path.join(checkpoint_dir, "final.pt"))

    print("\n[最终评估]")
    eval_metrics = agent.evaluate(n_episodes=20, deterministic=True)
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行全量测试套件**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/ -v --tb=short
```

预期：所有测试通过（`test_scheduler.py`、`test_algorithm.py`、`test_env.py`、`test_ppo_utils.py`、`test_networks.py`）

- [ ] **Step 4: 提交**

```bash
cd /root/autodl-tmp/safecomm-marl && git add configs/default.yaml train.py
git commit -m "chore: update config and train.py for SafeComm-VoI Phase 1"
```

---

## Task 4：冒烟测试——验证 Phase 1 成功标准

**Files:**
- 只运行测试和短训练，不修改任何文件

### Phase 1 成功标准

1. pytest 全部通过
2. 每步 |E_t| ≤ k
3. λ_s 更新方向正确（成本超预算时 λ_s 上升）
4. β=0 vs β>0 调度差异可见

### 步骤

- [ ] **Step 1: 确认全量 pytest 通过**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/ -v 2>&1 | tail -20
```

预期：`passed` 行数 ≥ 全部测试数，无 `FAILED`

- [ ] **Step 2: 验证 |E_t| ≤ k（短脚本）**

```bash
cd /root/autodl-tmp/safecomm-marl && python -c "
import numpy as np
from envs.uav_formation_env import UAVFormationEnv
from algorithms.safecomm_psched import SafeCommVoI

env = UAVFormationEnv(n_agents=4, d_min=0.5, dt=0.1, max_steps=10, R_comm=5.0)
agent = SafeCommVoI(env, config={'k': 4, 'n_steps': 40, 'n_epochs': 1, 'batch_size': 20,
                                  'msg_dim': 32, 'hidden_dims': [64, 64], 'H': 2,
                                  'safety_budget': 0.5, 'beta': 1.0, 'tau_ref': 1.0, 'tau_max': 5.0},
                    device='cpu')
env.reset()
agent.scheduler.reset_episode(env.positions)
for _ in range(20):
    phys = env.get_physical_graph()
    result = agent.scheduler.schedule_step(env.positions, phys)
    assert len(result['active_edges']) <= 4, f'|E_t|={len(result[\"active_edges\"])} > k=4'
print('OK: |E_t| <= k 在 20 步内始终满足')
"
```

预期输出：`OK: |E_t| <= k 在 20 步内始终满足`

- [ ] **Step 3: 验证 λ_s 更新方向正确（短脚本）**

```bash
cd /root/autodl-tmp/safecomm-marl && python -c "
import numpy as np
from envs.uav_formation_env import UAVFormationEnv
from algorithms.safecomm_psched import SafeCommVoI

env = UAVFormationEnv(n_agents=4, d_min=0.3, dt=0.1, max_steps=10, R_comm=5.0)
cfg = {'k': 4, 'n_steps': 40, 'n_epochs': 2, 'batch_size': 20,
       'msg_dim': 32, 'hidden_dims': [64, 64], 'H': 2,
       'safety_budget': 0.001,  # 极低预算，必然超限
       'beta': 1.0, 'tau_ref': 1.0, 'tau_max': 5.0}
agent = SafeCommVoI(env, config=cfg, device='cpu')
lam_before = agent.lambda_s
agent.collect_rollout()
agent.update()
lam_after = agent.lambda_s
assert lam_after >= lam_before, f'λ_s 应 >= 初值 {lam_before}，实际 {lam_after}'
print(f'OK: λ_s {lam_before:.4f} -> {lam_after:.4f}（方向正确）')
"
```

预期输出：`OK: λ_s 0.0000 -> X.XXXX（方向正确）`，X > 0

- [ ] **Step 4: 验证 β=0 vs β>0 调度差异**

```bash
cd /root/autodl-tmp/safecomm-marl && python -c "
import numpy as np
from algorithms.scheduler import SafetyPriorityScheduler

positions = np.array([[0,0,0],[3,0,0],[10,0,0]], dtype=np.float64)
edges = [(0,1),(0,2),(1,2)]

s_voi = SafetyPriorityScheduler(n_agents=3, k=1, d_min=0.5, v_max=0.0, dt=0.1,
                                 beta=2.0, tau_ref=0.5, tau_max=10.0)
s_safe = SafetyPriorityScheduler(n_agents=3, k=1, d_min=0.5, v_max=0.0, dt=0.1,
                                  beta=0.0, tau_ref=0.5, tau_max=10.0)
s_voi.reset_episode(positions)
s_safe.reset_episode(positions)

# (0,2) 长期未通信：20 步 AoI
for _ in range(20):
    s_voi.update_history([(0,1)], positions)
    s_safe.update_history([(0,1)], positions)

prios_voi = s_voi.compute_voi_priority(positions, edges)
prios_safe = s_safe.compute_voi_priority(positions, edges)

print('VoI 优先级 (β=2.0):', {k: f'{v:.3f}' for k,v in sorted(prios_voi.items())})
print('安全优先（β=0.0）:', {k: f'{v:.3f}' for k,v in sorted(prios_safe.items())})
ratio_voi = prios_voi[(0,2)] / prios_voi[(0,1)]
ratio_safe = prios_safe[(0,2)] / prios_safe[(0,1)]
print(f'(0,2)/(0,1) 比值: β=2.0 → {ratio_voi:.3f}, β=0.0 → {ratio_safe:.3f}')
assert abs(ratio_voi - ratio_safe) > 1e-3, 'β=0 与 β>0 调度优先级比应不同'
print('OK: β=0 vs β>0 调度差异可见')
"
```

预期输出：两组优先级比值可见差异，末行打印 `OK: β=0 vs β>0 调度差异可见`

- [ ] **Step 5: 提交冒烟测试结果（如有新文件）**

若无新文件需要提交，跳过此步。否则：

```bash
cd /root/autodl-tmp/safecomm-marl && git add -p  # 仅选择需要的修改
git commit -m "test: verify Phase 1 success criteria (smoke tests pass)"
```

---

## 自审检查单（实施前参考）

| 条目 | 状态 |
|------|------|
| Proposition 1（|E_t|≤k）由 TopK 保证，调度器未破坏 | ✓ 覆盖 |
| h_sched 公式使用 p_hat_j（最近通信已知位置）而非当前位置 | ✓ 覆盖 |
| h_sched 只用于调度排序，未传入 QP（Phase 1 无 QP）| ✓ Phase 1 无 QP |
| β=0 严格退化为纯安全优先（消融基线） | ✓ 测试覆盖 |
| AoI = Δt_comm（秒）/ τ_ref，截断于 τ_max | ✓ 测试覆盖 |
| MGDA 从 Phase 1 移除（Phase 2 再加） | ✓ 任务 2 涵盖 |
| K_obs 更新为 4（规格默认值） | ✓ 任务 3 涵盖 |
| v_max_cbf 与 env.v_max 对齐（3.0 m/s） | ✓ 任务 3 涵盖 |
| λ_s 更新使用每步平均成本估计（非 J_safe 真值）| ✓ 现有逻辑不变 |
