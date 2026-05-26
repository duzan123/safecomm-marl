# SafeComm-VoI Phase 2 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Phase 1 基础上完成 SafeComm-VoI 完整方案：加入 C+ HOCBF-QP 执行安全过滤、双约束 Lagrangian（λ_int + safety gate）、层级 MGDA、全部 baseline，并输出可直接用于论文 §5 的主结果表和消融表。

**Architecture:** Phase 2 在 Phase 1 的 SafeCommVoI 主类上分三层扩展：(1) 新增独立 `HOCBFQPShield` 模块处理 per-agent QP 过滤，rollout 同时记录 a^π 和 a^*；(2) 扩展 RolloutBuffer 存储 intervention cost，IntCostCritic 估计 V^C_int，update() 中加入 safety gate EMA 和条件式 λ_int 更新；(3) 基于配置参数的 baseline 包装器类和独立 MACPO 类，evaluate.py 支持 with/without QP 两种评估模式。

**Tech Stack:** Python 3.x, NumPy, PyTorch, scipy（SLSQP QP solver；可在稳定后替换为 OSQP/CVXPY 后端），pytest, PyYAML

**外部代码复用边界（Phase 2）：**
- `external/Multi-Agent-Constrained-Policy-Optimisation` 只参考 MACPO/MAPPO-Lagrangian 的 cost critic、dual update、baseline 指标和实验命名；许可证文件含冲突标记，且 runner/env 接口与本项目不匹配，不复制代码。
- `external/gcbfplus` 可参考 pairwise CBF、safe/unsafe mask、图环境诊断和 GCBF+ 作为独立 baseline 的组织方式；SafeComm 主安全层仍自写 C+ HOCBF-QP。
- `external/macbf` 可参考 neural barrier 的 mask、loss 和可视化指标；不把 TF1 neural CBF 迁入 PyTorch 主线，也不替代 HOCBF-QP。
- `external/DGN` 无明确许可证，只参考 DGN+MAPPO baseline 的图通信设定；本项目用 PyTorch masked GAT/message passing 自写。
- `external/on-policy` 可继续参考 PPO 更新细节；Phase 2 不引入其完整 runner。

---

## 文件结构

| 文件 | 操作 | 说明 |
|------|------|------|
| `algorithms/hocbf_qp.py` | 新建 | C+ HOCBF-QP Shield，per-agent SLSQP，含 slack |
| `algorithms/networks.py` | 修改 | 新增 `IntCostCritic`，SafeCommNetworks 加 `int_cost_critic` |
| `algorithms/ppo_utils.py` | 修改 | RolloutBuffer 新增 `int_costs`、`int_cost_values`，GAE 扩展 |
| `algorithms/safecomm_psched.py` | 修改 | 集成 QP Shield、λ_int、safety gate EMA、层级 MGDA，save/load 扩展 |
| `baselines/__init__.py` | 新建 | 空模块文件 |
| `baselines/mappo.py` | 新建 | MAPPO（全通信，无安全）配置包装器 |
| `baselines/mappo_lag.py` | 新建 | MAPPO-Lag（全通信 + λ_s）配置包装器 |
| `baselines/dgn_mappo.py` | 新建 | DGN+MAPPO（random TopK，无安全）包装器 |
| `baselines/hocbf_ppo.py` | 新建 | HOCBF-PPO（全通信 + QP + λ_s）包装器 |
| `baselines/random_topk.py` | 新建 | SafeComm-RandomTopK（全安全 + random 调度器）包装器 |
| `baselines/macpo.py` | 新建 | MACPO（全通信 + 安全梯度投影，独立子类）|
| `evaluate.py` | 新建 | 评估脚本，支持 `--no_qp` 模式（Shield Independence Eval）|
| `configs/ablation/beta_zero.yaml` | 新建 | w/o AoI 消融 |
| `configs/ablation/no_cbf_buffer.yaml` | 新建 | w/o 保守缓冲消融 |
| `configs/ablation/no_hocbf_qp.yaml` | 新建 | w/o HOCBF-QP 消融 |
| `configs/ablation/no_lambda_int.yaml` | 新建 | w/o λ_int 消融 |
| `configs/ablation/no_mgda.yaml` | 新建 | w/o MGDA 消融 |
| `configs/ablation/random_topk.yaml` | 新建 | SafeComm-RandomTopK 消融 |
| `tests/test_hocbf_qp.py` | 新建 | HOCBF-QP 单元测试 |
| `tests/test_baselines.py` | 新建 | baselines 冒烟测试 |

**Phase 1 不修改的文件（保持不变）：** `algorithms/scheduler.py`、`envs/uav_formation_env.py`、`train.py`

---

## Task 1：C+ HOCBF-QP Shield

**Files:**
- Create: `algorithms/hocbf_qp.py`
- Test: `tests/test_hocbf_qp.py`

**实现边界：** 该模块按本项目 double-integrator C+ HOCBF 公式自实现。`gcbfplus` 和 `macbf` 只能用于理解 pairwise barrier mask、safe/unsafe 诊断和可视化指标；不要把 neural/graph CBF 代码复制进 `hocbf_qp.py`，也不要把它们作为 HOCBF-QP 的替代实现。

### 步骤

- [ ] **Step 1: 写失败测试**

新建 `tests/test_hocbf_qp.py`：

```python
"""
HOCBF-QP Shield 单元测试（规格 §2.2）

覆盖：
  1. 无邻居时原样返回策略动作
  2. 安全区外：QP 不修改动作（已安全）
  3. 危险接近时：QP 强制介入（intervention > 0）
  4. slack 在多约束冲突时允许不可行约束放松
  5. Proposition 2：dt_comm 越大 → d_eff 越大 → 约束越紧
  6. filter_all_agents 接口正确性
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from algorithms.hocbf_qp import HOCBFQPShield


class TestNoNeighbors:
    def test_no_neighbors_returns_a_pi(self):
        """无邻居时原样返回策略动作。"""
        shield = HOCBFQPShield(d_min=0.5, a_max=2.0)
        a_pi = np.array([1.0, 0.5, -0.3])
        a_star, xi, interv = shield.filter_action(
            agent_id=0,
            pos_i=np.zeros(3),
            vel_i=np.zeros(3),
            a_pi=a_pi,
            known_neighbors=[],
        )
        assert np.allclose(a_star, a_pi)
        assert xi == 0.0
        assert interv == 0.0


class TestSafeRegion:
    def test_safe_pair_no_intervention(self):
        """两 UAV 距离 > d_eff，策略动作远离对方，QP 不应介入。"""
        shield = HOCBFQPShield(d_min=0.5, a_max=2.0, alpha1=1.0, alpha2=1.0, v_max=0.0, dt=0.1)
        # UAV 0 at origin moving right, UAV 1 far away at (10, 0, 0)
        a_pi = np.array([1.0, 0.0, 0.0])   # moving away from UAV 1 → safe
        nbrs = [{'id': 1, 'pos_hat': np.array([10.0, 0.0, 0.0]),
                  'vel_hat': np.zeros(3), 'dt_comm': 0.0}]
        a_star, xi, interv = shield.filter_action(
            agent_id=0, pos_i=np.zeros(3), vel_i=np.zeros(3),
            a_pi=a_pi, known_neighbors=nbrs,
        )
        # Should be close to a_pi (minimal intervention)
        assert interv < 0.5, f"Expected minimal intervention, got {interv}"


class TestDangerousApproach:
    def test_head_on_collision_intervenes(self):
        """两 UAV 正面高速接近时，QP 必须介入（intervention > 0）。"""
        shield = HOCBFQPShield(d_min=0.5, a_max=2.0, alpha1=1.0, alpha2=1.0, v_max=0.0, dt=0.1)
        # UAV 0 at (0,0,0) moving toward UAV 1, UAV 1 at (0.8,0,0)
        # Distance = 0.8, d_eff = 0.5, margin = 0.3 (small)
        # With v_i = [2,0,0] (moving toward UAV 1), a_pi = [2,0,0] (still accelerating toward)
        pos_i = np.array([0.0, 0.0, 0.0])
        vel_i = np.array([2.0, 0.0, 0.0])   # heading toward UAV 1
        a_pi = np.array([2.0, 0.0, 0.0])    # still accelerating toward
        nbrs = [{'id': 1, 'pos_hat': np.array([0.8, 0.0, 0.0]),
                  'vel_hat': np.zeros(3), 'dt_comm': 0.0}]
        a_star, xi, interv = shield.filter_action(
            agent_id=0, pos_i=pos_i, vel_i=vel_i,
            a_pi=a_pi, known_neighbors=nbrs,
        )
        assert interv > 0.01, f"Expected QP intervention, got intervention={interv}"

    def test_qp_output_bounded_by_a_max(self):
        """QP 输出动作不超过 a_max。"""
        shield = HOCBFQPShield(d_min=0.5, a_max=2.0)
        pos_i = np.array([0.0, 0.0, 0.0])
        vel_i = np.array([2.0, 0.0, 0.0])
        a_pi = np.array([2.0, 0.0, 0.0])
        nbrs = [{'id': 1, 'pos_hat': np.array([0.8, 0.0, 0.0]),
                  'vel_hat': np.zeros(3), 'dt_comm': 0.0}]
        a_star, xi, _ = shield.filter_action(
            agent_id=0, pos_i=pos_i, vel_i=vel_i, a_pi=a_pi, known_neighbors=nbrs,
        )
        assert np.all(np.abs(a_star) <= 2.0 + 1e-6), f"a_star={a_star} exceeds a_max=2.0"


class TestSlackVariable:
    def test_slack_nonnegative(self):
        """Slack 变量 ξ ≥ 0 始终成立。"""
        shield = HOCBFQPShield(d_min=0.5, a_max=2.0, rho=100.0)
        pos_i = np.array([0.0, 0.0, 0.0])
        vel_i = np.array([3.0, 0.0, 0.0])
        a_pi = np.array([2.0, 0.0, 0.0])
        # Multiple close neighbors from different directions
        nbrs = [
            {'id': j, 'pos_hat': np.array([0.6, 0.0, 0.0]) * (1 if j % 2 == 0 else -1),
             'vel_hat': np.zeros(3), 'dt_comm': 0.0}
            for j in range(1, 4)
        ]
        _, xi, _ = shield.filter_action(
            agent_id=0, pos_i=pos_i, vel_i=vel_i, a_pi=a_pi, known_neighbors=nbrs,
        )
        assert xi >= -1e-6, f"Slack ξ={xi} should be >= 0"


class TestConservativeBuffer:
    def test_larger_dt_comm_tightens_constraint(self):
        """dt_comm 越大 → d_eff 越大 → 约束越紧（Proposition 2 Part iii）。"""
        shield = HOCBFQPShield(d_min=0.5, a_max=2.0, v_max=1.0, dt=0.1)
        pos_i = np.zeros(3)
        vel_i = np.array([1.0, 0.0, 0.0])
        a_pi = np.array([1.0, 0.0, 0.0])
        nbr_base = {'id': 1, 'pos_hat': np.array([1.5, 0.0, 0.0]), 'vel_hat': np.zeros(3)}

        nbrs_fresh = [{**nbr_base, 'dt_comm': 0.0}]
        nbrs_stale = [{**nbr_base, 'dt_comm': 1.0}]  # 1 second stale

        _, _, interv_fresh = shield.filter_action(
            agent_id=0, pos_i=pos_i, vel_i=vel_i, a_pi=a_pi, known_neighbors=nbrs_fresh,
        )
        _, _, interv_stale = shield.filter_action(
            agent_id=0, pos_i=pos_i, vel_i=vel_i, a_pi=a_pi, known_neighbors=nbrs_stale,
        )
        assert interv_stale >= interv_fresh - 1e-6, (
            f"Stale comm should lead to >= intervention: fresh={interv_fresh}, stale={interv_stale}"
        )


class TestFilterAllAgents:
    def test_filter_all_agents_shapes(self):
        """filter_all_agents 返回正确形状的数组。"""
        shield = HOCBFQPShield(d_min=0.5, a_max=2.0, v_max=1.0, dt=0.1)
        N = 4
        positions = np.random.rand(N, 3) * 5
        velocities = np.random.rand(N, 3)
        actions_pi = np.random.uniform(-2, 2, (N, 3))
        known_pos = positions.copy()
        known_vel = velocities.copy()
        last_comm_time = np.zeros((N, N), dtype=int)
        neighbor_lists = {i: [j for j in range(N) if j != i] for i in range(N)}

        a_star, xis, interventions = shield.filter_all_agents(
            positions, velocities, actions_pi, known_pos, known_vel,
            last_comm_time, neighbor_lists,
        )
        assert a_star.shape == (N, 3)
        assert xis.shape == (N,)
        assert interventions.shape == (N,)
        assert np.all(np.abs(a_star) <= 2.0 + 1e-6)
        assert np.all(xis >= -1e-6)
        assert np.all(interventions >= -1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

- [ ] **Step 2: 运行失败测试**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_hocbf_qp.py -v 2>&1 | head -20
```

预期：`ModuleNotFoundError: No module named 'algorithms.hocbf_qp'`

- [ ] **Step 3: 实现 algorithms/hocbf_qp.py**

新建 `algorithms/hocbf_qp.py`：

```python
"""
C+ HOCBF-QP 执行安全过滤（规格 §2.2）

对每个 agent i，解以下 per-agent QP（分散式执行）：
  a^*_i = argmin_{a_i, ξ_i ≥ 0}  ||a_i - a^π_i||^2 + ρ ξ_i^2
           s.t. 对所有已知邻居 j：
                  2 r_ij^T a_i + ξ_i ≥ rhs_j        (HOCBF 约束)
                  ||a_i||_∞ ≤ a_max                  (动作边界)

变量 x = [a_x, a_y, a_z, ξ] ∈ R^4
使用 scipy SLSQP 求解（tiny QP，4 变量，< 2ms per agent）。
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional


class HOCBFQPShield:
    """
    C+ HOCBF-QP 分散式执行安全过滤（规格 §2.2）。

    参数说明（来自规格）：
      d_min:  最小安全距离（CBF 硬边界）
      a_max:  最大动作幅度（L_inf 约束）
      alpha1: HOCBF 一阶增益（ψ_1 = ψ̇_0 + α_1 ψ_0）
      alpha2: HOCBF 二阶增益（约束：... + α_2 ψ_1 ≥ 0）
      rho:    slack 惩罚权重（防止多约束冲突导致 QP 不可行）
      v_max:  CBF 保守缓冲速度上界
      dt:     仿真步长（用于将 last_comm_step 转换为时间）
    """

    def __init__(
        self,
        d_min: float = 0.5,
        a_max: float = 2.0,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        rho: float = 100.0,
        v_max: float = 3.0,
        dt: float = 0.1,
    ):
        self.d_min = d_min
        self.a_max = a_max
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.rho = rho
        self.v_max = v_max
        self.dt = dt

    def _build_constraint(
        self,
        r_ij: np.ndarray,  # (3,) p_i - p_hat_j
        v_ij: np.ndarray,  # (3,) v_i - v_hat_j
        dt_comm: float,    # 秒
    ) -> Tuple[np.ndarray, float]:
        """
        为邻居 j 构建 HOCBF 线性约束（规格 §2.2）。

        约束形式：g_j @ x - rhs_j >= 0
        其中 x = [a_x, a_y, a_z, ξ]，g_j = [2*r_ij; 1] ∈ R^4

        Returns:
            g_j: (4,) 约束系数向量
            rhs_j: 约束右侧标量
        """
        d_buf = self.v_max * dt_comm
        d_eff = self.d_min + d_buf
        norm_r = float(np.linalg.norm(r_ij))

        # h^{C+} = ||r_ij||^2 - d_eff^2
        h = norm_r ** 2 - d_eff ** 2

        # ψ̇_0 = dh/dt = 2 r_ij^T v_ij
        dh_dt = 2.0 * float(r_ij @ v_ij)

        # ψ_1 = ψ̇_0 + α_1 * h
        psi1 = dh_dt + self.alpha1 * h

        # 鲁棒裕量：m_acc = 2 * a_max * ||r_ij||（worst-case 邻居加速度贡献）
        m_acc = 2.0 * self.a_max * norm_r

        # 约束：2 r_ij^T a_i + 2||v_ij||^2 + α_1 ψ̇_0 + α_2 ψ_1 - m_acc + ξ >= 0
        # 移项 → g_j @ [a_i; ξ] >= rhs_j
        constant_terms = (
            2.0 * float(v_ij @ v_ij)
            + self.alpha1 * dh_dt
            + self.alpha2 * psi1
            - m_acc
        )
        rhs_j = -constant_terms  # 2 r_ij^T a_i + ξ >= rhs_j

        g_j = np.append(2.0 * r_ij, 1.0)  # (4,)
        return g_j, rhs_j

    def filter_action(
        self,
        agent_id: int,
        pos_i: np.ndarray,         # (3,)
        vel_i: np.ndarray,         # (3,)
        a_pi: np.ndarray,          # (3,) 策略原始输出（已缩放到 a_max）
        known_neighbors: List[Dict],
        # 每个 dict: {'id': int, 'pos_hat': (3,), 'vel_hat': (3,), 'dt_comm': float}
    ) -> Tuple[np.ndarray, float, float]:
        """
        解 per-agent HOCBF-QP，返回过滤后动作。

        Returns:
            a_star: (3,) QP 过滤后动作
            xi:     slack 变量值（> 0 表示某约束被放松）
            intervention: ||a_pi - a_star||^2
        """
        if not known_neighbors:
            return a_pi.copy(), 0.0, 0.0

        # 构建约束列表
        scipy_constraints = []

        for nbr in known_neighbors:
            p_hat_j = np.asarray(nbr['pos_hat'])
            v_hat_j = np.asarray(nbr['vel_hat'])
            dt_comm = float(nbr.get('dt_comm', 0.0))

            r_ij = pos_i - p_hat_j
            v_ij = vel_i - v_hat_j

            norm_r = float(np.linalg.norm(r_ij))
            if norm_r < 1e-8:
                # 极端情况：两 agent 重叠，强制使用大 slack
                continue

            g_j, rhs_j = self._build_constraint(r_ij, v_ij, dt_comm)
            # scipy ineq: fun(x) >= 0
            scipy_constraints.append({
                'type': 'ineq',
                'fun': lambda x, g=g_j, r=rhs_j: float(g @ x) - r,
                'jac': lambda x, g=g_j: g,
            })

        # ξ ≥ 0
        scipy_constraints.append({
            'type': 'ineq',
            'fun': lambda x: x[3],
            'jac': lambda x: np.array([0.0, 0.0, 0.0, 1.0]),
        })

        # 边界：a ∈ [-a_max, a_max]^3, ξ ≥ 0
        bounds = [(-self.a_max, self.a_max)] * 3 + [(0.0, None)]

        # 初始猜测：策略动作 + ξ=0
        x0 = np.append(np.clip(a_pi, -self.a_max, self.a_max), 0.0)

        def objective(x):
            da = x[:3] - a_pi
            return float(da @ da) + self.rho * float(x[3] ** 2)

        def grad_objective(x):
            g = np.zeros(4)
            g[:3] = 2.0 * (x[:3] - a_pi)
            g[3] = 2.0 * self.rho * x[3]
            return g

        result = minimize(
            objective,
            x0,
            jac=grad_objective,
            constraints=scipy_constraints,
            bounds=bounds,
            method='SLSQP',
            options={'ftol': 1e-9, 'maxiter': 200, 'disp': False},
        )

        a_star = np.clip(result.x[:3], -self.a_max, self.a_max)
        xi = max(0.0, float(result.x[3]))
        intervention = float(np.sum((a_pi - a_star) ** 2))

        return a_star, xi, intervention

    def filter_all_agents(
        self,
        positions: np.ndarray,       # (N, 3) 当前真实位置
        velocities: np.ndarray,      # (N, 3) 当前真实速度
        actions_pi: np.ndarray,      # (N, 3) 策略原始动作（已缩放）
        known_pos: np.ndarray,       # (N, 3) 最近通信时已知位置 p_hat
        known_vel: np.ndarray,       # (N, 3) 最近通信时已知速度 v_hat
        last_comm_time: np.ndarray,  # (N, N) int，距上次通信的步数
        neighbor_lists: Dict[int, List[int]],  # 当前活跃通信图（此处仅为接口参考）
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对所有 agent 执行 per-agent QP 过滤。

        注意：QP 约束对象为"所有已知邻居"（即所有 j ≠ i），
        使用 known_pos/known_vel（最近通信时的保守估计）。

        Returns:
            actions_star: (N, 3)
            xis:          (N,)
            interventions:(N,)
        """
        N = positions.shape[0]
        actions_star = np.zeros_like(actions_pi)
        xis = np.zeros(N)
        interventions = np.zeros(N)

        for i in range(N):
            known_nbrs = []
            for j in range(N):
                if j == i:
                    continue
                dt_comm_steps = int(last_comm_time[i, j])
                known_nbrs.append({
                    'id': j,
                    'pos_hat': known_pos[j].copy(),
                    'vel_hat': known_vel[j].copy(),
                    'dt_comm': float(dt_comm_steps) * self.dt,
                })

            a_star, xi, interv = self.filter_action(
                agent_id=i,
                pos_i=positions[i],
                vel_i=velocities[i],
                a_pi=actions_pi[i],
                known_neighbors=known_nbrs,
            )
            actions_star[i] = a_star
            xis[i] = xi
            interventions[i] = interv

        return actions_star, xis, interventions
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_hocbf_qp.py -v
```

预期：所有测试 `PASSED`（除少数可能因几何条件边界情况 skip 的用例）

- [ ] **Step 5: 提交**

```bash
cd /root/autodl-tmp/safecomm-marl && git add algorithms/hocbf_qp.py tests/test_hocbf_qp.py
git commit -m "feat: add C+ HOCBF-QP shield with SLSQP per-agent filter (Phase 2)"
```

---

## Task 2：网络和缓冲区扩展

**Files:**
- Modify: `algorithms/networks.py`（新增 `IntCostCritic`，SafeCommNetworks 加 `int_cost_critic`）
- Modify: `algorithms/ppo_utils.py`（RolloutBuffer 新增 `int_costs`）

### 背景

Phase 2 需要：
1. `IntCostCritic`：估计 V^C_int（intervention cost 的价值函数），与 CostCritic 结构相同
2. `RolloutBuffer`：新增 `int_costs [T, n_agents]` 和 `int_cost_values [T+1]` 字段
3. `get_training_data()` 输出新增 `int_cost_advantages` 和 `int_cost_returns`

### 步骤

- [ ] **Step 1: 在 networks.py 中新增 IntCostCritic 并更新 SafeCommNetworks**

在 `CostCritic` 类定义之后（约第 463 行后），添加：

```python
class IntCostCritic(nn.Module):
    """
    集中式 intervention cost 价值网络 V_φ^{int}。

    结构与 CostCritic 相同，用于估计 J_int = E[Σ_t Σ_i ||a^π - a^*||^2]。
    """

    def __init__(self, global_state_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.int_cost_value_net = build_mlp(
            input_dim=global_state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation="tanh",
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.int_cost_value_net(global_state).squeeze(-1)
```

在 `SafeCommNetworks.__init__` 的 `self.cost_critic = CostCritic(...)` 之后添加：

```python
        self.int_cost_critic = IntCostCritic(
            global_state_dim=global_state_dim,
            hidden_dims=hidden_dims,
        )
```

在 `get_values` 方法中扩展返回值（返回三元组）：

```python
    def get_values(
        self,
        global_states: torch.Tensor,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量计算价值估计。

        Returns:
            values:          [batch] 任务价值
            cost_values:     [batch] 安全成本价值
            int_cost_values: [batch] intervention cost 价值（Phase 2）
        """
        values = self.critic(global_states)
        cost_values = self.cost_critic(global_states)
        int_cost_values = self.int_cost_critic(global_states)
        return values, cost_values, int_cost_values
```

在 `act` 方法中：原返回 `(actions, log_probs, values, cost_values)` → 扩展为：

```python
        # 在 act() 末尾，value 和 cost_value 计算后添加：
        int_cost_value = self.int_cost_critic(global_state.unsqueeze(0)).squeeze().cpu().numpy()

        # 修改最终 return：
        return actions, log_probs, values, cost_values, np.full(self.n_agents, float(int_cost_value))
```

在 `evaluate_actions_batch` 方法中，扩展返回值以包含 int_cost_values：

```python
        # 价值估计（原 get_values 返回 2 元组，现改为 3 元组）
        values, cost_values, int_cost_values = self.get_values(global_states)

        return log_probs, entropy, values, cost_values, int_cost_values, agg_msgs
```

- [ ] **Step 2: 在 ppo_utils.py 中扩展 RolloutBuffer**

在 `RolloutBuffer.reset()` 方法中的 `self.adj_matrices = ...` 之后添加：

```python
        # Phase 2 新增：intervention cost
        self.int_costs = np.zeros((T, N), dtype=np.float32)
        self.int_cost_values = np.zeros(T + 1, dtype=np.float32)
```

在 `add()` 方法的参数列表中添加新参数（有默认值，向后兼容）：

```python
    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: "List[float] | np.ndarray",
        costs: "List[float] | np.ndarray",
        value: float,
        cost_value: float,
        done: bool,
        global_state: np.ndarray,
        adj_matrix: np.ndarray,
        int_cost: Optional["np.ndarray"] = None,   # (n_agents,) 新增
        int_cost_value: float = 0.0,               # 新增
    ):
```

在 `add()` 方法体中，`self.adj_matrices[t] = adj_matrix` 之后添加：

```python
        if int_cost is not None:
            self.int_costs[t] = int_cost
        self.int_cost_values[t] = int_cost_value
```

在 `finish_rollout()` 中添加参数：

```python
    def finish_rollout(
        self,
        last_value: float,
        last_cost_value: float,
        last_int_cost_value: float = 0.0,  # 新增
    ):
        self.values[self.ptr] = last_value
        self.cost_values[self.ptr] = last_cost_value
        self.int_cost_values[self.ptr] = last_int_cost_value
```

在 `compute_advantages_and_returns()` 中添加 int_cost GAE：

```python
    def compute_advantages_and_returns(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        normalize_adv: bool = True,
        normalize_cost_adv: bool = True,
        normalize_int_adv: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            advantages, returns, cost_advantages, cost_returns, int_cost_advantages, int_cost_returns
        """
        T = self.ptr
        mean_rewards = self.rewards[:T].mean(axis=1)
        mean_costs = self.costs[:T].mean(axis=1)
        mean_int_costs = self.int_costs[:T].mean(axis=1)

        advantages, returns = compute_gae(
            mean_rewards, self.values[:T + 1], self.dones[:T], gamma, lam
        )
        cost_advantages, cost_returns = compute_gae(
            mean_costs, self.cost_values[:T + 1], self.dones[:T], gamma, lam
        )
        int_cost_advantages, int_cost_returns = compute_gae(
            mean_int_costs, self.int_cost_values[:T + 1], self.dones[:T], gamma, lam
        )

        if normalize_adv and advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if normalize_cost_adv and cost_advantages.std() > 1e-6:
            cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
        if normalize_int_adv and int_cost_advantages.std() > 1e-6:
            int_cost_advantages = (int_cost_advantages - int_cost_advantages.mean()) / (int_cost_advantages.std() + 1e-8)

        return advantages, returns, cost_advantages, cost_returns, int_cost_advantages, int_cost_returns
```

更新 `get_training_data()` 的返回字典（使用新的 6-tuple 返回）：

```python
    def get_training_data(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        normalize_adv: bool = True,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        T = self.ptr
        advantages, returns, cost_advantages, cost_returns, int_cost_advantages, int_cost_returns = (
            self.compute_advantages_and_returns(gamma, lam, normalize_adv)
        )

        def to_tensor(arr):
            return torch.FloatTensor(arr).to(device)

        return {
            "obs": to_tensor(self.obs[:T]),
            "actions": to_tensor(self.actions[:T]),
            "log_probs_old": to_tensor(self.log_probs[:T]),
            "advantages": to_tensor(advantages),
            "returns": to_tensor(returns),
            "cost_advantages": to_tensor(cost_advantages),
            "cost_returns": to_tensor(cost_returns),
            "int_cost_advantages": to_tensor(int_cost_advantages),
            "int_cost_returns": to_tensor(int_cost_returns),
            "values_old": to_tensor(self.values[:T]),
            "cost_values_old": to_tensor(self.cost_values[:T]),
            "int_cost_values_old": to_tensor(self.int_cost_values[:T]),
            "global_states": to_tensor(self.global_states[:T]),
            "adj_matrices": to_tensor(self.adj_matrices[:T]),
            "dones": torch.BoolTensor(self.dones[:T]).to(device),
        }
```

- [ ] **Step 3: 运行现有测试确认无回归**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_networks.py tests/test_ppo_utils.py -v 2>&1 | tail -20
```

预期：所有测试通过（注意 `act()` 和 `evaluate_actions_batch()` 的返回值变化，需检查 test_networks.py 是否有依赖这些返回值元组长度的测试，若有则更新）

- [ ] **Step 4: 提交**

```bash
cd /root/autodl-tmp/safecomm-marl && git add algorithms/networks.py algorithms/ppo_utils.py
git commit -m "feat: add IntCostCritic and extend RolloutBuffer for int_cost (Phase 2)"
```

---

## Task 3：双动作 collect_rollout（集成 HOCBF-QP）

**Files:**
- Modify: `algorithms/safecomm_psched.py`

### 背景

Phase 2 rollout 关键变化（规格 §2.2 Rollout 记录要求）：
- 环境执行 `a^*_i`（QP 过滤后）
- PPO log probability 使用 `a^π_i`（策略原始输出，存入 buffer）
- 新增 `int_cost = ||a^π_i - a^*_i||^2` 存入 buffer

### 步骤

- [ ] **Step 1: 更新 SafeCommVoI.__init__() 中的 imports 和初始化**

在 `safecomm_psched.py` 顶部 imports 区域添加：

```python
from algorithms.hocbf_qp import HOCBFQPShield
```

在 `__init__` 的 `default_config` 中新增 Phase 2 参数：

```python
            # Phase 2：HOCBF-QP Shield
            "use_hocbf_qp": True,          # 是否启用 QP 过滤（False = 消融 w/o HOCBF-QP）
            "alpha1": 1.0,                 # HOCBF 一阶增益
            "alpha2": 1.0,                 # HOCBF 二阶增益
            "qp_rho": 100.0,               # slack 惩罚权重
            # Phase 2：λ_int + safety gate
            "lambda_int_init": 0.0,        # λ_int 初值
            "lambda_int_max": 10.0,        # λ_int 上界
            "lr_lambda_int": 0.005,        # λ_int 更新步长（慢于 λ_s）
            "safety_budget_int": 0.5,      # d_int：intervention cost 上限
            "safety_gate_ema_window": 5,   # EMA 窗口 M（迭代数）
            "safety_gate_tolerance": 0.02, # δ_s：安全门阈值容差
            # Phase 2：MGDA
            "use_mgda": True,              # 是否启用层级 MGDA
            "lambda_threshold": 0.1,       # λ_s MGDA 激活阈值
            "lambda_int_threshold": 0.05,  # λ_int MGDA 激活阈值
```

在 `__init__` 的 Lagrangian 状态初始化区域添加：

```python
        # Lagrangian 乘子（Phase 2 扩展）
        self.lambda_s = self.config["lambda_init"]
        self.lambda_max = self.config["lambda_max"]
        self.lambda_int = self.config["lambda_int_init"]
        self.lambda_int_max = self.config["lambda_int_max"]

        # Safety gate EMA 状态
        self._j_safe_ema: Optional[float] = None  # EMA 初值（None = 未初始化）
        self._ema_alpha = 1.0 / max(self.config["safety_gate_ema_window"], 1)
        self.safe_gate_active: bool = False        # 当前 safety gate 状态
```

在 `__init__` 的 Critic 初始化区域添加 `int_cost_critic` 的 optimizer：

```python
        # 扩展 critic params 以包含 int_cost_critic
        critic_params = (
            list(self.networks.critic.parameters())
            + list(self.networks.cost_critic.parameters())
            + list(self.networks.int_cost_critic.parameters())   # Phase 2 新增
        )
```

在 buffer 初始化后添加 QP Shield 初始化：

```python
        # QP Shield（Phase 2）
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

- [ ] **Step 2: 更新 collect_rollout() 集成 QP**

找到 `collect_rollout()` 中的步进部分（约第 223–255 行的循环体），将以下核心代码段：

```python
            # 3. 策略推断（无梯度）
            actions, log_probs, values, cost_values = self.networks.act(...)
            # 4. 将 [-1,1] 动作缩放到 [-a_max, a_max]
            scaled_actions = actions * self.config["action_scale"]
            ...
            # 6. 环境步进
            next_obs_list, reward_list, cost_list, done, step_info = self.env.step(scaled_actions)
            # 7. 存入缓冲区
            self.buffer.add(
                obs=np.stack(obs_list, axis=0),
                actions=actions,
                ...
            )
```

替换为：

```python
            # 3. 策略推断（无梯度）—— 注意 Phase 2 返回 5-tuple
            actions_pi, log_probs, values, cost_values, int_cost_values_pred = self.networks.act(
                obs_list=obs_list,
                neighbor_lists=neighbor_lists,
                deterministic=False,
                device=self.device,
            )

            # 4. 缩放策略动作到 [-a_max, a_max]
            a_pi_scaled = actions_pi * self.config["action_scale"]

            # 5. HOCBF-QP 过滤（Phase 2）
            if self.shield is not None:
                # 获取调度器中已知位置/速度和通信时间（来自 _last_known_pos/_last_comm_time）
                known_pos = self.scheduler._last_known_pos.copy()
                known_vel = self.scheduler._last_known_vel.copy()
                last_comm_time = self.scheduler._last_comm_time.copy()

                a_star_scaled, xis, int_costs_step = self.shield.filter_all_agents(
                    positions=self.env.positions,
                    velocities=self.env.velocities,
                    actions_pi=a_pi_scaled,
                    known_pos=known_pos,
                    known_vel=known_vel,
                    last_comm_time=last_comm_time,
                    neighbor_lists=neighbor_lists,
                )
                exec_actions = a_star_scaled          # 执行 QP 过滤后动作
            else:
                exec_actions = a_pi_scaled            # 消融：直接执行策略动作
                int_costs_step = np.zeros(n)          # 无介入成本

            # 5b. 全局状态（Critic 输入）
            global_state = np.concatenate(obs_list, axis=0)
            if len(global_state) < self.global_state_dim:
                pad = np.zeros(self.global_state_dim - len(global_state))
                global_state = np.concatenate([global_state, pad])

            # 6. 环境步进（执行 a^*，不是 a^π）
            next_obs_list, reward_list, cost_list, done, step_info = self.env.step(exec_actions)

            # 7. 存入缓冲区（存 actions_pi 用于 PPO log prob）
            self.buffer.add(
                obs=np.stack(obs_list, axis=0),
                actions=actions_pi,                   # 存策略原始动作（未过滤）
                log_probs=log_probs,
                rewards=reward_list,
                costs=cost_list,
                value=float(values[0]),
                cost_value=float(cost_values[0]),
                done=done,
                global_state=global_state,
                adj_matrix=adj_matrix,
                int_cost=int_costs_step,              # Phase 2 新增
                int_cost_value=float(int_cost_values_pred[0]),  # Phase 2 新增
            )

            bw_utils.append(sched_result["bandwidth_utilization"])
```

同时更新 bootstrap 部分：

```python
        _, _, last_values, last_cost_values, last_int_cost_values = self.networks.act(
            obs_list=obs_list,
            neighbor_lists=sched_result["neighbor_lists"],
            deterministic=False,
            device=self.device,
        )
        self.buffer.finish_rollout(
            last_value=float(last_values[0]),
            last_cost_value=float(last_cost_values[0]),
            last_int_cost_value=float(last_int_cost_values[0]),  # Phase 2 新增
        )
```

- [ ] **Step 3: 运行全量测试确认无回归**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

预期：现有所有测试通过（collect_rollout 改动不影响测试，因为测试调用 collect_rollout 的 test_algorithm.py 使用了 Phase 1 配置）

- [ ] **Step 4: 提交**

```bash
cd /root/autodl-tmp/safecomm-marl && git add algorithms/safecomm_psched.py
git commit -m "feat: integrate HOCBF-QP dual-action rollout (a_pi and a_star) (Phase 2)"
```

---

## Task 4：λ_int + Safety Gate + 层级 MGDA

**Files:**
- Modify: `algorithms/safecomm_psched.py`

### 背景

Phase 2 update() 新增（规格 §3.1–3.2）：
1. 从 `get_training_data()` 取 `int_cost_advantages`，计算 `int_penalty = λ_int * int_cost_adv.mean()`
2. Safety gate（EMA）：`J_safe_ema ← (1-α) * J_safe_ema + α * J_safe_current`，`safe_gate = J_safe_ema ≤ d_safe + δ_s`
3. 条件式 `λ_int` 更新：safe_gate 激活后才更新
4. 层级 Two-Stage MGDA（见规格 §3.2）

### 步骤

- [ ] **Step 1: 更新 update() 的 Critic 部分**

在 `update()` 中，找到前向计算部分，将返回值从 4-tuple 更新为 6-tuple（含 int_cost_values_b）：

```python
                # 前向（Phase 2：evaluate_actions_batch 返回 6-tuple）
                log_probs_b, entropy_b, values_b, cost_values_b, int_cost_values_b, _ = (
                    self.networks.evaluate_actions_batch(
                        obs_batch=obs_b,
                        actions_batch=actions_b,
                        adj_matrices=adj_b,
                        global_states=global_states_b,
                    )
                )
```

在价值损失计算区域添加 int_cost 相关变量：

```python
                int_cost_adv_b = data["int_cost_advantages"][idx]
                int_cost_returns_b = data["int_cost_returns"][idx]
                int_cost_values_old_b = data["int_cost_values_old"][idx]

                # Intervention cost Lagrangian 惩罚
                int_penalty = self.lambda_int * int_cost_adv_b.mean()

                # 扩展 Critic 损失
                int_cost_vf_loss = value_loss(
                    values_pred=int_cost_values_b,
                    values_old=int_cost_values_old_b.detach(),
                    returns=int_cost_returns_b,
                    clip_eps=clip_eps,
                    use_clip=True,
                )
                total_critic_loss = (
                    self.config["value_coef"] * vf_loss
                    + self.config["cost_value_coef"] * cost_vf_loss
                    + self.config["cost_value_coef"] * int_cost_vf_loss
                )
```

- [ ] **Step 2: 在 update() 中实现层级 MGDA**

将 Actor backward 区域替换为层级 MGDA 实现：

```python
                # ==================== 层级 Two-Stage MGDA（规格 §3.2）====================
                actor_params = (
                    list(self.networks.encoder.parameters())
                    + list(self.networks.aggregator.parameters())
                    + list(self.networks.actor.parameters())
                )

                total_actor_loss = actor_loss + safety_penalty + int_penalty + entropy_loss
                use_mgda = self.config.get("use_mgda", True)

                if use_mgda and lambda_s > self.config.get("lambda_threshold", 0.1):
                    # --- 第一级 MGDA：task+entropy vs safety ---
                    self.actor_optimizer.zero_grad()
                    (actor_loss + entropy_loss).backward(retain_graph=True)
                    g_task = torch.cat([
                        p.grad.clone().flatten() if p.grad is not None
                        else torch.zeros(p.numel(), device=self.device)
                        for p in actor_params
                    ])

                    self.actor_optimizer.zero_grad()
                    safety_penalty.backward(retain_graph=True)
                    g_safe = torch.cat([
                        p.grad.clone().flatten() if p.grad is not None
                        else torch.zeros(p.numel(), device=self.device)
                        for p in actor_params
                    ])

                    w_task, w_safe = mgda_solve(g_task, g_safe)
                    g_ts = w_task * g_task + w_safe * g_safe
                else:
                    # 简单加权（λ_s 过小时跳过 MGDA）
                    self.actor_optimizer.zero_grad()
                    (actor_loss + entropy_loss + safety_penalty).backward(retain_graph=True)
                    g_ts = torch.cat([
                        p.grad.clone().flatten() if p.grad is not None
                        else torch.zeros(p.numel(), device=self.device)
                        for p in actor_params
                    ])

                if (use_mgda and self.safe_gate_active
                        and lambda_int > self.config.get("lambda_int_threshold", 0.05)):
                    # --- 第二级 MGDA：(task+safe) vs intervention ---
                    self.actor_optimizer.zero_grad()
                    int_penalty.backward(retain_graph=True)
                    g_int = torch.cat([
                        p.grad.clone().flatten() if p.grad is not None
                        else torch.zeros(p.numel(), device=self.device)
                        for p in actor_params
                    ])

                    w_ts, w_int = mgda_solve(g_ts, g_int)
                    g_final = w_ts * g_ts + w_int * g_int
                else:
                    # safe_gate 未激活或 λ_int 过小：直接减去 int_penalty 梯度
                    if self.safe_gate_active and lambda_int > 0:
                        self.actor_optimizer.zero_grad()
                        int_penalty.backward(retain_graph=True)
                        g_int = torch.cat([
                            p.grad.clone().flatten() if p.grad is not None
                            else torch.zeros(p.numel(), device=self.device)
                            for p in actor_params
                        ])
                        g_final = g_ts + g_int  # int_penalty 已含负号
                    else:
                        g_final = g_ts

                # 将 g_final 写入各参数的 .grad
                self.actor_optimizer.zero_grad()
                offset = 0
                for p in actor_params:
                    size = p.numel()
                    p.grad = g_final[offset:offset + size].view(p.shape).clone()
                    offset += size

                # 梯度裁剪 + 更新
                torch.nn.utils.clip_grad_norm_(actor_params, self.config["max_grad_norm"])
                self.actor_optimizer.step()
                # ==================== end MGDA ====================
```

- [ ] **Step 3: 在 Lagrangian 乘子更新区域添加 λ_int 和 safety gate**

将原有的 Lagrangian 乘子更新代码块（约 `self.lambda_s = ...` 行）替换为：

```python
        # ==================== Lagrangian 乘子更新（规格 §3.1）====================
        mean_step_cost = float(self.buffer.costs[:self.buffer.ptr].mean())
        estimated_episode_cost = mean_step_cost * self.env.max_steps
        cost_violation = estimated_episode_cost - self.config["safety_budget"]

        # λ_s 更新（无条件）
        self.lambda_s = float(np.clip(
            self.lambda_s + self.config["lr_lambda"] * cost_violation,
            0.0,
            self.lambda_max,
        ))

        # Safety gate EMA 更新（规格 §3.1）
        j_safe_current = estimated_episode_cost
        if self._j_safe_ema is None:
            self._j_safe_ema = j_safe_current
        else:
            alpha = self._ema_alpha
            self._j_safe_ema = (1.0 - alpha) * self._j_safe_ema + alpha * j_safe_current

        d_safe = self.config["safety_budget"]
        delta_s = self.config.get("safety_gate_tolerance", 0.02)
        self.safe_gate_active = bool(self._j_safe_ema <= d_safe + delta_s)

        # λ_int 更新（条件式，safe_gate 激活后才更新）
        mean_step_int_cost = float(self.buffer.int_costs[:self.buffer.ptr].mean())
        estimated_int_cost = mean_step_int_cost * self.env.max_steps
        int_cost_violation = estimated_int_cost - self.config.get("safety_budget_int", 0.5)

        if self.safe_gate_active:
            self.lambda_int = float(np.clip(
                self.lambda_int + self.config.get("lr_lambda_int", 0.005) * int_cost_violation,
                0.0,
                self.lambda_int_max,
            ))
        # 否则 λ_int 冻结（或极慢衰减，此处选择冻结）
        # ==================== end Lagrangian ====================
```

- [ ] **Step 4: 扩展 update_info 字典以包含 Phase 2 指标**

在 `update_info` 字典中添加：

```python
        update_info = {
            ...（原有字段）...
            # Phase 2 新增
            "lambda_int": self.lambda_int,
            "safe_gate_active": int(self.safe_gate_active),
            "j_safe_ema": float(self._j_safe_ema) if self._j_safe_ema else 0.0,
            "J_int": estimated_int_cost,
            "mean_xi": float(self.buffer.int_costs[:self.buffer.ptr].mean()),
        }
```

- [ ] **Step 5: 扩展 save/load 以保存 Phase 2 状态**

在 `save()` 的 state 字典中添加：

```python
        state = {
            ...（原有字段）...
            # Phase 2 新增
            "lambda_int": self.lambda_int,
            "j_safe_ema": self._j_safe_ema,
            "safe_gate_active": self.safe_gate_active,
        }
```

在 `load()` 中添加：

```python
        self.lambda_int = state.get("lambda_int", 0.0)
        self._j_safe_ema = state.get("j_safe_ema", None)
        self.safe_gate_active = state.get("safe_gate_active", False)
```

- [ ] **Step 6: 扩展 train() 日志打印**

```python
                print(
                    f"[iter {self.n_iterations:4d} | steps {self.total_steps:7d}] "
                    f"FE={rollout_info['mean_formation_error']:.4f}  "
                    f"cost={rollout_info['mean_episode_cost']:.4f}  "
                    f"λ_s={update_info['lambda_s']:.4f}  "
                    f"λ_int={update_info['lambda_int']:.4f}  "
                    f"gate={'ON' if update_info['safe_gate_active'] else 'off'}  "
                    f"J_int={update_info.get('J_int', 0):.4f}  "
                    f"BU={rollout_info['mean_bandwidth_util']:.3f}  "
                    f"KL={update_info['approx_kl']:.5f}  "
                    f"FPS={fps:.0f}"
                )
```

- [ ] **Step 7: 运行全量测试**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

预期：所有测试通过

- [ ] **Step 8: 提交**

```bash
cd /root/autodl-tmp/safecomm-marl && git add algorithms/safecomm_psched.py
git commit -m "feat: add lambda_int, safety gate EMA, hierarchical MGDA (Phase 2)"
```

---

## Task 5：Baselines（配置包装器类）

**Files:**
- Create: `baselines/__init__.py`
- Create: `baselines/mappo.py`
- Create: `baselines/mappo_lag.py`
- Create: `baselines/dgn_mappo.py`
- Create: `baselines/hocbf_ppo.py`
- Create: `baselines/random_topk.py`

### 背景

baseline 必须服务论文对照，而不是复刻外部仓库。MAPPO/MAPPO-Lag 参考 `external/on-policy` 与 MACPO/MAPPO-Lagrangian 的训练语义，DGN+MAPPO 只保留“通信受限图聚合”对照含义，所有类均复用本项目 `SafeCommVoI`、scheduler、buffer 和网络接口。

DGN 源仓库无明确许可证，因此 `baselines/dgn_mappo.py` 必须自写 PyTorch 版 masked graph aggregation，不能复制 `external/DGN` 的 TensorFlow 代码。RandomTopK 是 SafeComm 调度器消融，不是外部算法复现。

5 个 baseline 均基于 SafeCommVoI 基础设施，通过配置覆盖实现：
- 全通信：`schedule_mode="full"`，`k=9999`
- 无安全：`safety_budget=1e9`，`use_hocbf_qp=False`
- 无 λ_int：`lambda_int_init=0.0`，`safety_budget_int=1e9`

### 步骤

- [ ] **Step 1: 创建 baselines/__init__.py**

注意：此时 `macpo.py` 尚未创建（Task 6 创建），暂不导入。Task 6 Step 1 会更新此文件添加 MACPO。

```python
"""SafeComm-VoI Baselines for Phase 2 comparison."""
from baselines.mappo import MAPPOAgent
from baselines.mappo_lag import MAPPOLagAgent
from baselines.dgn_mappo import DGNMAPPOAgent
from baselines.hocbf_ppo import HOCBFPPOAgent
from baselines.random_topk import SafeCommRandomTopK

__all__ = [
    "MAPPOAgent", "MAPPOLagAgent", "DGNMAPPOAgent",
    "HOCBFPPOAgent", "SafeCommRandomTopK",
]
```

- [ ] **Step 2: 创建 baselines/mappo.py**

```python
"""MAPPO baseline: 全通信，无安全约束（任务性能上界）。"""
from algorithms.safecomm_psched import SafeCommVoI
from envs.uav_formation_env import UAVFormationEnv
from typing import Dict, Optional


class MAPPOAgent(SafeCommVoI):
    """
    MAPPO baseline（规格 §5.1）。
    配置：全通信（k=∞）+ 无安全约束 + 无 QP Shield。
    """

    def __init__(self, env: UAVFormationEnv, config: Optional[Dict] = None, device: str = "auto"):
        baseline_config = {
            # 全通信
            "schedule_mode": "full",
            "k": 9999,
            # 无安全
            "safety_budget": 1e9,
            "lambda_init": 0.0,
            "use_hocbf_qp": False,
            # 无 λ_int
            "lambda_int_init": 0.0,
            "safety_budget_int": 1e9,
            "use_mgda": False,
        }
        if config:
            baseline_config.update(config)
        super().__init__(env, baseline_config, device)
```

- [ ] **Step 3: 创建 baselines/mappo_lag.py**

```python
"""MAPPO-Lag baseline: 全通信 + λ_s Lagrangian，无通信约束。"""
from algorithms.safecomm_psched import SafeCommVoI
from envs.uav_formation_env import UAVFormationEnv
from typing import Dict, Optional


class MAPPOLagAgent(SafeCommVoI):
    """
    MAPPO-Lag baseline（规格 §5.1）。
    配置：全通信 + 单约束 Lagrangian（λ_s）+ 无 QP。
    """

    def __init__(self, env: UAVFormationEnv, config: Optional[Dict] = None, device: str = "auto"):
        baseline_config = {
            "schedule_mode": "full",
            "k": 9999,
            "use_hocbf_qp": False,
            "lambda_int_init": 0.0,
            "safety_budget_int": 1e9,
            "use_mgda": False,
        }
        if config:
            baseline_config.update(config)
        super().__init__(env, baseline_config, device)
```

- [ ] **Step 4: 创建 baselines/dgn_mappo.py**

```python
"""DGN+MAPPO baseline: random TopK 通信约束，无安全（通信受限 MARL 对照）。"""
from algorithms.safecomm_psched import SafeCommVoI
from envs.uav_formation_env import UAVFormationEnv
from typing import Dict, Optional


class DGNMAPPOAgent(SafeCommVoI):
    """
    DGN+MAPPO baseline（规格 §5.1）。
    配置：random TopK（k 受限）+ 无安全 + 无 QP。
    """

    def __init__(self, env: UAVFormationEnv, config: Optional[Dict] = None, device: str = "auto"):
        baseline_config = {
            "schedule_mode": "random",
            "safety_budget": 1e9,
            "lambda_init": 0.0,
            "use_hocbf_qp": False,
            "lambda_int_init": 0.0,
            "safety_budget_int": 1e9,
            "use_mgda": False,
            "beta": 0.0,   # 不使用 VoI 优先级
        }
        if config:
            baseline_config.update(config)
        super().__init__(env, baseline_config, device)
```

- [ ] **Step 5: 创建 baselines/hocbf_ppo.py**

```python
"""HOCBF-PPO baseline: 全通信 + QP Shield + λ_s（硬安全过滤 baseline）。"""
from algorithms.safecomm_psched import SafeCommVoI
from envs.uav_formation_env import UAVFormationEnv
from typing import Dict, Optional


class HOCBFPPOAgent(SafeCommVoI):
    """
    HOCBF-PPO baseline（规格 §5.1，自实现）。
    配置：全通信 + HOCBF-QP 过滤 + λ_s + 无 VoI 调度。
    """

    def __init__(self, env: UAVFormationEnv, config: Optional[Dict] = None, device: str = "auto"):
        baseline_config = {
            "schedule_mode": "full",
            "k": 9999,
            "use_hocbf_qp": True,
            "lambda_int_init": 0.0,
            "safety_budget_int": 1e9,
            "use_mgda": False,
            "beta": 0.0,
        }
        if config:
            baseline_config.update(config)
        super().__init__(env, baseline_config, device)
```

- [ ] **Step 6: 创建 baselines/random_topk.py**

```python
"""SafeComm-RandomTopK: 主方法 VoI → Random TopK（调度器消融 baseline）。"""
from algorithms.safecomm_psched import SafeCommVoI
from envs.uav_formation_env import UAVFormationEnv
from typing import Dict, Optional


class SafeCommRandomTopK(SafeCommVoI):
    """
    SafeComm-RandomTopK baseline（规格 §5.1，调度器消融）。
    配置：Random TopK 调度 + 完整三层安全（HOCBF-QP + λ_s + λ_int + MGDA）。
    """

    def __init__(self, env: UAVFormationEnv, config: Optional[Dict] = None, device: str = "auto"):
        baseline_config = {
            "schedule_mode": "random",
            "beta": 0.0,   # random 模式忽略 VoI 优先级
        }
        if config:
            baseline_config.update(config)
        super().__init__(env, baseline_config, device)
```

- [ ] **Step 7: 提交**

```bash
cd /root/autodl-tmp/safecomm-marl && git add baselines/
git commit -m "feat: add SafeCommVoI-based baseline wrappers (MAPPO, MAPPO-Lag, DGN+MAPPO, HOCBF-PPO, RandomTopK)"
```

---

## Task 6：MACPO Baseline

**Files:**
- Create: `baselines/macpo.py`
- Test: `tests/test_baselines.py`

### 背景

MACPO/MAPPO-Lagrangian 参考 Gu et al. 的多智能体约束策略优化路线，以及 `external/Multi-Agent-Constrained-Policy-Optimisation` 中 cost critic、约束优势和投影更新的组织方式。该外部仓库许可证文件含冲突标记，且实现与其 runner/env 强绑定，本项目只实现可测的简化 baseline，不复制源码。

MACPO baseline 使用安全梯度投影而非 Lagrangian。核心差异：
- 若 J_safe > d：将任务梯度投影到安全梯度的正交补空间
- 若 J_safe ≤ d：使用标准任务梯度

### 步骤

- [ ] **Step 1: 创建 baselines/macpo.py**

```python
"""
MACPO Baseline（Gu et al. MACPO 路线的简化版）。

核心机制：安全梯度投影（safe gradient projection），
比 MAPPO-Lag 更直接处理约束边界。

实现：MAPPO（全通信，无 QP）+ 安全梯度投影（替代 Lagrangian）。
"""
import numpy as np
import torch
from typing import Dict, List, Optional

from algorithms.safecomm_psched import SafeCommVoI
from algorithms.ppo_utils import mgda_solve
from envs.uav_formation_env import UAVFormationEnv


class MACPOAgent(SafeCommVoI):
    """
    MACPO baseline（规格 §5.1，CMDP 二阶优化近似）。

    配置：全通信 + 无 QP + 安全梯度投影（替代 Lagrangian 乘子更新）。
    """

    def __init__(self, env: UAVFormationEnv, config: Optional[Dict] = None, device: str = "auto"):
        baseline_config = {
            "schedule_mode": "full",
            "k": 9999,
            "use_hocbf_qp": False,
            "lambda_int_init": 0.0,
            "safety_budget_int": 1e9,
            "use_mgda": False,
        }
        if config:
            baseline_config.update(config)
        super().__init__(env, baseline_config, device)

    def update(self) -> Dict:
        """
        MACPO 更新：PPO + 安全梯度投影（CMDP 近似）。

        若 J_safe > d_safe：
            g_task_proj = g_task - (g_task · g_safe / ||g_safe||^2) * g_safe
            （将任务梯度投影到安全梯度正交补空间）
        否则：
            g_task_proj = g_task  （直接使用任务梯度）
        """
        import torch.nn.functional as F

        self.networks.train()
        data = self.buffer.get_training_data(
            gamma=self.config["gamma"],
            lam=self.config["lam"],
            normalize_adv=True,
            device=self.device,
        )

        T = len(data["obs"])
        batch_size = min(self.config["batch_size"], T)
        n_epochs = self.config["n_epochs"]
        clip_eps = self.config["clip_eps"]

        all_actor_losses, all_critic_losses, all_entropy, all_approx_kl = [], [], [], []

        actor_params = (
            list(self.networks.encoder.parameters())
            + list(self.networks.aggregator.parameters())
            + list(self.networks.actor.parameters())
        )

        # 估计 J_safe（用于判断约束是否激活）
        mean_step_cost = float(self.buffer.costs[:self.buffer.ptr].mean())
        estimated_episode_cost = mean_step_cost * self.env.max_steps
        constraint_active = estimated_episode_cost > self.config["safety_budget"]

        from algorithms.ppo_utils import (
            ppo_clip_loss, value_loss, normalize_advantages, explained_variance,
        )

        for epoch in range(n_epochs):
            indices = torch.randperm(T)
            for start in range(0, T, batch_size):
                idx = indices[start:start + batch_size]
                if len(idx) < 2:
                    continue

                obs_b = data["obs"][idx]
                actions_b = data["actions"][idx]
                log_probs_old_b = data["log_probs_old"][idx]
                advantages_b = data["advantages"][idx]
                returns_b = data["returns"][idx]
                cost_adv_b = data["cost_advantages"][idx]
                cost_returns_b = data["cost_returns"][idx]
                values_old_b = data["values_old"][idx]
                cost_values_old_b = data["cost_values_old"][idx]
                global_states_b = data["global_states"][idx]
                adj_b = data["adj_matrices"][idx]

                log_probs_b, entropy_b, values_b, cost_values_b, _, _ = (
                    self.networks.evaluate_actions_batch(
                        obs_batch=obs_b, actions_batch=actions_b,
                        adj_matrices=adj_b, global_states=global_states_b,
                    )
                )

                log_probs_mean = log_probs_b.mean(dim=1)
                log_probs_old_mean = log_probs_old_b.mean(dim=1)

                actor_loss, approx_kl = ppo_clip_loss(
                    log_probs_mean, log_probs_old_mean.detach(), advantages_b, clip_eps,
                )
                entropy_mean = entropy_b.mean()
                entropy_loss = -self.config["entropy_coef"] * entropy_mean

                loss_task = actor_loss + entropy_loss

                # 安全梯度投影（MACPO 核心）
                if constraint_active:
                    # 计算任务梯度
                    self.actor_optimizer.zero_grad()
                    loss_task.backward(retain_graph=True)
                    g_task = torch.cat([
                        p.grad.clone().flatten() if p.grad is not None
                        else torch.zeros(p.numel(), device=self.device)
                        for p in actor_params
                    ])

                    # 计算安全梯度
                    loss_safe = cost_adv_b.mean()
                    self.actor_optimizer.zero_grad()
                    loss_safe.backward(retain_graph=True)
                    g_safe = torch.cat([
                        p.grad.clone().flatten() if p.grad is not None
                        else torch.zeros(p.numel(), device=self.device)
                        for p in actor_params
                    ])

                    # 安全梯度投影：g_proj = g_task - (g_task · g_safe / ||g_safe||^2) * g_safe
                    safe_norm_sq = float((g_safe * g_safe).sum())
                    if safe_norm_sq > 1e-10:
                        dot = float((g_task * g_safe).sum())
                        g_proj = g_task - (dot / safe_norm_sq) * g_safe
                    else:
                        g_proj = g_task

                    # 应用投影梯度
                    self.actor_optimizer.zero_grad()
                    offset = 0
                    for p in actor_params:
                        size = p.numel()
                        p.grad = g_proj[offset:offset + size].view(p.shape).clone()
                        offset += size
                else:
                    # 约束满足：直接使用任务梯度
                    self.actor_optimizer.zero_grad()
                    loss_task.backward()

                torch.nn.utils.clip_grad_norm_(actor_params, self.config["max_grad_norm"])
                self.actor_optimizer.step()

                # Critic 更新
                vf_loss = value_loss(values_b, values_old_b.detach(), returns_b, clip_eps)
                cost_vf_loss = value_loss(cost_values_b, cost_values_old_b.detach(), cost_returns_b, clip_eps)
                total_critic_loss = (
                    self.config["value_coef"] * vf_loss
                    + self.config["cost_value_coef"] * cost_vf_loss
                )
                self.critic_optimizer.zero_grad()
                total_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.networks.critic.parameters())
                    + list(self.networks.cost_critic.parameters())
                    + list(self.networks.int_cost_critic.parameters()),
                    self.config["max_grad_norm"],
                )
                self.critic_optimizer.step()

                all_actor_losses.append(float(loss_task.detach()))
                all_critic_losses.append(float(total_critic_loss.detach()))
                all_entropy.append(float(entropy_mean.detach()))
                all_approx_kl.append(float(approx_kl.detach()))

        # λ_s 更新（MACPO 中用于监控，但不影响梯度）
        cost_violation = estimated_episode_cost - self.config["safety_budget"]
        self.lambda_s = float(np.clip(
            self.lambda_s + self.config["lr_lambda"] * cost_violation, 0.0, self.lambda_max,
        ))

        return {
            "actor_loss": np.mean(all_actor_losses),
            "critic_loss": np.mean(all_critic_losses),
            "entropy": np.mean(all_entropy),
            "approx_kl": np.mean(all_approx_kl),
            "lambda_s": self.lambda_s,
            "lambda_int": 0.0,
            "safe_gate_active": 0,
            "J_safe": estimated_episode_cost,
            "J_int": 0.0,
            "cost_violation": cost_violation,
            "constraint_active": int(constraint_active),
            "ev_task": explained_variance(
                self.buffer.values[:self.buffer.ptr],
                self.buffer.rewards[:self.buffer.ptr].mean(axis=1),
            ),
        }
```

- [ ] **Step 2: 更新 baselines/__init__.py，加入 MACPOAgent**

将 `baselines/__init__.py` 完整替换为（Task 5 版本 + MACPO）：

```python
"""SafeComm-VoI Baselines for Phase 2 comparison."""
from baselines.mappo import MAPPOAgent
from baselines.mappo_lag import MAPPOLagAgent
from baselines.dgn_mappo import DGNMAPPOAgent
from baselines.hocbf_ppo import HOCBFPPOAgent
from baselines.random_topk import SafeCommRandomTopK
from baselines.macpo import MACPOAgent

__all__ = [
    "MAPPOAgent", "MAPPOLagAgent", "DGNMAPPOAgent",
    "HOCBFPPOAgent", "SafeCommRandomTopK", "MACPOAgent",
]
```

- [ ] **Step 4: 写 baselines 冒烟测试**

新建 `tests/test_baselines.py`：

```python
"""
Baselines 冒烟测试：验证每个 baseline 可完成 1 次 collect_rollout + update。
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from envs.uav_formation_env import UAVFormationEnv
from baselines import MAPPOAgent, MAPPOLagAgent, DGNMAPPOAgent, HOCBFPPOAgent
from baselines import SafeCommRandomTopK, MACPOAgent

FAST_CONFIG = {
    "k": 2, "n_steps": 20, "n_epochs": 1, "batch_size": 10,
    "msg_dim": 16, "hidden_dims": [32, 32], "H": 2,
    "safety_budget": 0.5, "beta": 0.0, "tau_ref": 1.0, "tau_max": 5.0,
    "use_hocbf_qp": False,  # 冒烟测试禁用 QP 以加速
}


def make_env():
    return UAVFormationEnv(n_agents=3, d_min=0.3, dt=0.1, max_steps=10, R_comm=5.0)


@pytest.mark.parametrize("AgentCls", [
    MAPPOAgent, MAPPOLagAgent, DGNMAPPOAgent,
    HOCBFPPOAgent, SafeCommRandomTopK, MACPOAgent,
])
def test_baseline_one_iteration(AgentCls):
    """每个 baseline 能完成一次 collect_rollout + update，不抛出异常。"""
    env = make_env()
    agent = AgentCls(env, config=FAST_CONFIG, device="cpu")
    rollout_info = agent.collect_rollout()
    update_info = agent.update()
    assert "actor_loss" in update_info
    assert "lambda_s" in update_info
    assert rollout_info["n_episodes"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

- [ ] **Step 5: 运行 baseline 测试**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_baselines.py -v
```

预期：6 个 parametrized 测试全部 `PASSED`

- [ ] **Step 6: 提交**

```bash
cd /root/autodl-tmp/safecomm-marl && git add baselines/__init__.py baselines/macpo.py tests/test_baselines.py
git commit -m "feat: add MACPO baseline with safety gradient projection (Phase 2)"
```

---

## Task 7：evaluate.py + 消融配置 + 集成冒烟测试

**Files:**
- Create: `evaluate.py`
- Create: `configs/ablation/` 目录下 6 个 yaml
- Test: 集成冒烟测试（验证 Phase 2 成功标准）

### 步骤

- [ ] **Step 1: 创建 evaluate.py**

```python
"""
SafeComm-VoI Phase 2 评估脚本

用法：
  python evaluate.py --checkpoint checkpoints/final.pt --episodes 20
  python evaluate.py --checkpoint checkpoints/final.pt --no_qp    # Shield Independence Eval
  python evaluate.py --checkpoint checkpoints/final.pt --n_agents 8  # 扩展测试
"""
import argparse
import os
import numpy as np
import yaml

from envs.uav_formation_env import UAVFormationEnv
from algorithms.safecomm_psched import SafeCommVoI


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_evaluation(
    agent: SafeCommVoI,
    n_episodes: int = 20,
    deterministic: bool = True,
) -> dict:
    """
    运行 n_episodes 次评估，返回规格 §5.2 定义的全部指标。

    指标：FE, Success, Collision, CVR, MPS, InterventionRate
    """
    agent.networks.eval()
    env = agent.env

    all_fe, all_success, all_collision, all_cvr_flag = [], [], [], []
    all_mps, all_intervention_rate = [], []

    import torch
    with torch.no_grad():
        for ep in range(n_episodes):
            obs_list, _ = env.reset()
            agent.scheduler.reset_episode(env.positions)
            ep_fe, ep_collision, ep_mps, ep_interv = [], [], [], []
            ep_cost_sum = 0.0
            done = False

            while not done:
                phys_edges = env.get_physical_graph()
                sched_result = agent.scheduler.schedule_step(env.positions, phys_edges)
                agent.scheduler.update_history(sched_result["active_edges"], env.positions)

                actions_pi, _, _, _, _ = agent.networks.act(
                    obs_list=obs_list,
                    neighbor_lists=sched_result["neighbor_lists"],
                    deterministic=deterministic,
                    device=agent.device,
                )
                a_pi_scaled = actions_pi * agent.config["action_scale"]

                # 可选：Shield Independence Eval（关闭 QP，验证策略内化安全行为）
                if agent.shield is not None:
                    known_pos = agent.scheduler._last_known_pos.copy()
                    known_vel = agent.scheduler._last_known_vel.copy()
                    last_comm_time = agent.scheduler._last_comm_time.copy()
                    a_star, xis, interventions = agent.shield.filter_all_agents(
                        env.positions, env.velocities, a_pi_scaled,
                        known_pos, known_vel, last_comm_time,
                        sched_result["neighbor_lists"],
                    )
                    exec_actions = a_star
                    ep_interv.append(float(np.mean(interventions)))
                else:
                    exec_actions = a_pi_scaled
                    ep_interv.append(0.0)

                obs_list, rewards, costs, done, info = env.step(exec_actions)
                ep_fe.append(info.get("formation_error", 0.0))
                ep_collision.append(float(np.sum(costs) > 0))
                ep_cost_sum += float(np.mean(costs))
                ep_mps.append(float(sched_result["n_active"]))

            all_fe.append(float(np.mean(ep_fe)))
            all_success.append(float(info.get("success", False)))
            all_collision.append(float(np.sum(ep_collision)))
            all_cvr_flag.append(float(ep_cost_sum > agent.config["safety_budget"]))
            all_mps.append(float(np.mean(ep_mps)))
            all_intervention_rate.append(float(np.mean(ep_interv)))

    results = {
        "FE":                float(np.mean(all_fe)),
        "FE_std":            float(np.std(all_fe)),
        "Success":           float(np.mean(all_success)),
        "Collision":         float(np.mean(all_collision)),
        "CVR":               float(np.mean(all_cvr_flag)),
        "MPS":               float(np.mean(all_mps)),
        "InterventionRate":  float(np.mean(all_intervention_rate)),
        "lambda_s":          agent.lambda_s,
        "lambda_int":        agent.lambda_int,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="SafeComm-VoI Phase 2 评估")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--n_agents", type=int, default=None)
    parser.add_argument("--no_qp", action="store_true",
                        help="关闭 QP Shield（Shield Independence Eval）")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    env_cfg = cfg.get("env", {})
    if args.n_agents:
        env_cfg["n_agents"] = args.n_agents

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
        K_obs_neighbors=env_cfg.get("K_obs_neighbors", 4),
        seed=env_cfg.get("seed", 42),
    )

    agent = SafeCommVoI(env=env, device=args.device)
    agent.load(args.checkpoint)

    if args.no_qp:
        print("[Shield Independence Eval] QP Shield 已关闭")
        agent.shield = None

    results = run_evaluation(agent, n_episodes=args.episodes, deterministic=args.deterministic)

    mode = "no_qp" if args.no_qp else "with_qp"
    print(f"\n=== 评估结果（{mode}, N={env.n}, episodes={args.episodes}）===")
    for k, v in results.items():
        print(f"  {k:20s}: {v:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 创建消融配置文件**

创建 `configs/ablation/` 目录及 6 个 YAML：

**`configs/ablation/beta_zero.yaml`**（w/o AoI）：
```yaml
# 消融：w/o AoI（β=0，纯安全优先调度）
comm:
  beta: 0.0
```

**`configs/ablation/no_cbf_buffer.yaml`**（w/o 保守缓冲）：
```yaml
# 消融：w/o C+ buffer（v_max_cbf=0，等价于 d_buf=0）
comm:
  v_max_cbf: 0.0
```

**`configs/ablation/no_hocbf_qp.yaml`**（w/o HOCBF-QP）：
```yaml
# 消融：w/o HOCBF-QP 执行过滤
safety:
  use_hocbf_qp: false
```

**`configs/ablation/no_lambda_int.yaml`**（w/o λ_int）：
```yaml
# 消融：w/o λ_int（intervention cost 约束）
safety:
  lambda_int_init: 0.0
  safety_budget_int: 1.0e9   # 上限无穷大，λ_int 不会上升
```

**`configs/ablation/no_mgda.yaml`**（w/o MGDA）：
```yaml
# 消融：w/o 层级 MGDA（使用普通加权梯度）
safety:
  use_mgda: false
```

**`configs/ablation/random_topk.yaml`**（SafeComm-RandomTopK）：
```yaml
# 消融：SafeComm-RandomTopK（VoI → Random TopK）
comm:
  schedule_mode: "random"
  beta: 0.0
```

```bash
cd /root/autodl-tmp/safecomm-marl && mkdir -p configs/ablation
# 手工创建上述 6 个文件
```

- [ ] **Step 3: 验证消融配置可被 train.py 正确加载**

更新 `train.py` 以支持消融配置叠加（在 `flatten_config` 函数中添加对 `safety` 字段的支持）：

在 `flatten_config()` 函数末尾的 return 字典中添加：

```python
        # Phase 2 新增
        "use_hocbf_qp": safety_cfg.get("use_hocbf_qp", True),
        "use_mgda": safety_cfg.get("use_mgda", True),
        "lambda_int_init": safety_cfg.get("lambda_int_init", 0.0),
        "safety_budget_int": safety_cfg.get("safety_budget_int", 0.5),
```

- [ ] **Step 4: Phase 2 集成冒烟测试**

```bash
cd /root/autodl-tmp/safecomm-marl && python -c "
import numpy as np
from envs.uav_formation_env import UAVFormationEnv
from algorithms.safecomm_psched import SafeCommVoI

env = UAVFormationEnv(n_agents=4, d_min=0.3, dt=0.1, max_steps=10, R_comm=5.0)
cfg = {
    'k': 4, 'n_steps': 40, 'n_epochs': 2, 'batch_size': 20,
    'msg_dim': 32, 'hidden_dims': [64, 64], 'H': 2,
    'safety_budget': 0.5, 'beta': 1.0, 'tau_ref': 1.0, 'tau_max': 5.0,
    'use_hocbf_qp': True, 'qp_rho': 100.0, 'alpha1': 1.0, 'alpha2': 1.0,
    'lambda_int_init': 0.0, 'safety_budget_int': 0.5, 'use_mgda': True,
    'safety_gate_ema_window': 3, 'safety_gate_tolerance': 0.1,
}
agent = SafeCommVoI(env, config=cfg, device='cpu')

# 验证 1：一次完整 rollout + update 不报错
rollout_info = agent.collect_rollout()
update_info = agent.update()
print('OK: collect_rollout + update 正常完成')

# 验证 2：update_info 含 Phase 2 关键字段
assert 'lambda_int' in update_info, 'lambda_int 缺失'
assert 'safe_gate_active' in update_info, 'safe_gate_active 缺失'
assert 'J_int' in update_info, 'J_int 缺失'
print(f'OK: Phase 2 关键指标: lambda_int={update_info[\"lambda_int\"]:.4f}, safe_gate={update_info[\"safe_gate_active\"]}')

# 验证 3：|E_t| <= k 依然满足
obs_list, _ = env.reset()
agent.scheduler.reset_episode(env.positions)
for _ in range(20):
    phys = env.get_physical_graph()
    result = agent.scheduler.schedule_step(env.positions, phys)
    assert len(result['active_edges']) <= 4, f'|E_t|={len(result[\"active_edges\"])} > k=4'
print('OK: |E_t| <= k 依然满足')

# 验证 4：save/load 正常
import os, tempfile
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    tmppath = f.name
agent.save(tmppath)
agent2 = SafeCommVoI(env, config=cfg, device='cpu')
agent2.load(tmppath)
assert abs(agent2.lambda_int - agent.lambda_int) < 1e-6, 'lambda_int save/load 不一致'
os.unlink(tmppath)
print('OK: save/load 正常（含 lambda_int, j_safe_ema）')
print()
print('=== Phase 2 集成冒烟测试全部通过 ===')
"
```

预期：所有 4 条 OK 打印，最后打印"Phase 2 集成冒烟测试全部通过"

- [ ] **Step 5: 运行全量测试套件**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/ -v 2>&1 | tail -30
```

预期：所有测试通过（含 test_hocbf_qp.py、test_baselines.py、原有测试）

- [ ] **Step 6: 最终提交**

```bash
cd /root/autodl-tmp/safecomm-marl && git add evaluate.py configs/ablation/ train.py
git commit -m "feat: add evaluate.py, ablation configs, complete Phase 2 (full SafeComm-VoI)"
```

---

## Phase 2 成功标准核对

| 标准 | 验证方式 |
|------|---------|
| SafeComm-VoI FE/Collision/CVR 优于全部 baseline | evaluate.py 运行 20 episodes |
| RandomTopK 在 FE 上差于 VoI | evaluate.py 对比 |
| 消融显示 intervention rate 随训练降低 | log 中 `J_int` 和 `mean_xi` 下降趋势 |
| λ_int 在 safe_gate 激活后上升 | log 中 `gate=ON` + `lambda_int` 增加 |
| MGDA 两次 backward 仅在 λ > threshold 时激活 | `use_mgda=True` + `lambda_threshold=0.1` |
| Shield Independence Eval：`--no_qp` 下 FE 轻微上升 | evaluate.py --no_qp |
| N=4–8 扩展：n_agents 参数可配置 | evaluate.py --n_agents 8 |
