"""
仿真环境单元测试

覆盖：
  1. 观测空间维度正确性
  2. 动作执行后状态更新正确性（双积分器动力学）
  3. 安全成本计算（碰撞检测）
  4. 编队误差计算
  5. 成功判定
  6. 边界反弹
  7. 物理图与 CBF 计算
  8. 不同编队类型
  9. 不同 n_agents 支持
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from envs.uav_formation_env import UAVFormationEnv


class TestEnvInit:
    """环境初始化测试"""

    def test_default_init(self):
        """默认参数初始化不报错"""
        env = UAVFormationEnv(n_agents=4, seed=0)
        obs_list, info = env.reset()
        assert len(obs_list) == 4
        assert isinstance(info, dict)

    def test_obs_dim(self):
        """观测维度应等于 6 + K_obs * 6"""
        for k_obs in [1, 2, 3]:
            env = UAVFormationEnv(n_agents=4, K_obs_neighbors=k_obs)
            obs_list, _ = env.reset()
            expected_dim = 6 + k_obs * 6
            for obs in obs_list:
                assert obs.shape == (expected_dim,), (
                    f"K_obs={k_obs}: 观测维度应为 {expected_dim}，实际为 {obs.shape}"
                )

    def test_n_agents_variants(self):
        """支持 N=4,6,8"""
        for n in [4, 6, 8]:
            env = UAVFormationEnv(n_agents=n, seed=0)
            obs_list, info = env.reset()
            assert len(obs_list) == n

    def test_init_no_collision(self):
        """随机初始化应保证无初始碰撞（多次重置）"""
        env = UAVFormationEnv(n_agents=6, d_min=0.5, seed=123)
        for _ in range(20):
            obs_list, _ = env.reset()
            cost_list = env._compute_costs()
            assert all(c == 0.0 for c in cost_list), "初始化不应产生碰撞"


class TestDynamics:
    """双积分器动力学测试"""

    def test_velocity_update(self):
        """速度应随加速度增量更新"""
        env = UAVFormationEnv(n_agents=2, dt=0.1, a_max=2.0, v_max=10.0, seed=0)
        env.reset()
        # 将速度清零
        env.velocities = np.zeros((2, 3), dtype=np.float32)
        env.positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)

        actions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        env.step(actions)

        # UAV 0 的速度应约为 0 + 1.0 * 0.1 = 0.1
        assert abs(env.velocities[0, 0] - 0.1) < 1e-5, (
            f"速度更新错误: {env.velocities[0, 0]}"
        )

    def test_position_update(self):
        """位置应随速度更新"""
        env = UAVFormationEnv(n_agents=2, dt=0.1, a_max=5.0, v_max=10.0, seed=0)
        env.reset()
        env.velocities = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        env.positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)
        init_v = env.velocities[0, 0]  # 1.0

        actions = np.zeros((2, 3), dtype=np.float32)
        env.step(actions)

        # 位置应为 0 + 1.0 * 0.1 = 0.1（近似，因为速度可能有微小变化）
        assert abs(env.positions[0, 0] - 0.1) < 1e-4, (
            f"位置更新错误: {env.positions[0, 0]}"
        )

    def test_velocity_clip(self):
        """速度应被裁剪到 v_max"""
        env = UAVFormationEnv(n_agents=2, v_max=1.0, a_max=10.0, dt=1.0, seed=0)
        env.reset()
        env.velocities = np.zeros((2, 3), dtype=np.float32)
        env.positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)

        # 施加超大加速度
        actions = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        env.step(actions)

        # 速度不应超过 v_max
        assert abs(env.velocities[0, 0]) <= 1.0 + 1e-6

    def test_action_clip(self):
        """动作应被裁剪到 [-a_max, a_max]"""
        env = UAVFormationEnv(n_agents=2, a_max=2.0, v_max=10.0, dt=0.1, seed=0)
        env.reset()
        env.velocities = np.zeros((2, 3), dtype=np.float32)
        env.positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)

        actions = np.array([[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        env.step(actions)

        # 速度 = clip(100, -2, 2) * 0.1 = 0.2
        assert abs(env.velocities[0, 0] - 0.2) < 1e-5


class TestSafetyCost:
    """安全成本（碰撞）计算测试"""

    def test_collision_detected(self):
        """两 UAV 距离小于 d_min 时应触发碰撞成本"""
        env = UAVFormationEnv(n_agents=2, d_min=1.0, seed=0)
        env.reset()
        env.positions = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float32)
        costs = env._compute_costs()
        assert costs[0] == 1.0 and costs[1] == 1.0, "两者都应有碰撞成本"

    def test_no_collision_at_safe_dist(self):
        """距离 >= d_min 时无碰撞"""
        env = UAVFormationEnv(n_agents=2, d_min=0.5, seed=0)
        env.reset()
        env.positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        costs = env._compute_costs()
        assert costs[0] == 0.0 and costs[1] == 0.0

    def test_cost_range(self):
        """成本值应为 0 或 1"""
        env = UAVFormationEnv(n_agents=4, seed=42)
        env.reset()
        actions = np.random.uniform(-1, 1, (4, 3))
        _, _, cost_list, _, _ = env.step(actions)
        for c in cost_list:
            assert c in [0.0, 1.0], f"成本值应为 0 或 1，实际为 {c}"


class TestFormationError:
    """编队误差计算测试"""

    def test_zero_fe_at_goal(self):
        """当所有 UAV 到达目标时，编队误差应为 0"""
        env = UAVFormationEnv(n_agents=4, seed=0)
        env.reset()
        # 将位置直接设为目标
        env.positions = env.goal_positions.copy()
        fe = env._formation_error()
        assert fe < 1e-6, f"编队误差应接近 0，实际为 {fe}"

    def test_fe_positive_when_off(self):
        """偏离目标时编队误差 > 0"""
        env = UAVFormationEnv(n_agents=4, seed=0)
        env.reset()
        env.positions = env.goal_positions + 2.0  # 偏移 2 米
        fe = env._formation_error()
        assert fe > 0, "偏离目标时编队误差应为正"


class TestSuccess:
    """成功判定测试"""

    def test_success_at_goal(self):
        """所有 UAV 在目标阈值内时应判定成功"""
        env = UAVFormationEnv(n_agents=4, success_threshold=0.3, seed=0)
        env.reset()
        env.positions = env.goal_positions + 0.1  # 小于阈值
        assert env._check_success()

    def test_no_success_far_from_goal(self):
        """距目标超过阈值时不应判定成功"""
        env = UAVFormationEnv(n_agents=4, success_threshold=0.3, seed=0)
        env.reset()
        env.positions = env.goal_positions + 1.0  # 大于阈值
        assert not env._check_success()


class TestDoneCondition:
    """终止条件测试"""

    def test_done_at_max_steps(self):
        """超过最大步数时应终止"""
        env = UAVFormationEnv(n_agents=2, max_steps=5, seed=0)
        env.reset()
        done = False
        for _ in range(5):
            actions = np.zeros((2, 3))
            _, _, _, done, _ = env.step(actions)
        assert done, "应在 max_steps 时终止"

    def test_not_done_early(self):
        """未达到最大步数且未成功时不应终止（通常）"""
        env = UAVFormationEnv(n_agents=2, max_steps=100, arena_size=50.0, seed=0)
        env.reset()
        # 重置到远离目标位置
        env.positions = env.goal_positions + 10.0
        actions = np.zeros((2, 3))
        _, _, _, done, _ = env.step(actions)
        assert not done, "第一步不应终止"


class TestPhysicalGraph:
    """物理图构建测试"""

    def test_phys_graph_range(self):
        """物理图应只包含通信范围内的边"""
        env = UAVFormationEnv(n_agents=3, R_comm=2.0, seed=0)
        env.reset()
        env.positions = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],   # 距离 1.5 <= R_comm=2.0，可连接
            [5.0, 0.0, 0.0],   # 距离 5.0 > R_comm=2.0，不可连接
        ], dtype=np.float32)
        edges = env.get_physical_graph()
        edge_set = set(edges)
        assert (0, 1) in edge_set, "(0,1) 距离 1.5 <= 2.0 应在物理图中"
        assert (0, 2) not in edge_set, "(0,2) 距离 5.0 > 2.0 不应在物理图中"
        assert (1, 2) not in edge_set, "(1,2) 距离 3.5 > 2.0 不应在物理图中"

    def test_cbf_from_env(self):
        """环境的 CBF 计算应与调度器一致"""
        from algorithms.scheduler import SafetyPriorityScheduler

        env = UAVFormationEnv(n_agents=3, d_min=0.5, seed=0)
        env.reset()

        env_cbf = env.compute_cbf_values()
        scheduler = SafetyPriorityScheduler(n_agents=3, k=3, d_min=0.5)
        sched_cbf = scheduler.compute_cbf_values(env.positions)

        for (i, j), h in env_cbf.items():
            assert abs(h - sched_cbf[(i, j)]) < 1e-9, (
                f"CBF 值不一致 ({i},{j}): env={h}, scheduler={sched_cbf[(i,j)]}"
            )


class TestFormationTypes:
    """不同编队类型测试"""

    @pytest.mark.parametrize("formation", ["circle", "line", "grid", "v_shape"])
    def test_formation_resets(self, formation):
        """每种编队类型应能正常 reset"""
        env = UAVFormationEnv(n_agents=4, formation_type=formation, seed=0)
        obs_list, info = env.reset()
        assert len(obs_list) == 4
        assert info["formation_error"] >= 0

    def test_circle_formation_symmetry(self):
        """圆形编队应是对称的（各智能体到质心距离相等）"""
        env = UAVFormationEnv(n_agents=4, formation_type="circle")
        targets = env.formation_targets  # [n, 3]
        dists = np.linalg.norm(targets, axis=1)
        assert np.allclose(dists, dists[0], atol=1e-5), "圆形编队各 UAV 到质心距离应相等"


class TestGlobalState:
    """全局状态向量测试"""

    def test_state_shape(self):
        """全局状态应为 [6n] 向量"""
        for n in [4, 6]:
            env = UAVFormationEnv(n_agents=n, seed=0)
            env.reset()
            state = env.state
            assert state.shape == (6 * n,), f"全局状态维度应为 {6*n}，实际为 {state.shape}"


class TestReset:
    """重置功能测试"""

    def test_reset_returns_consistent_obs(self):
        """多次 reset 应返回相同形状的观测"""
        env = UAVFormationEnv(n_agents=4, seed=0)
        for _ in range(5):
            obs_list, _ = env.reset()
            assert len(obs_list) == 4
            for obs in obs_list:
                assert obs.shape == (env.obs_dim,)

    def test_time_resets_to_zero(self):
        """reset 后时间步应归零"""
        env = UAVFormationEnv(n_agents=4, seed=0)
        env.reset()
        for _ in range(10):
            env.step(np.zeros((4, 3)))
        assert env.t == 10
        env.reset()
        assert env.t == 0

    def test_custom_init_positions(self):
        """自定义初始位置应被正确设置"""
        env = UAVFormationEnv(n_agents=3, seed=0)
        custom_pos = np.array([
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 0.0],
            [5.0, 6.0, 0.0],
        ], dtype=np.float32)
        env.reset(init_positions=custom_pos)
        assert np.allclose(env.positions, custom_pos)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
