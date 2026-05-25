"""
SafeComm-PSched 训练入口

用法：
  python train.py                             # 使用默认配置
  python train.py --config configs/default.yaml
  python train.py --n_agents 6 --k 4 --total_steps 500000
"""

import argparse
import yaml
import os

from envs.uav_formation_env import UAVFormationEnv
from algorithms.safecomm_psched import SafeCommPSched


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def flatten_config(cfg: dict) -> dict:
    """将嵌套的 YAML 配置展平为 SafeCommPSched 所需的扁平字典。"""
    flat = {}

    env_cfg = cfg.get("env", {})
    comm_cfg = cfg.get("comm", {})
    net_cfg = cfg.get("network", {})
    ppo_cfg = cfg.get("ppo", {})
    lr_cfg = cfg.get("lr", {})
    safety_cfg = cfg.get("safety", {})
    train_cfg = cfg.get("training", {})

    flat.update({
        # 通信
        "k": comm_cfg.get("k", 4),
        "schedule_mode": comm_cfg.get("schedule_mode", "safety_priority"),
        # 网络
        "msg_dim": net_cfg.get("msg_dim", 64),
        "hidden_dims": net_cfg.get("hidden_dims", [256, 256]),
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
    })
    return flat


def main():
    parser = argparse.ArgumentParser(description="SafeComm-PSched 训练")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n_agents", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, default=None, help="从此检查点继续训练")
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)

    # CLI 覆盖
    if args.n_agents is not None:
        cfg["env"]["n_agents"] = args.n_agents
    if args.k is not None:
        cfg["comm"]["k"] = args.k
    if args.total_steps is not None:
        cfg["training"]["total_steps"] = args.total_steps
    if args.seed is not None:
        cfg["env"]["seed"] = args.seed
        cfg["training"]["seed"] = args.seed

    env_cfg = cfg.get("env", {})
    train_cfg = cfg.get("training", {})

    # 构建环境
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
        K_obs_neighbors=env_cfg.get("K_obs_neighbors", 2),
        seed=env_cfg.get("seed", 42),
    )

    # 展平算法配置
    agent_config = flatten_config(cfg)

    # 构建算法
    agent = SafeCommPSched(
        env=env,
        config=agent_config,
        device=args.device if args.device else train_cfg.get("device", "auto"),
    )

    # 可选：从检查点恢复
    if args.checkpoint and os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)

    # 训练
    total_steps = train_cfg.get("total_steps", 1_000_000)
    metrics = agent.train(total_steps=total_steps)

    # 保存最终模型
    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    agent.save(os.path.join(checkpoint_dir, "final.pt"))

    # 最终评估
    print("\n[最终评估]")
    eval_metrics = agent.evaluate(n_episodes=20, deterministic=True)
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
