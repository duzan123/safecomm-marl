"""SafeComm-VoI Phase 1 training entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from algorithms.safecomm_psched import SafeCommVoI
from envs.uav_formation_env import UAVFormationEnv


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return cfg


def _config_section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    section = cfg.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"config section '{name}' must be a mapping")
    return section


def _mutable_config_section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in cfg or cfg[name] is None:
        cfg[name] = {}
    section = cfg[name]
    if not isinstance(section, dict):
        raise ValueError(f"config section '{name}' must be a mapping")
    return section


def flatten_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested YAML sections into the config expected by SafeCommVoI."""
    env_cfg = _config_section(cfg, "env")
    comm_cfg = _config_section(cfg, "comm")
    network_cfg = _config_section(cfg, "network")
    ppo_cfg = _config_section(cfg, "ppo")
    lr_cfg = _config_section(cfg, "lr")
    safety_cfg = _config_section(cfg, "safety")
    training_cfg = _config_section(cfg, "training")

    return {
        "k": comm_cfg.get("k", 4),
        "schedule_mode": comm_cfg.get("schedule_mode", "safety_priority"),
        "beta": comm_cfg.get("beta", 1.0),
        "tau_ref": comm_cfg.get("tau_ref", 1.0),
        "tau_max": comm_cfg.get("tau_max", 5.0),
        "v_max_cbf": comm_cfg.get("v_max_cbf", 3.0),
        "msg_dim": network_cfg.get("msg_dim", 64),
        "hidden_dims": network_cfg.get("hidden_dims", [256, 256]),
        "H": network_cfg.get("H", 2),
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
        "lr_actor": lr_cfg.get("actor", 3.0e-4),
        "lr_critic": lr_cfg.get("critic", 1.0e-3),
        "lr_lambda": lr_cfg.get("lambda", 0.01),
        "safety_budget": safety_cfg.get("budget", 0.1),
        "lambda_init": safety_cfg.get("lambda_init", 0.0),
        "lambda_max": safety_cfg.get("lambda_max", 10.0),
        "log_interval": training_cfg.get("log_interval", 10),
        "eval_interval": training_cfg.get("eval_interval", 50),
        "dt": env_cfg.get("dt", 0.1),
        "d_min": env_cfg.get("d_min", 0.5),
        "action_scale": env_cfg.get("a_max", 2.0),
        "seed": training_cfg.get("seed", env_cfg.get("seed")),
    }


def _apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    env_cfg = _mutable_config_section(cfg, "env")
    comm_cfg = _mutable_config_section(cfg, "comm")
    training_cfg = _mutable_config_section(cfg, "training")

    if args.n_agents is not None:
        env_cfg["n_agents"] = args.n_agents
    if args.k is not None:
        comm_cfg["k"] = args.k
    if args.beta is not None:
        comm_cfg["beta"] = args.beta
    if args.total_steps is not None:
        training_cfg["total_steps"] = args.total_steps
    if args.seed is not None:
        env_cfg["seed"] = args.seed
        training_cfg["seed"] = args.seed
    if args.device is not None:
        training_cfg["device"] = args.device


def _build_env(cfg: dict[str, Any]) -> UAVFormationEnv:
    env_cfg = _config_section(cfg, "env")
    return UAVFormationEnv(
        n_agents=env_cfg.get("n_agents", 4),
        dt=env_cfg.get("dt", 0.1),
        max_steps=env_cfg.get("max_steps", 200),
        d_min=env_cfg.get("d_min", 0.5),
        R_comm=env_cfg.get("R_comm", 5.0),
        v_max=env_cfg.get("v_max", 3.0),
        a_max=env_cfg.get("a_max", 2.0),
        arena_size=env_cfg.get("arena_size", 10.0),
        w_formation=env_cfg.get("w_formation", 1.0),
        w_success=env_cfg.get("w_success", 5.0),
        success_threshold=env_cfg.get("success_threshold", 0.3),
        K_obs_neighbors=env_cfg.get("K_obs_neighbors", 4),
        seed=env_cfg.get("seed", 42),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SafeComm-VoI Phase 1 for UAV formation MARL."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--n_agents", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)

    checkpoint_path = None
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    training_cfg = _config_section(cfg, "training")
    env = _build_env(cfg)
    device = training_cfg.get("device", "auto")
    agent = SafeCommVoI(env, config=flatten_config(cfg), device=device)

    if checkpoint_path is not None:
        agent.load(checkpoint_path)

    metrics = agent.train(total_steps=training_cfg.get("total_steps", 1000000))

    checkpoint_dir = training_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_checkpoint = os.path.join(checkpoint_dir, "final.pt")
    agent.save(final_checkpoint)

    print("[train]")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"  checkpoint: {final_checkpoint}")

    print("[evaluate]")
    eval_metrics = agent.evaluate(
        n_episodes=training_cfg.get("eval_episodes", 10),
        deterministic=True,
    )
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
