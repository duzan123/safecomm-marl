"""Tests for the SafeComm-VoI training entry point."""

import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import train


def _minimal_config(tmp_path):
    return {
        "env": {
            "n_agents": 4,
            "dt": 0.1,
            "max_steps": 1,
            "d_min": 0.5,
            "R_comm": 5.0,
            "v_max": 3.0,
            "a_max": 2.0,
            "arena_size": 10.0,
            "w_formation": 1.0,
            "w_success": 5.0,
            "success_threshold": 0.3,
            "K_obs_neighbors": 4,
            "seed": 0,
        },
        "comm": {
            "k": 4,
            "schedule_mode": "safety_priority",
            "beta": 1.0,
            "tau_ref": 1.0,
            "tau_max": 5.0,
            "v_max_cbf": 3.0,
        },
        "network": {"msg_dim": 8, "hidden_dims": [16, 16], "H": 1},
        "ppo": {
            "n_steps": 1,
            "n_epochs": 1,
            "batch_size": 1,
            "gamma": 0.99,
            "lam": 0.95,
            "clip_eps": 0.2,
            "value_coef": 0.5,
            "cost_value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
        },
        "lr": {"actor": 3.0e-4, "critic": 1.0e-3, "lambda": 0.01},
        "safety": {"budget": 0.1, "lambda_init": 0.0, "lambda_max": 10.0},
        "training": {
            "total_steps": 0,
            "eval_episodes": 0,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "device": "cpu",
            "seed": 0,
        },
    }


def test_missing_checkpoint_path_fails_clearly(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_minimal_config(tmp_path)), encoding="utf-8")
    missing_checkpoint = tmp_path / "missing.pt"

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--checkpoint", str(missing_checkpoint)],
    )

    with pytest.raises(FileNotFoundError, match="missing.pt"):
        train.main()


def test_default_config_path_loads_outside_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train.py"])

    args = train._parse_args()
    cfg = train.load_config(args.config)

    assert cfg["comm"]["beta"] == 1.0


def test_flatten_config_rejects_non_mapping_section():
    cfg = _minimal_config(Path("."))
    cfg["comm"] = ["not", "a", "mapping"]

    with pytest.raises(ValueError, match="comm.*mapping"):
        train.flatten_config(cfg)
