"""
Checkpoint save/load helpers shared by single-domain and Schwarz simulators.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_DIRNAME = "checkpoint"
MANIFEST_FILENAME = "manifest.json"
PERF_STATS_FILENAME = "perf_stats.json"
CONFIG_BACKUP_FILENAME = "config_backup.json"

PARTICLE_STATE_FIELDS = (
    "x",
    "v",
    "F",
    "C",
    "particle_weight",
    "is_boundary_particle",
    "particle_material_id",
    "particle_mu",
    "particle_lam",
    "is_move_boundary_particle",
    "target_position",
    "arc_center",
    "arc_axis",
    "volume_force",
)


def get_checkpoint_dir(experiment_dir) -> Path:
    return Path(experiment_dir) / CHECKPOINT_DIRNAME


def get_manifest_path(experiment_dir) -> Path:
    return get_checkpoint_dir(experiment_dir) / MANIFEST_FILENAME


def get_perf_stats_path(experiment_dir) -> Path:
    return get_checkpoint_dir(experiment_dir) / PERF_STATS_FILENAME


def get_config_backup_path(experiment_dir) -> Path:
    config_backup_path = Path(experiment_dir) / CONFIG_BACKUP_FILENAME
    if not config_backup_path.exists():
        raise FileNotFoundError(
            f"恢复目录缺少配置备份文件: {config_backup_path}"
        )
    return config_backup_path


def save_manifest(experiment_dir, manifest: dict) -> Path:
    manifest_path = get_manifest_path(experiment_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(manifest)
    payload.setdefault("schema_version", CHECKPOINT_SCHEMA_VERSION)
    payload.setdefault("config_path", CONFIG_BACKUP_FILENAME)

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return manifest_path


def load_manifest(experiment_dir, expected_simulator_type: str) -> dict:
    manifest_path = get_manifest_path(experiment_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"恢复目录缺少 checkpoint manifest: {manifest_path}"
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    simulator_type = manifest.get("simulator_type")
    if simulator_type != expected_simulator_type:
        raise ValueError(
            "checkpoint simulator_type 不匹配: "
            f"期望 {expected_simulator_type}, 实际为 {simulator_type}"
        )

    return manifest


def save_particle_state(particles, state_dir) -> Path:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    for field_name in PARTICLE_STATE_FIELDS:
        np.save(
            state_dir / f"{field_name}.npy",
            getattr(particles, field_name).to_numpy(),
        )

    return state_dir


def load_particle_state(particles, state_dir) -> Path:
    state_dir = Path(state_dir)
    if not state_dir.exists():
        raise FileNotFoundError(f"恢复目录缺少粒子状态目录: {state_dir}")

    for field_name in PARTICLE_STATE_FIELDS:
        field_path = state_dir / f"{field_name}.npy"
        if not field_path.exists():
            raise FileNotFoundError(f"恢复目录缺少粒子状态文件: {field_path}")

        loaded_array = np.load(field_path, allow_pickle=False)
        target_field = getattr(particles, field_name)
        expected_shape = target_field.to_numpy().shape

        if loaded_array.shape != expected_shape:
            raise ValueError(
                "checkpoint 粒子状态形状不匹配: "
                f"{field_name} 期望 {expected_shape}, 实际 {loaded_array.shape}. "
                "请检查配置中的粒子数量或维度是否发生变化。"
            )

        target_field.from_numpy(loaded_array)

    return state_dir


def save_performance_stats(perf_stats, experiment_dir) -> Path:
    perf_stats_path = get_perf_stats_path(experiment_dir)
    perf_stats_path.parent.mkdir(parents=True, exist_ok=True)
    perf_stats.save_to_file(str(perf_stats_path))
    return perf_stats_path


def load_performance_stats(perf_stats, experiment_dir) -> bool:
    perf_stats_path = get_perf_stats_path(experiment_dir)
    if not perf_stats_path.exists():
        print(f"提示: 未找到 checkpoint 性能统计文件，跳过恢复: {perf_stats_path}")
        return False

    perf_stats.load_from_file(str(perf_stats_path))
    return True
