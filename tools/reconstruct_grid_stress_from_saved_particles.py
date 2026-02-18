#!/usr/bin/env python3
"""
Reconstruct grid stress from saved particle stress/positions with solver-consistent P2G.

Default behavior:
- Process latest frame only under <experiment_dir>/stress_data/frame_*
- Save reconstructed files with *_reconstructed suffix
"""

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _find_config_file(experiment_dir: Path) -> Path:
    candidates = [
        experiment_dir / "config.json",
        experiment_dir / "config_backup.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No config file found in {experiment_dir}. Tried: config.json, config_backup.json"
    )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_frame_number(frame_dir: Path) -> int:
    name = frame_dir.name
    if not name.startswith("frame_"):
        raise ValueError(f"Invalid frame directory name: {name}")
    return int(name.split("_", 1)[1])


def _list_frame_dirs(stress_data_dir: Path) -> Dict[int, Path]:
    if not stress_data_dir.exists():
        raise FileNotFoundError(f"stress_data directory not found: {stress_data_dir}")

    frame_map: Dict[int, Path] = {}
    for p in stress_data_dir.iterdir():
        if p.is_dir() and p.name.startswith("frame_"):
            frame_num = _parse_frame_number(p)
            frame_map[frame_num] = p
    if not frame_map:
        raise FileNotFoundError(f"No frame_* directories found under: {stress_data_dir}")
    return frame_map


def _detect_mode(frame_dir: Path) -> str:
    d1 = frame_dir / "domain1_stress.npy"
    d2 = frame_dir / "domain2_stress.npy"
    s = frame_dir / "stress.npy"

    if d1.exists() and d2.exists():
        return "schwarz"
    if s.exists():
        return "single"

    raise FileNotFoundError(
        f"Cannot detect mode from frame directory: {frame_dir}. "
        "Expected either domain1_stress.npy/domain2_stress.npy or stress.npy"
    )


def _extract_domain_config(config: dict, mode: str, domain_name: str = "") -> dict:
    if mode == "single":
        return config
    if mode == "schwarz":
        if domain_name not in config:
            raise KeyError(f"Missing {domain_name} in config")
        return config[domain_name]
    raise ValueError(f"Unsupported mode: {mode}")


def _parse_domain_grid_params(domain_cfg: dict, domain_label: str) -> dict:
    dim = int(domain_cfg.get("dim", 2))
    if dim not in (2, 3):
        raise ValueError(f"{domain_label}: unsupported dim={dim}")

    required = ["grid_nx", "grid_ny", "domain_width", "domain_height", "particles_per_grid", "material_params"]
    if dim == 3:
        required += ["grid_nz", "domain_depth"]

    missing = [k for k in required if k not in domain_cfg]
    if missing:
        raise KeyError(f"{domain_label}: missing required config fields: {missing}")

    nx = int(domain_cfg["grid_nx"])
    ny = int(domain_cfg["grid_ny"])
    nz = int(domain_cfg["grid_nz"]) if dim == 3 else None
    domain_width = float(domain_cfg["domain_width"])
    domain_height = float(domain_cfg["domain_height"])
    domain_depth = float(domain_cfg["domain_depth"]) if dim == 3 else 1.0
    particles_per_grid = int(domain_cfg["particles_per_grid"])

    mat_params = domain_cfg["material_params"]
    if not isinstance(mat_params, list) or len(mat_params) == 0:
        raise ValueError(f"{domain_label}: material_params is empty")
    rho = float(mat_params[0]["rho"])

    offset_default = [0.0, 0.0] if dim == 2 else [0.0, 0.0, 0.0]
    offset = domain_cfg.get("offset", offset_default)
    if len(offset) < dim:
        raise ValueError(f"{domain_label}: offset dimension mismatch, got {offset}")
    offset = [float(offset[d]) for d in range(dim)]

    dx_x = domain_width / nx
    dx_y = domain_height / ny
    inv_dx_x = nx / domain_width
    inv_dx_y = ny / domain_height
    if dim == 3:
        dx_z = domain_depth / nz
        inv_dx_z = nz / domain_depth
    else:
        dx_z = None
        inv_dx_z = None

    p_vol = dx_x * dx_y * (dx_z if dim == 3 else 1.0) / particles_per_grid
    p_mass = p_vol * rho

    params = {
        "dim": dim,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "domain_width": domain_width,
        "domain_height": domain_height,
        "domain_depth": domain_depth if dim == 3 else None,
        "offset": offset,
        "dx_x": dx_x,
        "dx_y": dx_y,
        "dx_z": dx_z,
        "inv_dx_x": inv_dx_x,
        "inv_dx_y": inv_dx_y,
        "inv_dx_z": inv_dx_z,
        "rho": rho,
        "particles_per_grid": particles_per_grid,
        "p_vol": p_vol,
        "p_mass": p_mass,
    }
    return params


def _reconstruct_grid_stress_p2g(
    positions_local: np.ndarray,
    stress: np.ndarray,
    params: dict,
    mass_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    dim = params["dim"]
    grid_shape = (params["nx"], params["ny"]) if dim == 2 else (params["nx"], params["ny"], params["nz"])

    if positions_local.ndim != 2 or positions_local.shape[1] != dim:
        raise ValueError(f"positions shape mismatch: expected (N,{dim}), got {positions_local.shape}")
    if stress.ndim != 3 or stress.shape[1:] != (dim, dim):
        raise ValueError(f"stress shape mismatch: expected (N,{dim},{dim}), got {stress.shape}")
    if positions_local.shape[0] != stress.shape[0]:
        raise ValueError(
            f"particle count mismatch: positions={positions_local.shape[0]}, stress={stress.shape[0]}"
        )

    n_particles = positions_local.shape[0]
    if n_particles == 0:
        grid_stress = np.zeros(grid_shape + (dim, dim), dtype=np.float64)
        grid_mass = np.zeros(grid_shape, dtype=np.float64)
        return grid_stress, grid_mass, 0

    inv_dx = np.array(
        [params["inv_dx_x"], params["inv_dx_y"]]
        if dim == 2
        else [params["inv_dx_x"], params["inv_dx_y"], params["inv_dx_z"]],
        dtype=np.float64,
    )

    # Keep consistent with requested reconstruction rule: base = floor(pos * inv_dx - 0.5)
    base = np.floor(positions_local * inv_dx - 0.5).astype(np.int64)
    fx = positions_local * inv_dx - base

    w0 = 0.5 * (1.5 - fx) ** 2
    w1 = 0.75 - (fx - 1.0) ** 2
    w2 = 0.5 * (fx - 0.5) ** 2
    w_choices = [w0, w1, w2]

    grid_mass = np.zeros(grid_shape, dtype=np.float64)
    grid_stress_sum = np.zeros(grid_shape + (dim, dim), dtype=np.float64)

    p_mass = float(params["p_mass"])
    particle_weight = 1.0

    for offset in itertools.product(range(3), repeat=dim):
        weight = np.ones(n_particles, dtype=np.float64)
        grid_indices: List[np.ndarray] = []

        for d, od in enumerate(offset):
            weight *= w_choices[od][:, d]
            size_d = grid_shape[d]
            idx_d = (base[:, d] + od) % size_d
            grid_indices.append(idx_d.astype(np.int64))

        m_contrib = weight * p_mass * particle_weight
        idx_tuple = tuple(grid_indices)

        np.add.at(grid_mass, idx_tuple, m_contrib)
        np.add.at(
            grid_stress_sum,
            idx_tuple + (slice(None), slice(None)),
            m_contrib[:, None, None] * stress,
        )

    grid_stress = np.zeros_like(grid_stress_sum)
    valid_mask = grid_mass > mass_threshold
    grid_stress[valid_mask] = grid_stress_sum[valid_mask] / grid_mass[valid_mask][..., None, None]
    valid_count = int(np.count_nonzero(valid_mask))
    return grid_stress, grid_mass, valid_count


def _target_paths(frame_dir: Path, mode: str, domain_prefix: str = "") -> Dict[str, Path]:
    if mode == "single":
        return {
            "stress": frame_dir / "grid_stress_reconstructed.npy",
            "mass": frame_dir / "grid_mass_reconstructed.npy",
            "meta": frame_dir / "grid_stress_reconstructed_meta.json",
        }
    if mode == "schwarz":
        return {
            "stress": frame_dir / f"{domain_prefix}_grid_stress_reconstructed.npy",
            "mass": frame_dir / f"{domain_prefix}_grid_mass_reconstructed.npy",
            "meta": frame_dir / f"{domain_prefix}_grid_stress_reconstructed_meta.json",
        }
    raise ValueError(f"Unsupported mode: {mode}")


def _should_skip(outputs: Dict[str, Path], force: bool) -> bool:
    existing = [p for p in outputs.values() if p.exists()]
    if not existing:
        return False
    if force:
        return False
    print(f"[SKIP] Reconstructed outputs already exist: {[str(p) for p in existing]}")
    return True


def _process_single_domain_frame(
    frame_dir: Path,
    frame_number: int,
    domain_cfg: dict,
    config_file: Path,
    mass_threshold: float,
    force: bool,
    verbose: bool,
) -> None:
    stress_file = frame_dir / "stress.npy"
    positions_file = frame_dir / "positions.npy"
    if not stress_file.exists() or not positions_file.exists():
        raise FileNotFoundError(
            f"Missing single-domain files in {frame_dir}: {stress_file.name}, {positions_file.name}"
        )

    outputs = _target_paths(frame_dir, mode="single")
    if _should_skip(outputs, force):
        return

    params = _parse_domain_grid_params(domain_cfg, domain_label="single")

    stress = np.load(stress_file)
    positions = np.load(positions_file)
    positions_local = positions  # single-domain saved positions are already local

    grid_stress, grid_mass, valid_count = _reconstruct_grid_stress_p2g(
        positions_local=positions_local,
        stress=stress,
        params=params,
        mass_threshold=mass_threshold,
    )

    np.save(outputs["stress"], grid_stress)
    np.save(outputs["mass"], grid_mass)

    meta = {
        "mode": "single",
        "domain": "single",
        "frame": frame_number,
        "dim": params["dim"],
        "nx": params["nx"],
        "ny": params["ny"],
        "nz": params["nz"],
        "domain_width": params["domain_width"],
        "domain_height": params["domain_height"],
        "domain_depth": params["domain_depth"],
        "dx_x": params["dx_x"],
        "dx_y": params["dx_y"],
        "dx_z": params["dx_z"],
        "offset": params["offset"],
        "material_id_used": 0,
        "rho": params["rho"],
        "particles_per_grid": params["particles_per_grid"],
        "p_vol": params["p_vol"],
        "p_mass": params["p_mass"],
        "particle_weight_assumed": 1.0,
        "mass_threshold": mass_threshold,
        "n_particles_input": int(positions.shape[0]),
        "valid_grid_count": valid_count,
        "source_stress_file": str(stress_file),
        "source_positions_file": str(positions_file),
        "config_file": str(config_file),
        "notes": (
            "reconstructed from saved particle stress; may differ from runtime full-particle p2g "
            "if saved data was filtered or per-particle metadata is unavailable"
        ),
    }
    with outputs["meta"].open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[OK] frame_{frame_number} single: stress_shape={grid_stress.shape}, "
        f"valid_grids={valid_count}, outputs={outputs['stress'].name}"
    )
    if verbose:
        print(f"      source: {stress_file}")
        print(f"      source: {positions_file}")


def _process_schwarz_domain_frame(
    frame_dir: Path,
    frame_number: int,
    domain_cfg: dict,
    domain_prefix: str,
    config_file: Path,
    mass_threshold: float,
    force: bool,
    verbose: bool,
) -> None:
    stress_file = frame_dir / f"{domain_prefix}_stress.npy"
    positions_file = frame_dir / f"{domain_prefix}_positions.npy"
    if not stress_file.exists() or not positions_file.exists():
        raise FileNotFoundError(
            f"Missing schwarz files in {frame_dir}: {stress_file.name}, {positions_file.name}"
        )

    outputs = _target_paths(frame_dir, mode="schwarz", domain_prefix=domain_prefix)
    if _should_skip(outputs, force):
        return

    params = _parse_domain_grid_params(domain_cfg, domain_label=domain_prefix)

    stress = np.load(stress_file)
    positions_global = np.load(positions_file)
    offset = np.array(params["offset"], dtype=np.float64)
    positions_local = positions_global - offset[None, :]

    grid_stress, grid_mass, valid_count = _reconstruct_grid_stress_p2g(
        positions_local=positions_local,
        stress=stress,
        params=params,
        mass_threshold=mass_threshold,
    )

    np.save(outputs["stress"], grid_stress)
    np.save(outputs["mass"], grid_mass)

    meta = {
        "mode": "schwarz",
        "domain": domain_prefix,
        "frame": frame_number,
        "dim": params["dim"],
        "nx": params["nx"],
        "ny": params["ny"],
        "nz": params["nz"],
        "domain_width": params["domain_width"],
        "domain_height": params["domain_height"],
        "domain_depth": params["domain_depth"],
        "dx_x": params["dx_x"],
        "dx_y": params["dx_y"],
        "dx_z": params["dx_z"],
        "offset": params["offset"],
        "material_id_used": 0,
        "rho": params["rho"],
        "particles_per_grid": params["particles_per_grid"],
        "p_vol": params["p_vol"],
        "p_mass": params["p_mass"],
        "particle_weight_assumed": 1.0,
        "mass_threshold": mass_threshold,
        "n_particles_input": int(positions_global.shape[0]),
        "valid_grid_count": valid_count,
        "source_stress_file": str(stress_file),
        "source_positions_file": str(positions_file),
        "config_file": str(config_file),
        "notes": (
            "reconstructed from saved particle stress; may differ from runtime full-particle p2g "
            "if saved data was filtered or per-particle metadata is unavailable"
        ),
    }
    with outputs["meta"].open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[OK] frame_{frame_number} {domain_prefix}: stress_shape={grid_stress.shape}, "
        f"valid_grids={valid_count}, outputs={outputs['stress'].name}"
    )
    if verbose:
        print(f"      source: {stress_file}")
        print(f"      source: {positions_file}")


def _select_frames(frame_map: Dict[int, Path], frame: int, all_frames: bool) -> List[Tuple[int, Path]]:
    if frame is not None and all_frames:
        raise ValueError("--frame and --all-frames cannot be used together")

    if frame is not None:
        if frame not in frame_map:
            available = sorted(frame_map.keys())
            raise ValueError(f"frame_{frame} not found. Available frames: {available}")
        return [(frame, frame_map[frame])]

    if all_frames:
        return [(k, frame_map[k]) for k in sorted(frame_map.keys())]

    latest = max(frame_map.keys())
    return [(latest, frame_map[latest])]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct grid stress from saved particle stress with solver-consistent P2G."
    )
    parser.add_argument("--dir", required=True, help="Experiment directory path")
    parser.add_argument("--frame", type=int, default=None, help="Only process this frame number")
    parser.add_argument("--all-frames", action="store_true", help="Process all frame_* directories")
    parser.add_argument("--force", action="store_true", help="Overwrite existing reconstructed files")
    parser.add_argument("--mass-threshold", type=float, default=1e-10, help="Mass threshold for valid grid")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    experiment_dir = Path(args.dir).resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    config_file = _find_config_file(experiment_dir)
    config = _load_json(config_file)
    stress_data_dir = experiment_dir / "stress_data"
    frame_map = _list_frame_dirs(stress_data_dir)
    selected = _select_frames(frame_map, args.frame, args.all_frames)

    print(f"Experiment dir: {experiment_dir}")
    print(f"Config file: {config_file}")
    print(f"Frames to process: {[f for f, _ in selected]}")

    for frame_number, frame_dir in selected:
        mode = _detect_mode(frame_dir)
        print(f"\nProcessing frame_{frame_number} ({mode}) at {frame_dir}")

        if mode == "single":
            _process_single_domain_frame(
                frame_dir=frame_dir,
                frame_number=frame_number,
                domain_cfg=_extract_domain_config(config, mode="single"),
                config_file=config_file,
                mass_threshold=args.mass_threshold,
                force=args.force,
                verbose=args.verbose,
            )
        else:
            _process_schwarz_domain_frame(
                frame_dir=frame_dir,
                frame_number=frame_number,
                domain_cfg=_extract_domain_config(config, mode="schwarz", domain_name="Domain1"),
                domain_prefix="domain1",
                config_file=config_file,
                mass_threshold=args.mass_threshold,
                force=args.force,
                verbose=args.verbose,
            )
            _process_schwarz_domain_frame(
                frame_dir=frame_dir,
                frame_number=frame_number,
                domain_cfg=_extract_domain_config(config, mode="schwarz", domain_name="Domain2"),
                domain_prefix="domain2",
                config_file=config_file,
                mass_threshold=args.mass_threshold,
                force=args.force,
                verbose=args.verbose,
            )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
