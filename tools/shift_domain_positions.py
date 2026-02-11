#!/usr/bin/env python3
"""
Shift Domain1/Domain2 particle positions by a fixed 2D vector.

Target files in the given directory:
  - domain1_positions.npy
  - domain2_positions.npy
"""

import argparse
import os
import shutil
import sys

import numpy as np


TARGET_FILES = ("domain1_positions.npy", "domain2_positions.npy")


def _format_range(arr: np.ndarray) -> str:
    x_min, y_min = arr.min(axis=0)
    x_max, y_max = arr.max(axis=0)
    return f"x:[{x_min:.6f}, {x_max:.6f}] y:[{y_min:.6f}, {y_max:.6f}]"


def shift_positions_in_dir(target_dir: str, shift_x: float, shift_y: float, backup: bool, dry_run: bool) -> int:
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Directory not found: {target_dir}")

    found_any = False
    updated_count = 0

    for filename in TARGET_FILES:
        path = os.path.join(target_dir, filename)
        if not os.path.exists(path):
            print(f"[WARN] File not found, skip: {path}")
            continue

        found_any = True
        positions = np.load(path)

        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"Invalid shape for {path}: {positions.shape}, expected (N, 2)")

        shift = np.array([shift_x, shift_y], dtype=positions.dtype)
        shifted = positions + shift

        print(f"\nFile: {path}")
        print(f"  before: {_format_range(positions)}")
        print(f"  after : {_format_range(shifted)}")
        print(f"  shift : [{shift_x}, {shift_y}]")

        if dry_run:
            print("  mode  : dry-run, no file written")
            continue

        if backup:
            backup_path = path + ".bak"
            shutil.copy2(path, backup_path)
            print(f"  backup: {backup_path}")

        np.save(path, shifted)
        updated_count += 1
        print("  write : done")

    if not found_any:
        raise FileNotFoundError(
            f"No target files found in {target_dir}. "
            f"Expected: {', '.join(TARGET_FILES)}"
        )

    print(f"\nCompleted. Updated {updated_count} file(s).")
    return updated_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Shift domain1/domain2 positions in a frame directory by a fixed 2D vector."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Target directory containing domain1_positions.npy/domain2_positions.npy",
    )
    parser.add_argument(
        "--shift",
        nargs=2,
        type=float,
        required=True,
        metavar=("DX", "DY"),
        help="2D shift vector (dx dy)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak backup before writing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing files",
    )

    args = parser.parse_args()
    dx, dy = args.shift

    try:
        shift_positions_in_dir(
            target_dir=args.dir,
            shift_x=dx,
            shift_y=dy,
            backup=args.backup,
            dry_run=args.dry_run,
        )
        return 0
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
