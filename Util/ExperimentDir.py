"""
实验目录管理：创建带时间戳的实验目录并备份配置文件。
"""

import os
import shutil
from datetime import datetime


def create_experiment_directory(prefix: str, config_path: str = None) -> tuple:
    """创建实验目录结构并备份配置文件。

    Args:
        prefix: 目录名前缀（如 'single_domain' 或 'schwarz'）
        config_path: 配置文件路径（可选，如果存在则复制为备份）

    Returns:
        (experiment_dir, stress_data_dir) 元组
    """
    # 创建带时间戳的主实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = "experiment_results"
    experiment_name = f"{prefix}_{timestamp}"
    experiment_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # 备份配置文件
    if config_path and os.path.exists(config_path):
        config_backup_path = os.path.join(experiment_dir, "config_backup.json")
        shutil.copy2(config_path, config_backup_path)
        print(f"配置文件已备份到: {config_backup_path}")

    # 创建 stress_data 子目录
    stress_data_dir = os.path.join(experiment_dir, "stress_data")
    os.makedirs(stress_data_dir, exist_ok=True)

    return experiment_dir, stress_data_dir
