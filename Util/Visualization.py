import numpy as np
import cv2
from collections import deque
from time import time


class FPSCounter:
    """计算和显示 FPS 的计数器"""

    def __init__(self, window_size=30):
        """初始化 FPS 计数器。

        Args:
            window_size: 用于计算 FPS 的帧数窗口大小
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time()

    def update(self):
        """更新 FPS 计数器（每帧调用一次）"""
        current_time = time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time

    def get_fps(self):
        """获取当前 FPS"""
        if len(self.frame_times) == 0:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def draw_on_image(self, img, position=(10, 30), font_scale=0.7, color=(0, 255, 0), thickness=2):
        """在图像上绘制 FPS 文本。

        Args:
            img: numpy array，要绘制的图像
            position: (x, y) tuple，文本位置
            font_scale: 字体缩放因子
            color: BGR 颜色
            thickness: 文字粗度
        """
        fps = self.get_fps()
        text = f"FPS: {fps:.1f}"
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def cv2_draw_particles(img, positions, radius, bgr_color):
    """在 img 上用实心圆绘制粒子（向量化，快速）。

    Args:
        img: numpy array (H, W, 3) BGR 图像
        positions: (N, 2) float array，值在 [0, 1] 范围（规范化坐标）
        radius: int，圆形半径（像素）
        bgr_color: tuple (B, G, R)，颜色
    """
    H, W = img.shape[:2]
    xs = np.clip((positions[:, 0] * W).astype(np.int32), 0, W - 1)
    ys = np.clip(((1.0 - positions[:, 1]) * H).astype(np.int32), 0, H - 1)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                img[np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)] = bgr_color


def cv2_draw_particles_colored(img, positions, hex_colors, radius):
    """在 img 上用逐粒子颜色（0xRRGGBB uint32）绘制粒子（向量化）。

    Args:
        img: numpy array (H, W, 3) BGR 图像
        positions: (N, 2) float array，值在 [0, 1] 范围（规范化坐标）
        hex_colors: (N,) uint32 array，RGB hex 颜色 (0xRRGGBB)
        radius: int，圆形半径（像素）
    """
    H, W = img.shape[:2]
    xs = np.clip((positions[:, 0] * W).astype(np.int32), 0, W - 1)
    ys = np.clip(((1.0 - positions[:, 1]) * H).astype(np.int32), 0, H - 1)
    r = ((hex_colors >> 16) & 0xFF).astype(np.uint8)
    g = ((hex_colors >> 8) & 0xFF).astype(np.uint8)
    b = (hex_colors & 0xFF).astype(np.uint8)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                vy = np.clip(ys + dy, 0, H - 1)
                vx = np.clip(xs + dx, 0, W - 1)
                img[vy, vx, 0] = b
                img[vy, vx, 1] = g
                img[vy, vx, 2] = r
