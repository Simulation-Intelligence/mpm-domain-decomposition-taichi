import numpy as np
import time
import taichi as ti

class ParticleRecorder:
    def __init__(self, palette=None, max_frames=300):
        """
        通用粒子动画录制器
        
        参数：
            palette: 颜色调色板，格式为uint32数组 (可选)
            max_frames: 最大录制帧数
        """
        # 默认调色板：[普通颜色，边界颜色]
        self.palette = np.array([0xFFFFFF, 0x66CCFF], dtype=np.uint32) if palette is None else palette
        self.frame_data = []
        self.max_frames = max_frames
        self.gui=ti.GUI("Record", res=1200)


    def capture(self, positions: np.ndarray, color_indices: np.ndarray):
        """
        捕获当前帧数据
        
        参数：
            positions: (N,2) 粒子位置数组，float32
            color_indices: (N,) 颜色索引数组，uint32
        """

        # 数据校验
        assert positions.dtype == np.float32, "Positions must be float32"
        assert color_indices.dtype == np.uint32, "Color indices must be uint32"
        assert positions.shape[0] == color_indices.shape[0], "Input arrays must have same length"
        
        # 存储为元组 (深拷贝避免数据污染)
        self.frame_data.append((
            positions.copy(),
            color_indices.copy()
        ))

    def play(self, loop=True, fps=60):
        """
        播放录制动画
        
        参数：
            gui: Taichi GUI实例
            loop: 是否循环播放
            fps: 帧率控制
        """
        if not self.frame_data:
            return

        self.recording = False
        frame_delay = 1.0 / fps
        current_idx = 0

        while self.gui.running:
            print(f"Playing frame {current_idx+1}/{len(self.frame_data)}")
            # 获取当前帧数据
            pos, indices = self.frame_data[current_idx]
            
            # 批量绘制粒子
            self.gui.circles(
                pos,
                palette=self.palette,
                palette_indices=indices,
                radius=3
            )
            self.gui.show()

            # 更新帧索引
            current_idx = (current_idx + 1) % len(self.frame_data)
            if not loop and current_idx == 0:
                print("Playback finished.")
                break

            # time.sleep(frame_delay)

# 使用示例
if __name__ == "__main__":
    # 初始化录制器（自定义调色板）
    palette = np.array([
        0x068587,  # 域1颜色
        0xED553B,  # 域2颜色
        0x66CCFF   # 边界颜色
    ], dtype=np.uint32)
    
    recorder = ParticleRecorder(palette=palette, max_frames=200)
    
    # 模拟捕获数据（示例数据）
    for _ in range(200):
        fake_positions = np.random.rand(1000, 2).astype(np.float32)
        fake_indices = np.random.randint(0, 3, 1000, dtype=np.uint32)
        recorder.capture(fake_positions, fake_indices)
    
    # 播放动画
    print("Start playback...")
    recorder.play(loop=True)