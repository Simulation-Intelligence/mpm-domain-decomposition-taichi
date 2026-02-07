import numpy as np
import time
import taichi as ti
import gc

class ParticleRecorder:
    def __init__(self, palette=None, max_frames=300, lines_begin=None, lines_end=None, lines_color=None):
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

        self.lines_begin = lines_begin if lines_begin is not None else np.array([], dtype=np.float32)
        self.lines_end = lines_end if lines_end is not None else np.array([], dtype=np.float32)
        self.lines_color = lines_color if lines_color is not None else np.array([], dtype=np.uint32)

        self.gui=ti.ui.Window("Record", res=(800, 800), vsync=False)


    def capture(self, positions: np.ndarray, color_indices: np.ndarray):
        """
        捕获当前帧数据
        
        参数：
            positions: (N,2) 粒子位置数组，float32
            color_indices: (N,) 颜色索引数组，uint32
        """

        # 数据校验
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
            
            # 使用ti.ui.Window的canvas API
            canvas = self.gui.get_canvas()
            canvas.set_background_color((0.067, 0.184, 0.255))

            # 创建临时Taichi字段用于渲染
            n_particles = len(pos)
            if n_particles > 0:
                # 创建临时字段来存储位置数据
                temp_pos = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
                temp_pos.from_numpy(pos)

                # 根据索引绘制不同颜色的粒子
                for i, color in enumerate(self.palette):
                    mask = indices == i
                    if np.any(mask):
                        # 将uint32颜色转换为归一化的RGB
                        r = ((color >> 16) & 0xFF) / 255.0
                        g = ((color >> 8) & 0xFF) / 255.0
                        b = (color & 0xFF) / 255.0

                        # 创建过滤后的位置字段
                        filtered_pos = pos[mask]
                        if len(filtered_pos) > 0:
                            temp_filtered = ti.Vector.field(2, dtype=ti.f32, shape=len(filtered_pos))
                            temp_filtered.from_numpy(filtered_pos.astype(np.float32))
                            canvas.circles(temp_filtered, radius=0.005, color=(r, g, b))

            if self.lines_begin.size > 0 and self.lines_end.size > 0:
                # 绘制线条 - 创建临时字段
                for i in range(len(self.lines_begin)):
                    color = self.lines_color[i] if i < len(self.lines_color) else 0xFFFFFF
                    r = ((color >> 16) & 0xFF) / 255.0
                    g = ((color >> 8) & 0xFF) / 255.0
                    b = (color & 0xFF) / 255.0

                    # 将begin和end点组合成vertices格式
                    line_vertices = np.array([self.lines_begin[i], self.lines_end[i]], dtype=np.float32)
                    temp_line = ti.Vector.field(2, dtype=ti.f32, shape=2)
                    temp_line.from_numpy(line_vertices)
                    canvas.lines(temp_line, 0.001, color=(r, g, b))

            self.gui.show()

            # 更新帧索引
            current_idx = (current_idx + 1) % len(self.frame_data)
            if not loop and current_idx == 0:
                print("Playback finished.")
                break

            # 每1000帧强制垃圾回收一次，释放临时Taichi字段
            if current_idx % 1000 == 0:
                gc.collect()

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