import numpy as np
import taichi as ti


class _PreallocFrameData:
    def __init__(self, recorder):
        self._recorder = recorder

    def __len__(self):
        return self._recorder._frame_cursor

    def __getitem__(self, idx):
        if idx < 0:
            idx = self._recorder._frame_cursor + idx
        if idx < 0 or idx >= self._recorder._frame_cursor:
            raise IndexError("frame index out of range")
        count = int(self._recorder._frame_counts[idx])
        return (
            self._recorder._frames_pos[idx, :count],
            self._recorder._frames_color[idx, :count],
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class ParticleRecorder:
    def __init__(self, palette=None, max_frames=300, lines_begin=None, lines_end=None, lines_color=None, max_particles=None):
        """
        通用粒子动画录制器

        参数：
            palette: 颜色调色板，格式为uint32数组 (可选)
            max_frames: 最大录制帧数
            max_particles: 最大粒子数，提供时启用预分配模式
        """
        # 默认调色板：[普通颜色，边界颜色]
        self.palette = np.array([0xFFFFFF, 0x66CCFF], dtype=np.uint32) if palette is None else palette
        self.max_frames = max_frames
        self.max_particles = max_particles
        self._prealloc = self.max_particles is not None

        if self._prealloc:
            self._frames_pos = np.zeros((self.max_frames, self.max_particles, 2), dtype=np.float32)
            self._frames_color = np.zeros((self.max_frames, self.max_particles), dtype=np.uint32)
            self._frame_counts = np.zeros((self.max_frames,), dtype=np.int32)
            self._frame_cursor = 0
            self.frame_data = _PreallocFrameData(self)
            # 预分配每种颜色的粒子位置缓冲区（playback 复用）
            self._play_bufs = [
                np.empty((self.max_particles, 2), dtype=np.float32)
                for _ in self.palette
            ]
        else:
            self.frame_data = []
            self._play_bufs = None

        lines_begin = lines_begin if lines_begin is not None else np.array([], dtype=np.float32)
        lines_end = lines_end if lines_end is not None else np.array([], dtype=np.float32)
        lines_color = lines_color if lines_color is not None else np.array([], dtype=np.uint32)

        self.lines_begin = lines_begin
        self.lines_end = lines_end
        self.lines_color = lines_color

        # 预分组线条数据：按颜色分组，playback 直接复用，不重新创建数组
        self._line_groups = {}  # color_hex (int) -> (begin_np, end_np)
        for i in range(len(lines_begin)):
            c = int(lines_color[i]) if i < len(lines_color) else 0xFFFFFF
            if c not in self._line_groups:
                self._line_groups[c] = ([], [])
            self._line_groups[c][0].append(lines_begin[i])
            self._line_groups[c][1].append(lines_end[i])
        self._line_groups = {
            c: (np.array(b, dtype=np.float32), np.array(e, dtype=np.float32))
            for c, (b, e) in self._line_groups.items()
        }

        # 使用 legacy ti.GUI：circles/lines 直接接受 numpy，无需 Taichi 字段
        self.gui = ti.GUI("Record", res=(800, 800), fast_gui=True)


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

        if self._prealloc:
            if self._frame_cursor >= self.max_frames:
                print("Warning: Recorder buffer full, dropping frame")
                return
            count = positions.shape[0]
            if count > self.max_particles:
                print("Warning: Frame has more particles than max_particles, truncating")
                count = self.max_particles
            self._frame_counts[self._frame_cursor] = count
            np.copyto(self._frames_pos[self._frame_cursor, :count], positions[:count])
            np.copyto(self._frames_color[self._frame_cursor, :count], color_indices[:count])
            self._frame_cursor += 1
            return

        # 存储为元组 (深拷贝避免数据污染)
        self.frame_data.append((
            positions.copy(),
            color_indices.copy()
        ))

    def play(self, loop=True):
        """
        播放录制动画

        参数：
            loop: 是否循环播放
            fps: 帧率控制
        """
        if self._prealloc:
            total_frames = self._frame_cursor
            if total_frames == 0:
                return
        else:
            if not self.frame_data:
                return
            total_frames = len(self.frame_data)

        current_idx = 0

        while self.gui.running:
            # 获取当前帧数据（view，不拷贝）
            if self._prealloc:
                count = int(self._frame_counts[current_idx])
                pos = self._frames_pos[current_idx, :count]
                indices = self._frames_color[current_idx, :count]
            else:
                pos, indices = self.frame_data[current_idx]
                count = len(pos)

            print(f"Playing frame {current_idx + 1}/{total_frames}")

            self.gui.clear(0x112F41)

            # 按颜色分组绘制粒子（直接用 numpy，不创建 Taichi 字段）
            if count > 0:
                for ci, color in enumerate(self.palette):
                    mask = indices == ci
                    n = int(np.count_nonzero(mask))
                    if n > 0:
                        if self._prealloc and self._play_bufs is not None:
                            buf = self._play_bufs[ci]
                            buf[:n] = pos[mask]
                            self.gui.circles(buf[:n], color=int(color), radius=3)
                        else:
                            self.gui.circles(pos[mask], color=int(color), radius=3)

            # 绘制线条（使用预分组 numpy，不创建 Taichi 字段）
            for color_hex, (begin_np, end_np) in self._line_groups.items():
                self.gui.lines(begin_np, end_np, radius=1, color=color_hex)

            self.gui.show()

            # 更新帧索引
            current_idx = (current_idx + 1) % total_frames
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
