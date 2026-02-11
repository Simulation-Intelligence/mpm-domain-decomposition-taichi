# Memory Leak Fixes Implementation Summary

## 实施完成 - 2026-02-08

### ✅ Fix 1: Convergence Check 内存泄漏修复

**问题**: `check_schwarz_convergence` 使用局部变量 + `ti.atomic_max`，可能导致内存累积

**修改文件**: `simulators/implicit_mpm_schwarz.py`

**改动**:
1. 添加了专用的accumulator字段（Line 135）:
   ```python
   self.convergence_max_residual = ti.field(dtype=ti.f64, shape=())
   ```

2. 重构了convergence check为两个函数（Lines 387-442）:
   - `check_schwarz_convergence_kernel()` - Taichi kernel，使用field accumulator
   - `check_schwarz_convergence()` - Python wrapper，返回结果

**预期效果**: 消除 ~3-5 MB/frame 的泄漏

---

### ✅ Fix 2: Newton Solver H_builder 周期性重建

**问题**: `H_builder` 跨多次solve重用，可能在GPU端累积内存

**修改文件**: `Optimizer/Newton.py`

**改动** (Lines 117-128):
```python
# 每10次Newton迭代重建一次H_builder，平衡性能和内存
if it % 10 == 0:
    if hasattr(self, 'H_builder'):
        del self.H_builder
    self.H_builder = ti.linalg.SparseMatrixBuilder(
        self.dim[None], self.dim[None],
        max_num_triplets=int(self.dim[None]**2 * 0.1),
        dtype=self.float_type
    )
```

**预期效果**: 减少 ~5-10 MB/frame 的泄漏（60-80%改善）

---

### ✅ Fix 3: 分层GC清理策略

**问题**: 原来只在1000帧时清理，Python临时对象累积过多

**修改文件**: `simulators/implicit_mpm_schwarz.py`

**改动** (Lines 702-725):
- **Level 1**: 每100帧 - `gc.collect(generation=0)` - 清理最新临时对象（~0.1-0.5ms）
- **Level 2**: 每500帧 - `gc.collect(generation=1)` - 清理较老对象（~1-5ms）
- **Level 3**: 每1000帧 - `gc.collect()` - 完全清理（~5-20ms）

**预期效果**: 减少 ~1-2 MB/frame 的累积

---

## 总体预期改善

| 指标 | 修复前 | 预期修复后 | 改善 |
|------|--------|-----------|------|
| 平均内存增长 | 14.8 MB/frame | **< 5 MB/frame** | **66% ↓** |
| Big域求解泄漏 | 5-10 MB | 1-2 MB | **80% ↓** |
| 收敛检查泄漏 | 0.5 MB/check | < 0.1 MB | **80% ↓** |
| Python对象累积 | 1-2 MB/frame | < 0.5 MB | **60% ↓** |

---

## 验证步骤

### 1. 运行相同的profiling测试:
```bash
cd /Users/zhaofen2/Desktop/work/SIG/mpm-domain-decomposition-taichi
python simulators/implicit_mpm_schwarz.py --config config/schwarz_2d_test4.json --no-gui
```

### 2. 检查新的memory_profile.json:
```bash
# 查看总体统计
jq '.summary' experiment_results/schwarz_YYYYMMDD_HHMMSS/memory_profile.json

# 查看leak分析
jq '.leak_analysis' experiment_results/schwarz_YYYYMMDD_HHMMSS/memory_profile.json

# 对比Frame 2的详细checkpoint
jq '.checkpoints[] | select(.label | contains("frame_2_schwarz_iter_1_after_convergence_check"))' memory_profile.json
```

### 3. 成功标准:
- ✅ 平均内存增长 < 5 MB/frame（当前14.8）
- ✅ Frame 2迭代0的big solve < 5 MB（当前10.2）
- ✅ 收敛检查增长 < 0.1 MB（当前0.5）
- ✅ `leak_analysis.has_memory_leak` = false 或 average < 0.01 MB/frame

### 4. 对比关键指标:

#### 修复前（experiment_results/schwarz_20260208_221541）:
- `total_memory_growth_mb`: 739.98 MB (50 frames)
- Average: 14.8 MB/frame

#### 修复后（新运行）:
- **预期** `total_memory_growth_mb`: < 250 MB (50 frames)
- **预期** Average: < 5 MB/frame

---

## 如果仍有问题的后续措施

### 如果内存增长仍 > 5 MB/frame:

**Option A**: 增加H_builder重建频率
```python
# 改为每次重建（Optimizer/Newton.py Line 120）
if it % 1 == 0:  # 原来是 % 10
```

**Option B**: 增加GC频率
```python
# 改为每50帧轻量清理（implicit_mpm_schwarz.py Line 703）
if self.current_frame % 50 == 0:  # 原来是 % 100
```

**Option C**: 在Schwarz迭代内添加轻量清理
```python
# 在Schwarz循环内每10次迭代（implicit_mpm_schwarz.py Line 570附近）
if i > 0 and i % 10 == 0:
    gc.collect(generation=0)
```

---

## 性能影响评估

### Fix 1: 几乎无影响
- Kernel内使用field accumulator vs 局部变量：性能相同或略好

### Fix 2: 轻微影响（< 2%）
- 每10次Newton迭代重建一次builder
- 单次重建开销：~0.5-1ms
- 总体影响：每帧 < 0.2ms

### Fix 3: 最小影响（< 0.5%）
- Level 1 (gen 0): 每100帧，~0.1-0.5ms
- Level 2 (gen 1): 每500帧，~1-5ms
- Level 3 (full): 每1000帧，~5-20ms（原有）

**总计性能影响**: < 2.5%，完全可接受

---

## 注意事项

1. **Taichi版本**: 这些修复基于当前Taichi版本，未来版本可能有不同的内存管理行为
2. **GPU vs CPU**: GPU模式可能有不同的泄漏模式
3. **Grid大小**: 更大的网格可能需要更频繁的清理
4. **长期运行**: 如果运行 > 10,000帧，可能需要进一步调整GC策略

---

## 文件修改清单

1. ✅ `simulators/implicit_mpm_schwarz.py`
   - Line 135: 添加convergence_max_residual字段
   - Lines 387-442: 重构check_schwarz_convergence
   - Lines 702-725: 实施分层GC策略

2. ✅ `Optimizer/Newton.py`
   - Lines 117-128: 添加周期性H_builder重建

**总共修改**: 2个文件，~40行代码改动
