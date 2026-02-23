# PKL 样本文件使用指南

## 概述

现在支持从你的 `samples_with_gradient.pkl` 文件直接加载样本，无需重新计算，大幅节省时间！

## 配置文件

编辑 `config/config.yaml`：

```yaml
phase1:
  sampling:
    # pkl文件路径（优先使用）
    pkl_file: "data/samples/samples_with_gradient.pkl"
    
    # npz缓存路径（备用）
    npz_file: "data/samples/gravity_samples.npz"
    
    # 是否复用已存在的样本
    reuse_existing: true
```

## 加载优先级

代码会自动按以下优先级加载样本：

1. **pkl文件**（如果存在且 `reuse_existing: true`）
   - 直接从你的 `samples_with_gradient.pkl` 加载
   - 包含 positions, gravity, gradient
   - 显示元数据信息
   
2. **npz缓存**（如果pkl不存在但npz存在）
   - 从 `gravity_samples.npz` 加载
   - 作为pkl的快速缓存格式
   
3. **重新计算**（如果都不存在）
   - 使用PLY模型计算新的样本
   - 同时计算引力和梯度
   - 保存为npz格式供下次使用

## 使用方法

### 方式一：使用现有的 pkl 文件

确保你的 pkl 文件在正确位置：
```
data/samples/samples_with_gradient.pkl
```

然后直接运行：
```bash
python verify/verify_phase1.py
```

输出示例：
```
📂 正在从pkl文件加载样本: data/samples/samples_with_gradient.pkl
✅ pkl样本加载成功: 5000 个点
   包含引力: (5000, 3), 梯度: (5000, 9)
   元数据:
     ply_model_path: Castalia Radar-based.ply
     asteroid_radius: 700.0
     asteroid_density: 2670
     num_samples: 5000
✅ 从PKL加载样本: (5000, 3)
   引力范围: [-1.234567e-03, 1.234567e-03] m/s²
   梯度范围: [-5.678901e-07, 5.678901e-07] 1/s²
```

### 方式二：指定自定义 pkl 路径

修改配置文件中的路径：
```yaml
pkl_file: "path/to/your/custom_samples.pkl"
```

### 方式三：强制重新计算

如果想忽略现有的pkl文件，重新计算样本：

```yaml
reuse_existing: false
```

或者临时删除/重命名pkl文件。

## PKL 文件格式

你的 pkl 文件应该包含以下结构：

```python
{
    "positions": np.array,  # 形状: (N, 3) - 采样点位置
    "gravity": np.array,    # 形状: (N, 3) - 引力加速度
    "gradient": np.array,   # 形状: (N, 9) - 引力梯度（3x3矩阵展平）
    "metadata": {
        "ply_model_path": str,      # PLY模型路径
        "asteroid_radius": float,   # 小行星半径
        "asteroid_density": float,  # 密度
        "num_samples": int,         # 样本数量
        "sampling_time": str        # 采样时间
    }
}
```

## 性能对比

| 方式 | 5000样本耗时 | 说明 |
|------|-------------|------|
| 从pkl加载 | ~1-2秒 | 直接读取，无需计算 |
| 从npz加载 | ~1-2秒 | 二进制格式，快速读取 |
| 重新计算 | ~10-30分钟 | 需要遍历所有面元计算 |

**节省 99% 以上的时间！**

## 故障排除

### 问题：pkl文件不存在

```
⚠️ 从pkl加载失败: pkl文件不存在: data/samples/samples_with_gradient.pkl
   尝试其他方式...
```

**解决**：
- 检查文件路径是否正确
- 确保文件在 `data/samples/` 目录下
- 或在配置文件中指定正确的路径

### 问题：pkl文件损坏

```
⚠️ 从pkl加载失败: [错误信息]
   尝试其他方式...
```

**解决**：
- 代码会自动尝试从npz加载或重新计算
- 或者删除损坏的pkl文件，重新计算

### 问题：内存不足

如果pkl文件很大（>1GB），可能会内存不足。

**解决**：
- 减小 batch_size
- 或者分批次处理（需要修改代码）

## 最佳实践

1. **保留原始pkl文件**：不要删除你的 `samples_with_gradient.pkl`
2. **设置正确路径**：在配置文件中指定pkl文件的准确路径
3. **启用复用**：设置 `reuse_existing: true`
4. **自动缓存**：代码会自动创建npz缓存，加速后续加载

## 示例工作流

```bash
# 1. 确保pkl文件存在
ls data/samples/samples_with_gradient.pkl

# 2. 运行第一阶段（自动从pkl加载）
python verify/verify_phase1.py

# 3. 查看输出确认从pkl加载
# 应该看到："✅ 从PKL加载样本: (N, 3)"

# 4. 继续运行第二阶段
python verify/verify_phase2.py
```

享受快速加载吧！🚀
