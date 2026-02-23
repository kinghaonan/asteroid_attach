# 配置文件说明

项目使用YAML配置文件管理所有参数，无需修改代码即可调整各阶段的设置。

## 配置文件位置

`config/config.yaml`

## 配置文件结构

```yaml
phase1:    # 第一阶段：引力场学习
phase2:    # 第二阶段：轨迹优化
phase3:    # 第三阶段：控制与仿真
global:    # 全局设置
```

---

## 第一阶段：引力场学习 (phase1)

### PLY模型设置
```yaml
ply_file: "Castalia Radar-based.ply"    # PLY模型文件路径
target_diameter_m: 1400.0                # 目标直径（米）
asteroid_density: 2670                   # 小行星密度（kg/m³）
```

### 样本生成设置
```yaml
sampling:
  num_samples: 500              # 样本数量
  min_r_ratio: 1.1              # 最小采样半径倍数
  max_r_ratio: 5.0              # 最大采样半径倍数
  reuse_existing: true          # 是否复用已存在的样本
  sample_file: "data/samples/gravity_samples.npz"  # 样本文件路径
```

**重要**：设置 `reuse_existing: true` 可以在重新训练时复用已计算的样本，大幅节省计算时间。

### DNN模型架构
```yaml
model:
  input_dim: 3                  # 输入维度（位置xyz）
  gravity_dim: 3                # 引力输出维度
  gradient_dim: 9               # 梯度输出维度
  shared_features: [256, 512, 512, 512, 256]  # 共享层结构
  head_features: [128]          # 输出头层结构
```

### 训练参数
```yaml
training:
  epochs: 20                    # 训练轮次
  batch_size: 64                # 批次大小
  learning_rate: 0.0001         # 学习率
  lr_scheduler_step: 20         # 学习率调整步长
  lr_scheduler_gamma: 0.5       # 学习率调整系数
  gravity_weight: 1.0           # 引力损失权重
  gradient_weight: 0.5          # 梯度损失权重
  print_freq: 5                 # 打印频率
```

**调整建议**：
- 如果训练不收敛，尝试减小 `learning_rate` 或增加 `epochs`
- 如果内存不足，减小 `batch_size`
- 如果过拟合，减小 `epochs` 或调整权重

---

## 第二阶段：轨迹优化 (phase2)

### 航天器参数
```yaml
spacecraft:
  T_max: 20.0                   # 最大推力（N）
  I_sp: 400.0                   # 比冲（s）
  g0: 9.81                      # 地球重力加速度（m/s²）
  m0: 1000.0                    # 初始质量（kg）
```

### 边界条件
```yaml
boundary_conditions:
  r0: [10177, 6956, 8256]       # 初始位置（m）
  v0: [-25, -12, -17]           # 初始速度（m/s）
  rf: [676, 5121, 449]          # 目标位置（m）
  vf: [0, 0, 0]                 # 目标速度（m/s）
  t_span: [0, 770]              # 时间区间（s）
```

### 优化算法参数

#### 伪谱法
```yaml
pseudospectral:
  enabled: true                 # 是否启用
  n_nodes: 20                   # 节点数
  max_iter: 100                 # 最大迭代次数
```

**调整建议**：
- 增加 `n_nodes` 提高精度，但计算时间增加
- 增加 `max_iter` 提高收敛概率

#### 打靶法
```yaml
shooting:
  enabled: true                 # 是否启用
  n_guesses: 4                  # 初始猜测数量
```

#### 同伦法
```yaml
homotopy:
  enabled: true                 # 是否启用
  n_steps: 5                    # 同伦步数
```

---

## 第三阶段：控制与仿真 (phase3)

### PID控制器参数
```yaml
pid:
  Kp: 2.0                       # 比例增益
  Ki: 0.2                       # 积分增益
  Kd: 1.0                       # 微分增益
  integral_limit: 100.0         # 积分限幅
```

**PID调参建议**：
- `Kp` 增大：响应加快，但可能超调
- `Ki` 增大：消除稳态误差，但可能引起振荡
- `Kd` 增大：抑制超调，但对噪声敏感

### 自适应PID（可选）
```yaml
adaptive_pid:
  enabled: false                # 是否使用自适应PID
  adaptation_rate: 0.01
  error_threshold: 1.0
```

### 轨迹跟踪仿真参数
```yaml
tracking:
  dt: 1.0                       # 时间步长（s）
  n_steps: 10                   # 模拟步数
  position_noise: 10.0          # 位置扰动（m）
  velocity_noise: 1.0           # 速度扰动（m/s）
```

### 蒙特卡洛模拟参数
```yaml
monte_carlo:
  enabled: true                 # 是否启用
  n_simulations: 20             # 模拟次数
  position_noise: 50.0          # 初始位置扰动（m）
  velocity_noise: 2.0           # 初始速度扰动（m/s）
  success_threshold: 70.0       # 成功率阈值（%）
```

**调整建议**：
- 正式验证时建议 `n_simulations: 100` 或更多
- 增加扰动测试系统鲁棒性

### 评估标准
```yaml
evaluation:
  max_position_error: 50.0      # 最大允许位置误差（m）
  max_velocity_error: 5.0       # 最大允许速度误差（m/s）
  min_success_rate: 70.0        # 最小成功率（%）
```

---

## 全局设置 (global)

### 随机种子
```yaml
global:
  random_seed: 42               # 随机种子（用于可重复性）
```

设置相同的随机种子可以确保结果可重复。

### 绘图设置
```yaml
  plotting:
    dpi: 150                    # 图像分辨率
    figure_size: [15, 10]       # 默认图像尺寸
    font_family: "SimHei"       # 中文字体
    save_format: "png"          # 图像格式
```

### 调试模式
```yaml
  debug: false                  # 是否打印调试信息
```

---

## 常用配置示例

### 快速验证配置（用于测试）
```yaml
phase1:
  sampling:
    num_samples: 100            # 少量样本
    reuse_existing: true        # 复用已有样本
  training:
    epochs: 5                   # 少量轮次

phase2:
  pseudospectral:
    n_nodes: 10                 # 减少节点
    max_iter: 50                # 减少迭代

phase3:
  monte_carlo:
    n_simulations: 10           # 少量模拟
```

### 高精度验证配置（用于正式验证）
```yaml
phase1:
  sampling:
    num_samples: 5000           # 大量样本
  training:
    epochs: 100                 # 多轮训练
    learning_rate: 0.00005      # 更小学习率

phase2:
  pseudospectral:
    n_nodes: 50                 # 增加节点
    max_iter: 500               # 增加迭代
  shooting:
    enabled: true
  homotopy:
    enabled: true
    n_steps: 20                 # 增加同伦步数

phase3:
  monte_carlo:
    n_simulations: 200          # 大量模拟
```

### 节省计算时间配置
```yaml
phase1:
  sampling:
    reuse_existing: true        # 复用样本（如果存在）
  training:
    epochs: 10                  # 适当减少轮次

phase2:
  pseudospectral:
    enabled: true
  shooting:
    enabled: false              # 禁用打靶法
  homotopy:
    enabled: false              # 禁用同伦法

phase3:
  monte_carlo:
    n_simulations: 20           # 减少模拟次数
```

---

## 使用建议

1. **首次运行**：使用默认配置，确保流程正确
2. **快速测试**：使用"快速验证配置"减少计算时间
3. **正式验证**：使用"高精度验证配置"获得准确结果
4. **调整参数**：根据结果调整配置，无需修改代码

---

## 注意事项

- 修改配置后重新运行相应阶段即可生效
- 各阶段的数据会自动衔接（通过文件路径）
- 建议保留一份默认配置作为备份
- 可以在不同配置文件之间切换，测试不同参数组合
