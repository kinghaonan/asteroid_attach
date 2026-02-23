# 小行星附着最优轨迹设计项目

小行星探测任务附着段轨迹优化设计与仿真分析。

## 项目概述

本项目针对小行星探测任务中的附着段，研究基于多面体引力场模型和深度学习的智能轨迹优化方法。

## 项目结构

```
asteroid_contral/
├── algorithms/                     # 轨迹优化算法库
│   ├── shooting_method.py         # 打靶法
│   ├── multiple_shooting.py       # 多重打靶法
│   ├── pseudospectral.py          # 伪谱法
│   ├── homotopy.py                # 同伦法
│   ├── direct_method.py           # 直接法/凸优化
│   └── README.md                  # 算法说明文档
│
├── gravity_learning/              # 第一部分：引力场学习
│   ├── __init__.py
│   ├── ply_model.py               # PLY模型加载和多面体引力计算
│   └── gravity_dnn.py             # 深度学习引力场模型
│
├── trajectory_optimization/        # 第二部分：轨迹优化
│   └── __init__.py                # 引用algorithms模块
│
├── control_simulation/             # 第三部分：控制与仿真
│   ├── __init__.py
│   ├── pid_controller.py          # PID控制器
│   ├── monte_carlo.py             # 蒙特卡洛模拟
│   └── validator.py               # 结果验证
│
├── tests/                          # 测试与验证
│   └── 分阶段验证指南.md          # 详细验证说明
│
├── data/                           # 数据目录
│   ├── models/                     # 训练好的模型
│   └── samples/                    # 样本数据
│
├── results/                        # 结果目录
│   ├── phase1/                     # 第一阶段结果
│   ├── phase2/                     # 第二阶段结果
│   └── phase3/                     # 第三阶段结果
│
├── config/                         # 配置文件
├── scripts/                        # 运行脚本
├── docs/                          # 文档目录
├── Castalia Radar-based.ply       # 小行星PLY模型文件
├── requirements.txt               # Python依赖
├── pyproject.toml                # 项目配置
└── README.md                      # 本文件
```

## 三大模块

### 1. 引力场学习 (gravity_learning/)

**功能**：小行星多面体建模和引力场深度学习

**核心类**：
- `PLYAsteroidModel` - PLY模型加载和预处理
- `PolyhedralGravitySampler` - 多面体引力计算
- `GravityAndGradientDNN` - 引力场神经网络
- `GravityGradientTrainer` - 模型训练器

**使用示例**：
```python
from gravity_learning import PLYAsteroidModel, PolyhedralGravitySampler
from gravity_learning import GravityAndGradientDNN, GravityGradientTrainer

# 加载PLY模型
model = PLYAsteroidModel("Castalia Radar-based.ply")

# 计算引力样本
sampler = PolyhedralGravitySampler(model)
points = sampler.generate_sampling_points(num_samples=1000)
gravity = sampler.calculate_polyhedral_gravity(points)

# 训练DNN
dnn = GravityAndGradientDNN()
trainer = GravityGradientTrainer(dnn)
```

### 2. 轨迹优化 (trajectory_optimization/)

**功能**：多种轨迹优化算法实现

**优化算法**：
- 打靶法 (Shooting Method)
- 多重打靶法 (Multiple Shooting)
- 伪谱法 (Pseudospectral)
- 同伦法 (Homotopy)
- 直接法 (Direct Method)

**使用示例**：
```python
from trajectory_optimization import ShootingMethodOptimizer, PseudospectralOptimizer

# 使用打靶法
optimizer = ShootingMethodOptimizer(asteroid, spacecraft)
result = optimizer.optimize_with_multiple_guesses(r0, v0, m0, rf, vf, t_span)

# 使用伪谱法
ps_opt = PseudospectralOptimizer(asteroid, spacecraft, n_nodes=30)
result = ps_opt.optimize(r0, v0, m0, rf, vf, t_span)
```

### 3. 控制与仿真 (control_simulation/)

**功能**：轨迹跟踪控制和蒙特卡洛仿真

**核心类**：
- `PIDController` - PID控制器
- `AdaptivePIDController` - 自适应PID
- `MonteCarloSimulator` - 蒙特卡洛模拟器
- `ResultValidator` - 结果验证器

**使用示例**：
```python
from control_simulation import PIDController, MonteCarloSimulator

# PID控制
pid = PIDController(Kp=1.0, Ki=0.1, Kd=0.5)
control = pid.compute_control(ref_pos, ref_vel, curr_pos, curr_vel, dt)

# 蒙特卡洛模拟
mc = MonteCarloSimulator(optimizer, asteroid, spacecraft, n_simulations=100)
stats = mc.run_monte_carlo(r0, v0, m0, rf, vf, t_span)
```

## 分阶段验证

详细验证步骤参见 `tests/分阶段验证指南.md`

### 快速验证

```bash
# 验证模块导入
python -c "from gravity_learning import *; print('✓ 引力场模块')"
python -c "from trajectory_optimization import *; print('✓ 轨迹优化模块')"
python -c "from control_simulation import *; print('✓ 控制仿真模块')"
```

## 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖：
- numpy
- scipy
- matplotlib
- torch
- sklearn
- plyfile

## 使用流程

1. **第一阶段**：训练引力场模型
   - 加载PLY模型
   - 生成引力样本
   - 训练DNN模型

2. **第二阶段**：优化轨迹
   - 选择优化算法
   - 设置边界条件
   - 执行优化

3. **第三阶段**：仿真验证
   - PID轨迹跟踪
   - 蒙特卡洛分析
   - 结果评估

## 开发团队

- 作者: Manus AI
- 项目类型: 科研计算项目

## 许可证

本项目用于学术研究目的。
