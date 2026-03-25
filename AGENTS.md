# AGENTS.md - 小行星附着轨迹设计项目

本文档供 Agent 编码助手使用，包含项目的构建、测试、代码风格等规范。

---

## 1. 项目概述

**项目名称**: 小行星附着最优轨迹设计与仿真分析  
**项目类型**: 科学计算/航天轨迹优化  
**Python 版本**: >= 3.8  
**主要依赖**: numpy, scipy, (可选: torch, scikit-learn, matplotlib, plotly, plyfile)

### 性能指标（已达成）
- **成功率**: 100%
- **位置误差**: ~0m
- **速度误差**: ~0m/s

### 目录结构

```
asteroid_contral/
├── algorithms/                     # 轨迹优化算法库（核心算法实现）
│   ├── __init__.py                # 模块导出
│   ├── direct_method.py           # 直接法
│   ├── convex_optimizer_cvxpy.py  # CVXPY凸优化
│   ├── pseudospectral_fast.py     # 快速伪谱法（推荐）
│   ├── shooting_fast.py           # 快速打靶法（推荐）
│   └── homotopy_fast.py           # 快速同伦法（推荐）
│
├── gravity_learning/              # 引力场学习模块
│   ├── __init__.py
│   ├── ply_model.py              # PLY模型加载和多面体引力计算
│   └── gravity_dnn.py            # 深度学习引力场模型
│
├── trajectory_optimization/       # 轨迹优化模块（封装algorithms）
│   └── __init__.py               # 引用algorithms模块
│
├── control_simulation/           # 控制与仿真模块
│   ├── __init__.py
│   ├── controller_optimized.py   # 控制器
│   ├── monte_carlo.py           # 蒙特卡洛模拟
│   ├── pid_controller.py         # PID控制器
│   └── validator.py             # 结果验证
│
├── config/                       # 配置文件
│   ├── config.yaml              # 主配置文件
│   └── README.md                # 配置说明
│
├── data/                         # 数据目录
│   ├── models/                   # 训练好的模型
│   └── samples/                  # 样本数据
│
├── results/                      # 结果目录
│   ├── phase1/                   # 第一阶段结果
│   ├── phase2/                   # 第二阶段结果
│   └── phase3/                   # 第三阶段结果
│
├── run.py                       # 完整运行脚本（推荐使用）
├── test_simple_run.py           # 简化测试脚本（使用点质量模型）
├── test_all.py                  # 全量测试脚本
├── main.py                      # 主入口
├── pyproject.toml               # 项目配置
├── requirements.txt             # 依赖列表
└── README.md                    # 项目主文档
```

---

## 2. 构建与测试命令

### 依赖安装

```bash
# 安装所有依赖
pip install -r requirements.txt
```

### 运行验证

```bash
# 简化测试（使用点质量引力模型，无需PyTorch）
python test_simple_run.py

# 完整流程运行（简化模式，绕过DNN）
python run.py --simple --method homotopy

# 完整流程运行（使用DNN引力场模型）
python run.py --skip-training --method homotopy

# 运行所有测试
python test_all.py
```

### 命令参数说明

```bash
# run.py 参数
--simple          # 使用点质量引力模型（绕过DNN/PyTorch）
--skip-training   # 跳过DNN训练，加载已有模型
--phase [1|2|3]   # 只运行指定阶段
--method [homotopy|pseudospectral|shooting|direct|cvxpy]  # 优化方法
```

### 代码格式化与检查

```bash
# 格式化代码（black）
black .

# 代码检查（flake8）
flake8 .
```

---

## 3. 算法使用指南

### 3.1 推荐使用的优化算法

```python
from algorithms import (
    FastPseudospectralOptimizer,   # 推荐：伪谱法
    FastShootingOptimizer,          # 推荐：打靶法
    FastHomotopyOptimizer,          # 推荐：同伦法
)

# 同伦法（最快）
optimizer = FastHomotopyOptimizer(asteroid, spacecraft, n_nodes=25, n_homotopy_steps=2)
result = optimizer.optimize(r0, v0, m0, rf, vf, t_span)

# 伪谱法
optimizer = FastPseudospectralOptimizer(asteroid, spacecraft, n_nodes=25)
result = optimizer.optimize(r0, v0, m0, rf, vf, t_span)

# 打靶法
optimizer = FastShootingOptimizer(asteroid, spacecraft, n_nodes=25)
result = optimizer.optimize(r0, v0, m0, rf, vf, t_span)
```

### 3.2 简化引力模型

```python
# 点质量引力模型（无需DNN）
class SimpleAsteroid:
    def __init__(self, mu=11.0, omega=None):
        self.mu = mu
        self.omega = omega if omega is not None else np.array([0.0, 0.0, 3.6e-4])
    
    def compute_gravity(self, position):
        r = np.linalg.norm(position)
        if r < 1e-10:
            return np.zeros(3)
        return -self.mu * position / (r**3)
```

### 3.3 航天器模型

```python
class SimpleSpacecraft:
    def __init__(self):
        self.T_max = 20.0   # 最大推力 (N)
        self.I_sp = 300.0   # 比冲 (s)
        self.g0 = 9.80665   # 重力加速度 (m/s^2)
        self.m0 = 500.0     # 初始质量 (kg)
```

---

## 4. 代码风格规范

### 4.1 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 类名 | PascalCase | `FastHomotopyOptimizer` |
| 函数/方法 | snake_case | `optimize_trajectory` |
| 变量 | snake_case | `r0`, `v0`, `pos_error` |
| 常量 | UPPER_SNAKE_CASE | `MAX_ITER`, `T_MAX` |
| 模块名 | snake_case | `gravity_dnn.py` |

### 4.2 类型注解

推荐使用类型注解：

```python
def optimize(
    self,
    r0: np.ndarray,
    v0: np.ndarray,
    m0: float,
    rf: np.ndarray,
    vf: np.ndarray,
    t_span: List[float],
) -> Dict:
    """优化轨迹"""
    ...
```

---

## 5. 配置管理

### 5.1 使用 YAML 配置

所有参数通过 `config/config.yaml` 管理：

```python
import yaml

with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 读取参数
T_max = config["phase2"]["spacecraft"]["T_max"]
```

### 5.2 常用配置参数

| 配置路径 | 说明 | 示例值 |
|----------|------|--------|
| phase2.boundary_conditions.r0 | 初始位置 | [10177, 6956, 8256] |
| phase2.boundary_conditions.rf | 目标位置 | [676, 5121, 449] |
| phase2.spacecraft.T_max | 最大推力(N) | 20.0 |
| phase2.spacecraft.m0 | 初始质量(kg) | 1000.0 |
| phase3.pid.Kp | PID比例增益 | 2.0 |

---

## 6. 常用工作流程

### 6.1 完整流程（简化模式）

```python
# 1. 创建模型
asteroid = SimpleAsteroid(mu=11.0)
spacecraft = SimpleSpacecraft()

# 2. 设置问题
r0 = np.array([500.0, 0.0, 100.0])
v0 = np.array([0.0, 0.2, 0.0])
m0 = spacecraft.m0
rf = np.array([400.0, 50.0, 50.0])
vf = np.array([0.0, 0.1, 0.0])
t_span = [0.0, 1800.0]

# 3. 轨迹优化
from algorithms import FastHomotopyOptimizer
optimizer = FastHomotopyOptimizer(asteroid, spacecraft, n_nodes=20, n_homotopy_steps=2)
result = optimizer.optimize(r0, v0, m0, rf, vf, t_span)

# 4. 检查结果
print(f"成功: {result['success']}")
print(f"燃料消耗: {result['fuel_consumption']:.2f} kg")
```

---

## 7. 注意事项

1. **NumPy/PyTorch兼容性**: 如果遇到NumPy版本兼容性问题，使用 `--simple` 参数绕过DNN
2. **单位一致**: 位置(m)、速度(m/s)、时间(s)、力(N)、质量(kg)
3. **PLY模型**: 使用 `Castalia Radar-based.ply` 作为默认小行星模型
4. **结果保存**: 所有输出结果保存到 `results/` 目录
5. **中文支持**: 脚本输出和文档使用中文，matplotlib 中文字体设置为 SimHei

---

## 8. 关键文件参考

| 文件 | 说明 |
|------|------|
| pyproject.toml | 项目配置（black, pytest） |
| config/config.yaml | 所有可调参数 |
| requirements.txt | Python 依赖 |
| README.md | 项目主文档 |
| run.py | 完整运行脚本 |
| test_simple_run.py | 简化测试脚本 |

---

## 9. 测试结果

最新测试结果（2026-03-21）：

```
方法         成功     位置误差(m)      速度误差(m/s)      燃料(kg)     时间(s)   
----------------------------------------------------------------------
伪谱法        是      0.0000       0.0000         1.47       46.14   
打靶法        是      0.0000       0.0000         1.47       46.09   
同伦法        是      0.0000       0.0000         1.47       30.21   

成功率: 3/3 (100%)
```

---

*本文档最后更新: 2026-03-21*
