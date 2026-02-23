# 小行星附着最优轨迹设计 - 算法库

本目录包含小行星附着任务的轨迹优化算法实现。

## 目录结构

```
algorithms/
├── __init__.py              # 包初始化，导出所有优化器
├── shooting_method.py       # 打靶法
├── multiple_shooting.py     # 多重打靶法
├── pseudospectral.py        # 伪谱法
├── homotopy.py              # 同伦法
└── direct_method.py         # 直接法和凸优化
```

## 算法简介

### 1. 打靶法 (Shooting Method)
**文件**: `shooting_method.py`

基于最优控制理论的间接法，求解两点边值问题(TPBVP)。

**特点**:
- 使用Pontryagin极大值原理
- Bang-bang控制（开关控制）
- 需要求解协态方程
- 适合燃料最优问题

**核心类**: `ShootingMethodOptimizer`

### 2. 多重打靶法 (Multiple Shooting)
**文件**: `multiple_shooting.py`

将轨迹分为多段，分别求解后通过连续性约束连接。

**特点**:
- 分段积分，更好的数值稳定性
- 并行化潜力
- 适合长轨迹和复杂动力学
- 收敛性优于单段打靶法

**核心类**: `MultipleShootingOptimizer`

### 3. 伪谱法 (Pseudospectral Method)
**文件**: `pseudospectral.py`

使用Legendre-Gauss-Lobatto (LGL) 配点的全局优化方法。

**特点**:
- 高精度全局优化
- 切比雪夫节点和微分矩阵
- 状态归一化处理
- 适合高精度轨迹规划

**核心类**: `PseudospectralOptimizer`

### 4. 同伦法 (Homotopy Method)
**文件**: `homotopy.py`

从能量最优平滑过渡到燃料最优。

**特点**:
- 同伦参数 ζ ∈ [0, 1]
- ζ=0: 能量最优（连续推力）
- ζ=1: 燃料最优（bang-bang控制）
- 路径跟踪算法
- 解决燃料最优的收敛性问题

**核心类**: `HomotopyOptimizer`

### 5. 直接法 (Direct Method)
**文件**: `direct_method.py`

直接离散化最优控制问题为NLP问题。

**特点**:
- 直接优化状态和控制
- 无需计算协态
- 易于处理复杂约束
- 适合实时规划

**核心类**: `DirectMethodOptimizer`, `ConvexOptimizer`

## 使用方法

### 基本使用

```python
from algorithms import ShootingMethodOptimizer, PseudospectralOptimizer

# 初始化优化器
optimizer = ShootingMethodOptimizer(asteroid, spacecraft)

# 执行优化
result = optimizer.optimize_with_multiple_guesses(
    r0, v0, m0,  # 初始状态
    rf, vf,      # 终端状态
    t_span       # 时间区间
)

# 提取结果
trajectory_data = optimizer.extract_trajectory_data(result['trajectory'])
optimizer.plot_results(trajectory_data, r0, rf)
```

### 算法组合使用

```python
# 1. 先用伪谱法求解能量最优
ps_optimizer = PseudospectralOptimizer(asteroid, spacecraft)
ps_result = ps_optimizer.optimize(r0, v0, m0, rf, vf, t_span)

# 2. 再用同伦法过渡到燃料最优
homotopy_optimizer = HomotopyOptimizer(asteroid, spacecraft)
init_lam_r = ps_result['lam_r'][0]
init_lam_v = ps_result['lam_v'][0]
init_lam_m = ps_result['lam_m'][0]

result = homotopy_optimizer.solve_homotopy(
    r0, v0, m0, rf, vf, t_span,
    init_lam_r, init_lam_v, init_lam_m
)
```

## 依赖要求

```
numpy
scipy
matplotlib
```

## 输入要求

### 小行星对象 (asteroid)
```python
class Asteroid:
    omega: np.ndarray          # 自旋角速度 (3,)
    mu: float                  # 引力常数
    
    def gravity_gradient(self, r: np.ndarray) -> np.ndarray:
        """计算引力加速度"""
        pass
    
    def gravity_hessian(self, r: np.ndarray) -> np.ndarray:
        """计算引力梯度矩阵 (3,3)"""
        pass
```

### 航天器对象 (spacecraft)
```python
class Spacecraft:
    T_max: float               # 最大推力 (N)
    I_sp: float                # 比冲 (s)
    g0: float                  # 地球重力加速度 (m/s²)
    m0: float                  # 初始质量 (kg)
```

## 各算法适用场景

| 算法 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 打靶法 | 燃料最优 | 精度高 | 收敛难 |
| 多重打靶法 | 长轨迹 | 稳定性好 | 计算量大 |
| 伪谱法 | 高精度 | 全局最优 | 计算复杂 |
| 同伦法 | 燃料最优 | 收敛性好 | 需要初值 |
| 直接法 | 实时规划 | 易于实现 | 精度较低 |

## 版本历史

- **v1.0.0** (2025-02-13)
  - 初始版本
  - 实现5种轨迹优化算法
  - 统一接口设计
  - 完整的文档和示例

## 参考

- Betts, J.T. (2010). Practical Methods for Optimal Control and Estimation Using Nonlinear Programming.
- Conway, B.A. (2012). A Survey of Methods Available for the Numerical Optimization of Continuous Dynamic Systems.
