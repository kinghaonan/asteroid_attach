# 项目重组迁移指南

## 概述

本项目已完成结构重组，从混乱的单一目录结构转变为清晰的模块化结构。

## 新目录结构

```
asteroid_mission/
├── src/asteroid_mission/              # 主源代码包
│   ├── __init__.py
│   ├── gravity_modeling/              # 引力场建模模块
│   │   ├── __init__.py
│   │   ├── ply_loader.py              # (原ply模型训练的部分功能)
│   │   └── gravity_dnn.py             # 神经网络模型
│   ├── trajectory/                    # 轨迹优化模块
│   │   ├── __init__.py
│   │   ├── base.py                    # Asteroid, Spacecraft 类
│   │   ├── shooting.py                # 打靶法
│   │   ├── multiple_shooting.py       # 多重打靶法
│   │   ├── pseudospectral.py          # 伪谱法
│   │   └── homotopy.py                # 同伦法
│   ├── control/                       # 控制律模块
│   │   ├── __init__.py
│   │   └── direct_method.py           # 直接法
│   ├── simulation/                    # 仿真验证模块
│   │   ├── __init__.py
│   │   └── validator.py               # 结果验证
│   └── visualization/                 # 可视化模块
│       ├── __init__.py
│       └── dnn_architecture.py        # DNN架构图
├── data/models/                       # 预训练模型文件
├── scripts/                           # 运行脚本
├── docs/                             # 文档
├── results/                          # 结果输出
└── tests/                            # 单元测试
```

## 文件迁移对照表

| 原文件 | 新位置 | 说明 |
|--------|--------|------|
| ply模型训练.py | src/asteroid_mission/gravity_modeling/ | 拆分为多个模块 |
| DNN类架构图.py | src/asteroid_mission/visualization/dnn_architecture.py | 重命名 |
| 伪谱法.py | src/asteroid_mission/trajectory/pseudospectral.py | 重命名 |
| 同伦法.py | src/asteroid_mission/trajectory/homotopy.py | 重命名 |
| 打靶法改.py | src/asteroid_mission/trajectory/shooting.py | 重命名 |
| 多重打靶法改.py | src/asteroid_mission/trajectory/multiple_shooting.py | 重命名 |
| 简化轨迹约束.py | src/asteroid_mission/control/direct_method.py | 重命名 |
| validate_results.py | src/asteroid_mission/simulation/validator.py | 重命名 |

## 代码使用方式变更

### 旧方式（不再推荐）

```python
# 直接导入根目录的文件
from ply模型训练 import GravityDNN
from 伪谱法 import Asteroid, Spacecraft
```

### 新方式（推荐）

```python
# 添加src到Python路径
import sys
sys.path.insert(0, 'src')

# 从新模块导入
from asteroid_mission.gravity_modeling import GravityDNN, PLYAsteroidModel
from asteroid_mission.trajectory import Asteroid, Spacecraft, ShootingOptimizer
from asteroid_mission.visualization import plot_dnn_architecture
```

## 主要改进

1. **模块化组织**: 按功能划分为5个清晰模块
2. **消除代码重复**: 统一了Asteroid和Spacecraft类定义
3. **标准命名**: 使用英文命名，符合Python规范
4. **完整配置**: 添加了pyproject.toml, requirements.txt等
5. **文档完善**: 添加了README和详细的模块文档

## 后续工作

### 需要完成的内容

1. **代码功能迁移**: 将原文件中的完整功能迁移到新模块
2. **单元测试**: 为各模块添加单元测试
3. **集成测试**: 确保模块间协同工作
4. **文档补充**: 添加更多使用示例和API文档
5. **旧文件清理**: 确认新结构可用后，可清理根目录旧文件

### 注意事项

- 当前新模块中的代码为简化版本，保留了核心接口
- 完整功能需要从原文件中提取并整合
- 建议逐步迁移和测试，确保功能完整性

## 如何运行

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
python scripts/example_trajectory.py
```

### 安装为包

```bash
pip install -e .
```

然后可以直接导入：

```python
from asteroid_mission.gravity_modeling import GravityDNN
```
