#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 算法库

提供多种轨迹优化算法：

基础算法:
1. 打靶法 (Shooting Method) - shooting_method.py
2. 多重打靶法 (Multiple Shooting) - multiple_shooting.py
3. 伪谱法 (Pseudospectral) - pseudospectral.py
4. 同伦法 (Homotopy) - homotopy.py
5. 直接法 (Direct Method) - direct_method.py

优化版本:
6. 打靶法优化版 - shooting_method_optimized.py
7. 伪谱法优化版 - pseudospectral_optimized.py

高级算法:
8. 混合优化法 (Hybrid) - hybrid_optimization.py
9. 自适应同伦法 (Adaptive Homotopy) - adaptive_homotopy.py

工具模块:
10. 约束处理 - constraint_handler.py
11. 结果对比 - trajectory_comparator.py
12. 统一接口 - unified_optimizer.py

使用方法:
    from algorithms import (
        ShootingMethodOptimizer,
        PseudospectralOptimizer,
        HybridTrajectoryOptimizer,
        UnifiedTrajectoryOptimizer,
    )
"""

__version__ = "2.0.0"
__author__ = "Asteroid Mission Team"

# 基础算法
from .shooting_method import ShootingMethodOptimizer
from .multiple_shooting import MultipleShootingOptimizer
from .pseudospectral import PseudospectralOptimizer
from .homotopy import HomotopyOptimizer
from .direct_method import DirectMethodOptimizer, ConvexOptimizer

# 优化版本
from .shooting_method_optimized import ShootingMethodOptimizerOptimized
from .pseudospectral_optimized import PseudospectralOptimizerOptimized

# 凸优化（SOCP）- 可选依赖
try:
    from .convex_optimizer_socp import SOCPTrajectoryOptimizer

    SOCP_AVAILABLE = True
except ImportError:
    SOCPTrajectoryOptimizer = None
    SOCP_AVAILABLE = False

# 高级算法
from .hybrid_optimization import HybridTrajectoryOptimizer
from .adaptive_homotopy import AdaptiveHomotopyOptimizer

# 工具模块
from .constraint_handler import (
    ConstraintManager,
    ThrustConstraints,
    PathConstraints,
    StateConstraints,
    TerminalConstraints,
)
from .trajectory_comparator import TrajectoryComparator, TrajectoryMetrics
from .unified_optimizer import (
    UnifiedTrajectoryOptimizer,
    TrajectoryEvaluator,
    OptimizationMethod,
    OptimizationConfig,
    OptimizationResult,
)

__all__ = [
    # 基础算法
    "ShootingMethodOptimizer",
    "MultipleShootingOptimizer",
    "PseudospectralOptimizer",
    "HomotopyOptimizer",
    "DirectMethodOptimizer",
    "ConvexOptimizer",
    # 优化版本
    "ShootingMethodOptimizerOptimized",
    "PseudospectralOptimizerOptimized",
    # 凸优化（SOCP）- 可选
    "SOCPTrajectoryOptimizer",
    "SOCP_AVAILABLE",
    # 高级算法
    "HybridTrajectoryOptimizer",
    "AdaptiveHomotopyOptimizer",
    # 工具模块
    "ConstraintManager",
    "ThrustConstraints",
    "PathConstraints",
    "StateConstraints",
    "TerminalConstraints",
    "TrajectoryComparator",
    "TrajectoryMetrics",
    "UnifiedTrajectoryOptimizer",
    "TrajectoryEvaluator",
    "OptimizationMethod",
    "OptimizationConfig",
    "OptimizationResult",
]
