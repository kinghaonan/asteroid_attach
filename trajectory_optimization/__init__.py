#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹优化模块

包含多种核心轨迹优化算法：
- 打靶法
- 伪谱法
- 凸优化(SCP) - 推荐使用
- 直接法

推荐使用优化版本算法：
- OptimizedShootingMethodOptimizer
- OptimizedPseudospectralOptimizer
- OptimizedSCPOptimizer
"""

import sys
import os

algorithms_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "algorithms")
if algorithms_path not in sys.path:
    sys.path.insert(0, algorithms_path)

# 原版算法
from shooting_method import ShootingMethodOptimizer
from pseudospectral import PseudospectralOptimizer
from convex_optimizer_scp import SCPOptimizer
from direct_method import DirectMethodOptimizer, ConvexOptimizer

# 优化版算法
from shooting_method_optimized import OptimizedShootingMethodOptimizer
from pseudospectral_optimized import OptimizedPseudospectralOptimizer
from convex_optimizer_scp_optimized import OptimizedSCPOptimizer

# 别名（向后兼容）
ShootingMethodOptimizerOptimized = OptimizedShootingMethodOptimizer
PseudospectralOptimizerOptimized = OptimizedPseudospectralOptimizer
SCPOptimizerOptimized = OptimizedSCPOptimizer

__all__ = [
    # 原版
    "ShootingMethodOptimizer",
    "PseudospectralOptimizer",
    "SCPOptimizer",
    "DirectMethodOptimizer",
    "ConvexOptimizer",
    # 优化版
    "OptimizedShootingMethodOptimizer",
    "OptimizedPseudospectralOptimizer",
    "OptimizedSCPOptimizer",
    # 别名
    "ShootingMethodOptimizerOptimized",
    "PseudospectralOptimizerOptimized",
    "SCPOptimizerOptimized",
]