#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹优化算法库

成功的方法：
1. 直接法 (DirectMethodOptimizer)
2. CVXPY凸优化 (CVXPYSCPOptimizer) 
3. 快速伪谱法 (FastPseudospectralOptimizer)
4. 快速打靶法 (FastShootingOptimizer)
5. 快速同伦法 (FastHomotopyOptimizer) - 推荐

性能对比：
- 燃料最优：快速伪谱法/打靶法/同伦法 (30.85kg)
- 速度最快：直接法 (~5s)
- 平衡推荐：CVXPY凸优化 或 快速同伦法
"""

__version__ = "4.0.0"

# 直接法
from .direct_method import DirectMethodOptimizer

# CVXPY凸优化
from .convex_optimizer_cvxpy import CVXPYSCPOptimizer

# 快速优化器（基于bang-bang控制和同伦策略）
from .pseudospectral_fast import FastPseudospectralOptimizer
from .shooting_fast import FastShootingOptimizer
from .homotopy_fast import FastHomotopyOptimizer

__all__ = [
    "DirectMethodOptimizer",
    "CVXPYSCPOptimizer",
    "FastPseudospectralOptimizer",
    "FastShootingOptimizer",
    "FastHomotopyOptimizer",
]
