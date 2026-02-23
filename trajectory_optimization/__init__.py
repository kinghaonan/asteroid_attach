#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹优化模块 - 第二部分

包含多种轨迹优化算法：
- 打靶法
- 多重打靶法
- 伪谱法
- 同伦法
- 直接法/凸优化
"""

import sys
import os

# 将algorithms目录添加到路径
algorithms_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "algorithms")
if algorithms_path not in sys.path:
    sys.path.insert(0, algorithms_path)

# 从algorithms导入所有优化器
from shooting_method import ShootingMethodOptimizer
from multiple_shooting import MultipleShootingOptimizer
from pseudospectral import PseudospectralOptimizer
from homotopy import HomotopyOptimizer
from direct_method import DirectMethodOptimizer, ConvexOptimizer

__all__ = [
    "ShootingMethodOptimizer",
    "MultipleShootingOptimizer",
    "PseudospectralOptimizer",
    "HomotopyOptimizer",
    "DirectMethodOptimizer",
    "ConvexOptimizer",
]
