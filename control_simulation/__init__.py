#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
控制与仿真模块

包含：
1. PID 控制器（原版）
2. 自适应 PID 控制器
3. 优化版控制器（推荐）
4. 轨迹跟踪控制器
5. 蒙特卡洛模拟
6. 结果验证
"""

# 原版控制器
from .pid_controller import PIDController, AdaptivePIDController
from .monte_carlo import MonteCarloSimulator
from .validator import ResultValidator, evaluate_project

# 优化版控制器
from .controller_optimized import (
    OptimizedPIDController,
    FeedforwardController,
    TrajectoryTracker,
    AdaptiveController,
)

__all__ = [
    # 原版
    "PIDController",
    "AdaptivePIDController",
    "MonteCarloSimulator",
    "ResultValidator",
    "evaluate_project",
    # 优化版
    "OptimizedPIDController",
    "FeedforwardController",
    "TrajectoryTracker",
    "AdaptiveController",
]