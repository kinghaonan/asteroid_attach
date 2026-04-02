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

# Keep imports optional so this package can be imported without all legacy modules.
_exports = {}

def _try_import(name: str):
    try:
        module = __import__(name, fromlist=["*"])
        return module
    except Exception:
        return None

# Try legacy modules (may not exist in this repo)
_shooting = _try_import("shooting_method")
_pseudospectral = _try_import("pseudospectral")
_scp = _try_import("convex_optimizer_scp")
_direct = _try_import("direct_method")
_shooting_opt = _try_import("shooting_method_optimized")
_pseudospectral_opt = _try_import("pseudospectral_optimized")
_scp_opt = _try_import("convex_optimizer_scp_optimized")

if _shooting and hasattr(_shooting, "ShootingMethodOptimizer"):
    _exports["ShootingMethodOptimizer"] = _shooting.ShootingMethodOptimizer
if _pseudospectral and hasattr(_pseudospectral, "PseudospectralOptimizer"):
    _exports["PseudospectralOptimizer"] = _pseudospectral.PseudospectralOptimizer
if _scp and hasattr(_scp, "SCPOptimizer"):
    _exports["SCPOptimizer"] = _scp.SCPOptimizer
if _direct and hasattr(_direct, "DirectMethodOptimizer"):
    _exports["DirectMethodOptimizer"] = _direct.DirectMethodOptimizer
if _direct and hasattr(_direct, "ConvexOptimizer"):
    _exports["ConvexOptimizer"] = _direct.ConvexOptimizer

if _shooting_opt and hasattr(_shooting_opt, "OptimizedShootingMethodOptimizer"):
    _exports["OptimizedShootingMethodOptimizer"] = _shooting_opt.OptimizedShootingMethodOptimizer
if _pseudospectral_opt and hasattr(_pseudospectral_opt, "OptimizedPseudospectralOptimizer"):
    _exports["OptimizedPseudospectralOptimizer"] = _pseudospectral_opt.OptimizedPseudospectralOptimizer
if _scp_opt and hasattr(_scp_opt, "OptimizedSCPOptimizer"):
    _exports["OptimizedSCPOptimizer"] = _scp_opt.OptimizedSCPOptimizer

# Back-compat aliases if optimized versions exist
if "OptimizedShootingMethodOptimizer" in _exports:
    _exports["ShootingMethodOptimizerOptimized"] = _exports["OptimizedShootingMethodOptimizer"]
if "OptimizedPseudospectralOptimizer" in _exports:
    _exports["PseudospectralOptimizerOptimized"] = _exports["OptimizedPseudospectralOptimizer"]
if "OptimizedSCPOptimizer" in _exports:
    _exports["SCPOptimizerOptimized"] = _exports["OptimizedSCPOptimizer"]

globals().update(_exports)
__all__ = sorted(_exports.keys())
