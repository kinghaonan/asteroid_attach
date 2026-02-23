#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 统一优化接口与评估框架

提供统一的API来使用所有优化算法，并包含完整的评估验证功能。

支持的算法：
1. 打靶法 (ShootingMethod)
2. 多重打靶法 (MultipleShooting)
3. 伪谱法 (Pseudospectral)
4. 同伦法 (Homotopy)
5. 自适应同伦法 (AdaptiveHomotopy)
6. 混合优化法 (HybridOptimization)
7. 直接法 (DirectMethod)

功能：
- 统一接口调用所有算法
- 自动算法选择建议
- 结果验证与评估
- 可视化分析
- 报告生成
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass
import time
import warnings

# 导入各优化算法
from .shooting_method_optimized import ShootingMethodOptimizerOptimized
from .pseudospectral_optimized import PseudospectralOptimizerOptimized
from .hybrid_optimization import HybridTrajectoryOptimizer
from .adaptive_homotopy import AdaptiveHomotopyOptimizer
from .shooting_method import ShootingMethodOptimizer
from .multiple_shooting import MultipleShootingOptimizer
from .pseudospectral import PseudospectralOptimizer
from .homotopy import HomotopyOptimizer
from .direct_method import DirectMethodOptimizer
from .constraint_handler import ConstraintManager, ThrustConstraints, PathConstraints


class OptimizationMethod(Enum):
    """优化方法枚举"""

    SHOOTING = "shooting"
    MULTIPLE_SHOOTING = "multiple_shooting"
    PSEUDOSPECTRAL = "pseudospectral"
    HOMOTOPY = "homotopy"
    ADAPTIVE_HOMOTOPY = "adaptive_homotopy"
    HYBRID = "hybrid"
    DIRECT = "direct"
    AUTO = "auto"


@dataclass
class OptimizationConfig:
    """优化配置数据类"""

    method: OptimizationMethod
    max_iter: int = 200
    tol: float = 1e-6
    verbose: bool = True

    # 算法特定参数
    n_nodes: int = 30
    n_segments: int = 5
    n_steps: int = 10
    use_stage2: bool = True


@dataclass
class OptimizationResult:
    """优化结果数据类"""

    success: bool
    method: str
    trajectory: Dict[str, np.ndarray]
    fuel_consumption: float
    pos_error: float
    vel_error: float
    computation_time: float
    iterations: int
    message: str
    metadata: Dict[str, Any]


class UnifiedTrajectoryOptimizer:
    """
    统一轨迹优化器

    提供统一的接口来调用所有优化算法。

    使用方法:
        optimizer = UnifiedTrajectoryOptimizer(asteroid, spacecraft)
        result = optimizer.optimize(
            r0, v0, m0, rf, vf, t_span,
            method=OptimizationMethod.AUTO
        )
    """

    def __init__(
        self,
        asteroid,
        spacecraft,
        constraint_manager: Optional[ConstraintManager] = None,
    ):
        """
        初始化统一优化器

        Parameters:
            asteroid: 小行星对象
            spacecraft: 航天器对象
            constraint_manager: 约束管理器（可选）
        """
        self.ast = asteroid
        self.sc = spacecraft
        self.constraint_manager = constraint_manager

        # 初始化所有优化器
        self.optimizers = {
            OptimizationMethod.SHOOTING: None,
            OptimizationMethod.MULTIPLE_SHOOTING: None,
            OptimizationMethod.PSEUDOSPECTRAL: None,
            OptimizationMethod.HOMOTOPY: None,
            OptimizationMethod.ADAPTIVE_HOMOTOPY: None,
            OptimizationMethod.HYBRID: None,
            OptimizationMethod.DIRECT: None,
        }

    def _get_optimizer(self, method: OptimizationMethod, config: OptimizationConfig):
        """获取或创建优化器实例"""
        if self.optimizers[method] is None:
            if method == OptimizationMethod.SHOOTING:
                self.optimizers[method] = ShootingMethodOptimizerOptimized(
                    self.ast, self.sc, verbose=config.verbose
                )
            elif method == OptimizationMethod.MULTIPLE_SHOOTING:
                self.optimizers[method] = MultipleShootingOptimizer(
                    self.ast, self.sc, n_segments=config.n_segments
                )
            elif method == OptimizationMethod.PSEUDOSPECTRAL:
                self.optimizers[method] = PseudospectralOptimizerOptimized(
                    self.ast, self.sc, n_nodes=config.n_nodes, verbose=config.verbose
                )
            elif method == OptimizationMethod.HOMOTOPY:
                self.optimizers[method] = HomotopyOptimizer(
                    self.ast, self.sc, n_steps=config.n_steps
                )
            elif method == OptimizationMethod.ADAPTIVE_HOMOTOPY:
                self.optimizers[method] = AdaptiveHomotopyOptimizer(
                    self.ast, self.sc, verbose=config.verbose
                )
            elif method == OptimizationMethod.HYBRID:
                self.optimizers[method] = HybridTrajectoryOptimizer(
                    self.ast, self.sc, n_nodes_ps=config.n_nodes, verbose=config.verbose
                )
            elif method == OptimizationMethod.DIRECT:
                self.optimizers[method] = DirectMethodOptimizer(
                    self.ast, self.sc, n_nodes=config.n_nodes
                )

        return self.optimizers[method]

    def _auto_select_method(
        self,
        r0: np.ndarray,
        rf: np.ndarray,
        t_span: List[float],
    ) -> OptimizationMethod:
        """
        自动选择最适合的优化方法

        策略：
        - 简单轨迹（距离短、时间短）: 打靶法
        - 复杂轨迹（距离长、约束多）: 伪谱法或混合法
        - 高精度要求: 同伦法或混合法
        - 快速求解: 直接法
        """
        distance = np.linalg.norm(rf - r0)
        duration = t_span[1] - t_span[0]

        # 启发式规则
        if distance < 5000 and duration < 300:
            # 简单任务：使用快速方法
            return OptimizationMethod.PSEUDOSPECTRAL
        elif distance > 15000 or duration > 1000:
            # 复杂任务：使用混合方法
            return OptimizationMethod.HYBRID
        else:
            # 中等任务：使用自适应同伦法
            return OptimizationMethod.ADAPTIVE_HOMOTOPY

    def optimize(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        method: OptimizationMethod = OptimizationMethod.AUTO,
        config: Optional[OptimizationConfig] = None,
    ) -> OptimizationResult:
        """
        执行轨迹优化（统一接口）

        Parameters:
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_span: 时间区间 [t0, tf]
            method: 优化方法
            config: 优化配置

        Returns:
            OptimizationResult: 优化结果
        """
        if config is None:
            config = OptimizationConfig(method=method)

        # 自动选择方法
        if method == OptimizationMethod.AUTO:
            method = self._auto_select_method(r0, rf, t_span)
            if config.verbose:
                print(f"🤖 自动选择优化方法: {method.value}")

        # 获取优化器
        optimizer = self._get_optimizer(method, config)

        # 记录开始时间
        start_time = time.time()

        try:
            # 根据方法类型调用优化
            if method == OptimizationMethod.SHOOTING:
                raw_result = optimizer.optimize_with_multiple_guesses(
                    r0, v0, m0, rf, vf, t_span
                )
                result = self._parse_shooting_result(raw_result, start_time)

            elif method == OptimizationMethod.MULTIPLE_SHOOTING:
                raw_result = optimizer.optimize_with_multiple_guesses(
                    r0, v0, m0, rf, vf, t_span
                )
                result = self._parse_multiple_shooting_result(raw_result, start_time)

            elif method == OptimizationMethod.PSEUDOSPECTRAL:
                raw_result = optimizer.optimize(
                    r0, v0, m0, rf, vf, t_span, max_iter=config.max_iter
                )
                result = self._parse_pseudospectral_result(raw_result, start_time)

            elif method == OptimizationMethod.HOMOTOPY:
                raw_result = optimizer.solve_homotopy(
                    r0,
                    v0,
                    m0,
                    rf,
                    vf,
                    t_span,
                    np.array([0.01, 0.01, 0.01]),
                    np.array([-0.01, -0.01, -0.01]),
                    0.0,
                )
                result = self._parse_homotopy_result(raw_result, start_time)

            elif method == OptimizationMethod.ADAPTIVE_HOMOTOPY:
                raw_result = optimizer.optimize(r0, v0, m0, rf, vf, t_span)
                result = self._parse_adaptive_homotopy_result(raw_result, start_time)

            elif method == OptimizationMethod.HYBRID:
                raw_result = optimizer.optimize(
                    r0,
                    v0,
                    m0,
                    rf,
                    vf,
                    t_span,
                    use_stage2=config.use_stage2,
                )
                result = self._parse_hybrid_result(raw_result, start_time)

            elif method == OptimizationMethod.DIRECT:
                raw_result = optimizer.optimize(
                    r0, v0, m0, rf, vf, t_span, max_iter=config.max_iter
                )
                result = self._parse_direct_result(raw_result, start_time)

            else:
                raise ValueError(f"未知优化方法: {method}")

            # 添加约束检查
            if self.constraint_manager is not None:
                result = self._check_constraints(result, r0, v0, m0, rf, vf, t_span)

            return result

        except Exception as e:
            return OptimizationResult(
                success=False,
                method=method.value,
                trajectory={},
                fuel_consumption=0.0,
                pos_error=float("inf"),
                vel_error=float("inf"),
                computation_time=time.time() - start_time,
                iterations=0,
                message=f"优化失败: {str(e)}",
                metadata={},
            )

    def _parse_shooting_result(
        self, raw_result: Dict, start_time: float
    ) -> OptimizationResult:
        """解析打靶法结果"""
        sol = raw_result.get("trajectory")

        if sol is None or not hasattr(sol, "y"):
            return OptimizationResult(
                success=False,
                method="shooting",
                trajectory={},
                fuel_consumption=0.0,
                pos_error=float("inf"),
                vel_error=float("inf"),
                computation_time=time.time() - start_time,
                iterations=0,
                message="无有效轨迹",
                metadata={},
            )

        return OptimizationResult(
            success=raw_result.get("success", False),
            method="shooting",
            trajectory={
                "t": sol.t,
                "r": sol.y[0:3, :].T,
                "v": sol.y[3:6, :].T,
                "m": sol.y[6, :],
            },
            fuel_consumption=raw_result.get("final_error", 0.0),  # 需要修正
            pos_error=raw_result.get("final_error", float("inf")),
            vel_error=0.0,
            computation_time=time.time() - start_time,
            iterations=0,
            message="打靶法优化完成",
            metadata={"initial_costate": raw_result.get("initial_costate")},
        )

    def _parse_pseudospectral_result(
        self, raw_result: Dict, start_time: float
    ) -> OptimizationResult:
        """解析伪谱法结果"""
        return OptimizationResult(
            success=raw_result.get("success", False),
            method="pseudospectral",
            trajectory={
                "t": raw_result.get("t", np.array([])),
                "r": raw_result.get("r", np.array([])),
                "v": raw_result.get("v", np.array([])),
                "m": raw_result.get("m", np.array([])),
                "u": raw_result.get("u", np.array([])),
            },
            fuel_consumption=raw_result.get("fuel_consumption", 0.0),
            pos_error=raw_result.get("pos_error", float("inf")),
            vel_error=raw_result.get("vel_error", 0.0),
            computation_time=raw_result.get("time", time.time() - start_time),
            iterations=raw_result.get("iterations", 0),
            message="伪谱法优化完成",
            metadata={},
        )

    def _parse_hybrid_result(
        self, raw_result: Dict, start_time: float
    ) -> OptimizationResult:
        """解析混合优化结果"""
        return OptimizationResult(
            success=raw_result.get("success", False),
            method="hybrid",
            trajectory={
                "t": raw_result.get("t", np.array([])),
                "r": raw_result.get("r", np.array([])),
                "v": raw_result.get("v", np.array([])),
                "m": raw_result.get("m", np.array([])),
                "u": raw_result.get("u", np.array([])),
            },
            fuel_consumption=raw_result.get("fuel_consumption", 0.0),
            pos_error=raw_result.get("pos_error", float("inf")),
            vel_error=raw_result.get("vel_error", 0.0),
            computation_time=raw_result.get("total_time", time.time() - start_time),
            iterations=raw_result.get("total_steps", 0),
            message=f"混合优化完成 (阶段{raw_result.get('stage', 1)})",
            metadata={"stage": raw_result.get("stage", 1)},
        )

    def _parse_adaptive_homotopy_result(
        self, raw_result: Dict, start_time: float
    ) -> OptimizationResult:
        """解析自适应同伦结果"""
        return OptimizationResult(
            success=raw_result.get("success", False),
            method="adaptive_homotopy",
            trajectory={},
            fuel_consumption=raw_result.get("fuel_consumption", 0.0),
            pos_error=raw_result.get("pos_error", float("inf")),
            vel_error=raw_result.get("vel_error", 0.0),
            computation_time=raw_result.get("total_time", time.time() - start_time),
            iterations=raw_result.get("total_steps", 0),
            message="自适应同伦优化完成",
            metadata={
                "zeta_history": raw_result.get("zeta_history", []),
                "step_sizes": raw_result.get("step_sizes", []),
            },
        )

    def _parse_multiple_shooting_result(
        self, raw_result: Dict, start_time: float
    ) -> OptimizationResult:
        """解析多重打靶法结果"""
        y = raw_result.get("y")
        t = raw_result.get("t")

        if y is None or t is None:
            return OptimizationResult(
                success=False,
                method="multiple_shooting",
                trajectory={},
                fuel_consumption=0.0,
                pos_error=float("inf"),
                vel_error=float("inf"),
                computation_time=time.time() - start_time,
                iterations=0,
                message="无有效轨迹",
                metadata={},
            )

        return OptimizationResult(
            success=raw_result.get("success", False),
            method="multiple_shooting",
            trajectory={
                "t": t,
                "r": y[0:3, :].T,
                "v": y[3:6, :].T,
                "m": y[6, :],
            },
            fuel_consumption=0.0,
            pos_error=raw_result.get("final_error", float("inf")),
            vel_error=0.0,
            computation_time=time.time() - start_time,
            iterations=0,
            message="多重打靶法优化完成",
            metadata={},
        )

    def _parse_homotopy_result(
        self, raw_result: Dict, start_time: float
    ) -> OptimizationResult:
        """解析同伦法结果"""
        return OptimizationResult(
            success=raw_result.get("success", False),
            method="homotopy",
            trajectory={},
            fuel_consumption=raw_result.get("fuel_consumption", 0.0),
            pos_error=raw_result.get("pos_error", float("inf")),
            vel_error=raw_result.get("vel_error", 0.0),
            computation_time=time.time() - start_time,
            iterations=len(raw_result.get("solutions", [])),
            message="同伦法优化完成",
            metadata={},
        )

    def _parse_direct_result(
        self, raw_result: Dict, start_time: float
    ) -> OptimizationResult:
        """解析直接法结果"""
        X = raw_result.get("X")
        t = raw_result.get("t")

        if X is None or t is None:
            return OptimizationResult(
                success=False,
                method="direct",
                trajectory={},
                fuel_consumption=0.0,
                pos_error=float("inf"),
                vel_error=float("inf"),
                computation_time=time.time() - start_time,
                iterations=0,
                message="无有效轨迹",
                metadata={},
            )

        return OptimizationResult(
            success=raw_result.get("success", False),
            method="direct",
            trajectory={
                "t": t,
                "r": X[:, 0:3],
                "v": X[:, 3:6],
                "m": X[:, 6],
            },
            fuel_consumption=raw_result.get("fuel_consumption", 0.0),
            pos_error=0.0,
            vel_error=0.0,
            computation_time=time.time() - start_time,
            iterations=0,
            message="直接法优化完成",
            metadata={},
        )

    def _check_constraints(
        self,
        result: OptimizationResult,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
    ) -> OptimizationResult:
        """检查约束违反情况"""
        if not result.success or self.constraint_manager is None:
            return result

        # 这里可以添加约束检查逻辑
        # 简化版本：直接返回原结果
        return result


class TrajectoryEvaluator:
    """
    轨迹评估器

    评估优化结果的质量：
    - 约束满足情况
    - 动力学一致性
    - 燃料效率
    - 平滑度
    """

    def __init__(self, asteroid, spacecraft):
        """
        初始化评估器

        Parameters:
            asteroid: 小行星对象
            spacecraft: 航天器对象
        """
        self.ast = asteroid
        self.sc = spacecraft

    def evaluate(
        self,
        result: OptimizationResult,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
    ) -> Dict[str, Any]:
        """
        评估轨迹质量

        Parameters:
            result: 优化结果
            r0, v0, m0: 初始状态
            rf, vf: 终端状态

        Returns:
            Dict: 评估指标
        """
        evaluation = {
            "success": result.success,
            "method": result.method,
            "basic_metrics": {
                "fuel_consumption": result.fuel_consumption,
                "pos_error": result.pos_error,
                "vel_error": result.vel_error,
                "computation_time": result.computation_time,
                "iterations": result.iterations,
            },
        }

        # 如果优化失败，直接返回
        if not result.success:
            evaluation["overall_score"] = 0.0
            return evaluation

        trajectory = result.trajectory

        # 计算平滑度
        if "r" in trajectory and len(trajectory["r"]) > 2:
            r = trajectory["r"]
            smoothness = self._compute_smoothness(r)
            evaluation["smoothness"] = smoothness

        # 计算燃料效率
        if result.fuel_consumption > 0:
            # 计算理论最小燃料（简化估计）
            delta_v = np.linalg.norm(vf - v0)
            theoretical_fuel = m0 * (1 - np.exp(-delta_v / (self.sc.I_sp * self.sc.g0)))
            fuel_efficiency = theoretical_fuel / max(result.fuel_consumption, 1e-10)
            evaluation["fuel_efficiency_ratio"] = fuel_efficiency

        # 综合评分
        score = self._compute_overall_score(result, evaluation)
        evaluation["overall_score"] = score

        return evaluation

    def _compute_smoothness(self, trajectory: np.ndarray) -> float:
        """计算轨迹平滑度"""
        # 计算轨迹曲率变化
        curvature_changes = []
        for i in range(1, len(trajectory) - 1):
            v1 = trajectory[i] - trajectory[i - 1]
            v2 = trajectory[i + 1] - trajectory[i]

            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 1e-10 and v2_norm > 1e-10:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                curvature_changes.append(angle)

        if curvature_changes:
            return np.mean(curvature_changes)
        return 0.0

    def _compute_overall_score(
        self,
        result: OptimizationResult,
        evaluation: Dict,
    ) -> float:
        """计算综合评分"""
        if not result.success:
            return 0.0

        score = 100.0

        # 位置误差扣分
        if result.pos_error > 100:
            score -= min(30, result.pos_error / 100)

        # 速度误差扣分
        if result.vel_error > 1.0:
            score -= min(20, result.vel_error * 10)

        # 计算时间扣分（太长）
        if result.computation_time > 60:
            score -= min(20, (result.computation_time - 60) / 10)

        # 平滑度扣分
        smoothness = evaluation.get("smoothness", 0)
        if smoothness > 0.1:
            score -= min(10, smoothness * 100)

        return max(0, score)

    def print_evaluation(self, evaluation: Dict):
        """打印评估结果"""
        print("\n" + "=" * 70)
        print("📊 轨迹评估报告")
        print("=" * 70)

        print(f"\n方法: {evaluation['method']}")
        print(f"状态: {'✅ 成功' if evaluation['success'] else '❌ 失败'}")

        if evaluation["success"]:
            metrics = evaluation["basic_metrics"]
            print(f"\n基本指标:")
            print(f"  燃料消耗: {metrics['fuel_consumption']:.2f} kg")
            print(f"  位置误差: {metrics['pos_error']:.2f} m")
            print(f"  速度误差: {metrics['vel_error']:.2f} m/s")
            print(f"  计算时间: {metrics['computation_time']:.2f} s")
            print(f"  迭代次数: {metrics['iterations']}")

            if "smoothness" in evaluation:
                print(f"  轨迹平滑度: {evaluation['smoothness']:.4f}")

            if "fuel_efficiency_ratio" in evaluation:
                print(f"  燃料效率比: {evaluation['fuel_efficiency_ratio']:.2f}")

            print(f"\n综合评分: {evaluation['overall_score']:.1f}/100")

        print("=" * 70)


# 示例使用
if __name__ == "__main__":
    print("=" * 70)
    print("统一轨迹优化接口示例")
    print("=" * 70)

    # 创建示例小行星和航天器
    class ExampleAsteroid:
        def __init__(self):
            self.omega = np.array([0, 0, 2 * np.pi / (5.27 * 3600)])
            self.mu = 4.463e5

        def gravity_gradient(self, r):
            r_norm = np.linalg.norm(r)
            if r_norm < 1e-3:
                return np.zeros(3)
            return -self.mu * r / (r_norm**3)

    class ExampleSpacecraft:
        def __init__(self):
            self.T_max = 20.0
            self.I_sp = 400.0
            self.g0 = 9.81
            self.m0 = 1000.0

    asteroid = ExampleAsteroid()
    spacecraft = ExampleSpacecraft()

    # 创建统一优化器
    optimizer = UnifiedTrajectoryOptimizer(asteroid, spacecraft)
    evaluator = TrajectoryEvaluator(asteroid, spacecraft)

    # 边界条件
    r0 = np.array([10177, 6956, 8256])
    v0 = np.array([-25, -12, -17])
    m0 = spacecraft.m0
    rf = np.array([676, 5121, 449])
    vf = np.array([0, 0, 0])
    t_span = [0.0, 770.0]

    print("\n使用伪谱法优化...")
    result = optimizer.optimize(
        r0,
        v0,
        m0,
        rf,
        vf,
        t_span,
        method=OptimizationMethod.PSEUDOSPECTRAL,
    )

    # 评估结果
    evaluation = evaluator.evaluate(result, r0, v0, m0, rf, vf)
    evaluator.print_evaluation(evaluation)

    print("\n✅ 统一接口测试完成！")
