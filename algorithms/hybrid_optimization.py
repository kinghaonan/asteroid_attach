#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 混合优化算法（Hybrid Optimization）

结合伪谱法的全局搜索能力和打靶法的局部精度优势。
核心策略：
1. 阶段1：使用伪谱法（直接法）进行粗搜索，快速获得可行解
2. 阶段2：将伪谱法结果转换为打靶法初始猜测
3. 阶段3：使用打靶法（间接法）进行精细优化，获得高精度解

优势：
- 伪谱法：全局收敛性好，适合快速获得粗略轨迹
- 打靶法：局部精度高，燃料最优性强
- 混合策略：兼顾效率和精度

参考：小天体附着多约束轨迹优化（公式3-1）
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, fsolve
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Callable
import time
import warnings
from tqdm import tqdm

from .pseudospectral_optimized import PseudospectralOptimizerOptimized, StateScaler


class HybridTrajectoryOptimizer:
    """
    混合轨迹优化器

    结合伪谱法和打靶法的优势，实现高效且高精度的轨迹优化。

    Attributes:
        asteroid: 小行星模型
        spacecraft: 航天器参数
        ps_optimizer: 伪谱法优化器（粗搜索）
        verbose: 是否显示详细信息
        stage1_result: 阶段1（伪谱法）结果
        stage2_result: 阶段2（打靶法）结果
    """

    def __init__(
        self,
        asteroid,
        spacecraft,
        n_nodes_ps: int = 25,
        verbose: bool = True,
    ):
        """
        初始化混合优化器

        Parameters:
            asteroid: 小行星对象
            spacecraft: 航天器对象
            n_nodes_ps: 伪谱法节点数（默认25）
            verbose: 是否显示详细信息
        """
        self.ast = asteroid
        self.sc = spacecraft
        self.verbose = verbose
        self.n_nodes_ps = n_nodes_ps

        # 初始化伪谱法优化器（阶段1）
        self.ps_optimizer = PseudospectralOptimizerOptimized(
            asteroid, spacecraft, n_nodes=n_nodes_ps, verbose=verbose
        )

        self.stage1_result = None
        self.stage2_result = None
        self.best_error = float("inf")

    def stage1_pseudospectral_search(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        max_iter: int = 200,
    ) -> Dict:
        """
        阶段1：伪谱法粗搜索

        使用直接法伪谱优化快速获得可行轨迹。
        这是混合算法的第一阶段，目标是获得一个合理的初始猜测。

        Parameters:
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_span: 时间区间
            max_iter: 最大迭代次数

        Returns:
            result: 伪谱法优化结果
        """
        if self.verbose:
            print(f"\n{'=' * 70}")
            print("🔮 阶段1: 伪谱法粗搜索 (全局优化)")
            print(f"{'=' * 70}")
            print("目标: 快速获得可行轨迹，为打靶法提供初始猜测")

        start_time = time.time()

        # 使用伪谱法优化
        result = self.ps_optimizer.optimize(
            r0, v0, m0, rf, vf, t_span, max_iter=max_iter
        )

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"\n⏱️  伪谱法耗时: {elapsed:.2f}秒")
            if result["success"]:
                print(f"✅ 伪谱法收敛成功")
                print(f"   燃料消耗: {result['fuel_consumption']:.2f} kg")
                print(f"   位置误差: {result['pos_error']:.2f} m")
            else:
                print(f"⚠️ 伪谱法未完全收敛，但仍可使用结果")

        self.stage1_result = result
        result["stage"] = 1
        result["time"] = elapsed

        return result

    def convert_ps_to_shooting_guess(
        self,
        ps_result: Dict,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
    ) -> np.ndarray:
        """
        将伪谱法结果转换为打靶法初始猜测

        基于伪谱法的轨迹，通过反向积分估计初始协态。
        这是混合算法的关键步骤，需要精确计算。

        Parameters:
            ps_result: 伪谱法结果
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_span: 时间区间

        Returns:
            lam0_guess: 打靶法初始协态猜测 [lam_r0(3), lam_v0(3), lam_m0]
        """
        if self.verbose:
            print(f"\n{'=' * 70}")
            print("🔄 阶段1→2: 转换伪谱结果为打靶法初始猜测")
            print(f"{'=' * 70}")

        t_nodes = ps_result["t"]
        r_nodes = ps_result["r"]
        v_nodes = ps_result["v"]
        m_nodes = ps_result["m"]
        u_nodes = ps_result["u"]

        # 方法1: 基于最优控制理论反向估计协态
        # 在最优控制中，推力方向 χ = -λ_v / ||λ_v||
        # 因此可以从推力方向估计 λ_v 的方向

        # 获取第一个节点的信息
        r0_ps = r_nodes[0]
        v0_ps = v_nodes[0]
        m0_ps = m_nodes[0]
        u0_ps = u_nodes[0]

        # 计算需要的速度变化
        delta_v = vf - v0
        delta_v_norm = np.linalg.norm(delta_v)

        # 计算需要的速度方向
        if delta_v_norm > 1e-6:
            thrust_dir = delta_v / delta_v_norm
        else:
            thrust_dir = np.array([0, 0, 1])

        # 估计初始速度协态
        # 在最优控制中，推力方向与 -λ_v 同向
        lam_v0_guess = -thrust_dir * 0.1

        # 估计初始位置协态
        # 基于位置偏差和速度关系
        delta_r = rf - r0
        delta_r_norm = np.linalg.norm(delta_r)
        if delta_r_norm > 1e-6:
            lam_r0_guess = delta_r / delta_r_norm * 0.05
        else:
            lam_r0_guess = np.array([0.01, 0.01, 0.01])

        # 估计初始质量协态
        # 基于燃料消耗估计
        fuel_rate = self.sc.T_max / (self.sc.I_sp * self.sc.g0)
        total_fuel = m0 - m_nodes[-1]
        avg_thrust_time = total_fuel / fuel_rate / (t_span[1] - t_span[0])

        # 开关函数 H_r = 1 - λ_m - (I_sp*g0/m)*||λ_v||
        # 在燃料最优时，H_r ≈ 0
        lam_v_norm = np.linalg.norm(lam_v0_guess)
        lam_m0_guess = 1 - (self.sc.I_sp * self.sc.g0 / m0) * lam_v_norm
        lam_m0_guess = np.clip(lam_m0_guess, -1.0, 1.0)

        lam0_guess = np.concatenate([lam_r0_guess, lam_v0_guess, [lam_m0_guess]])

        if self.verbose:
            print(f"初始协态猜测:")
            print(
                f"   λ_r0: [{lam_r0_guess[0]:.4f}, {lam_r0_guess[1]:.4f}, {lam_r0_guess[2]:.4f}]"
            )
            print(
                f"   λ_v0: [{lam_v0_guess[0]:.4f}, {lam_v0_guess[1]:.4f}, {lam_v0_guess[2]:.4f}]"
            )
            print(f"   λ_m0: {lam_m0_guess:.4f}")

        return lam0_guess

    def dynamics_full(self, t: float, z: np.ndarray) -> np.ndarray:
        """
        完整的状态-协态动力学方程（用于打靶法）

        基于小天体附着多约束轨迹优化方程（3-1式）
        """
        r = z[0:3]
        v = z[3:6]
        m = z[6]
        lam_r = z[7:10]
        lam_v = z[10:13]
        lam_m = z[13]

        # 计算最优控制
        chi, u = self._optimal_control_bang_bang(lam_v, lam_m, m)

        # 状态方程（3-1式）
        # ṙ = v
        drdt = v

        # v̇ = -2ω×v - ω×(ω×r) + ∇U + (T_max/m)uχ
        omega = self.ast.omega
        coriolis = -2 * np.cross(omega, v)
        centrifugal = -np.cross(omega, np.cross(omega, r))
        gravity = self.ast.gravity_gradient(r)

        thrust_acc = np.zeros(3)
        if m > 1e-3:
            thrust_acc = (self.sc.T_max / m) * u * chi

        dvdt = coriolis + centrifugal + gravity + thrust_acc

        # ṁ = -T_max*u / (I_sp*g0)
        dmdt = -self.sc.T_max * u / (self.sc.I_sp * self.sc.g0) if m > 1e-3 else 0

        # 协态方程
        # dλ_r/dt = -∂H/∂r
        # dλ_v/dt = -∂H/∂v
        # dλ_m/dt = -∂H/∂m

        omega_skew = np.array(
            [
                [0, -omega[2], omega[1]],
                [omega[2], 0, -omega[0]],
                [-omega[1], omega[0], 0],
            ]
        )

        # 引力梯度矩阵
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-3:
            G = np.zeros((3, 3))
        else:
            G = -self.ast.mu * (3 * np.outer(r, r) / r_norm**5 - np.eye(3) / r_norm**3)

        dlam_rdt = -G.T @ lam_v + omega_skew @ (omega_skew @ lam_r) - omega_skew @ lam_v
        dlam_vdt = -lam_r + 2 * omega_skew @ lam_v
        dlam_mdt = (
            -np.linalg.norm(lam_v) * self.sc.T_max * u / (m**2) if m > 1e-3 else 0
        )

        return np.concatenate([drdt, dvdt, [dmdt], dlam_rdt, dlam_vdt, [dlam_mdt]])

    def _optimal_control_bang_bang(
        self, lam_v: np.ndarray, lam_m: float, m: float
    ) -> Tuple[np.ndarray, float]:
        """
        计算Bang-Bang最优控制（燃料最优）

        开关函数: H_r = 1 - λ_m - (I_sp*g0/m)*||λ_v||
        """
        lam_v_norm = np.linalg.norm(lam_v)
        if lam_v_norm > 1e-6:
            chi = -lam_v / lam_v_norm
        else:
            chi = np.array([0.0, 0.0, 1.0])

        # 开关函数
        H_r = 1 - lam_m - (self.sc.I_sp * self.sc.g0 / m) * lam_v_norm

        # Bang-Bang控制
        if H_r > 0:
            u = 0.0
        elif H_r < 0:
            u = 1.0
        else:
            u = 0.5

        return chi, u

    def shooting_function(
        self,
        lam0: np.ndarray,
        t_span: List[float],
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
    ) -> np.ndarray:
        """
        打靶函数：计算终端误差

        Parameters:
            lam0: 初始协态 [lam_r0(3), lam_v0(3), lam_m0]
            t_span: 时间区间
            r0, v0, m0: 初始状态
            rf, vf: 终端状态

        Returns:
            error: 终端误差向量 [r_err(3), v_err(3), lam_m_err]
        """
        z0 = np.concatenate([r0, v0, [m0], lam0])

        try:
            sol = solve_ivp(
                self.dynamics_full,
                t_span,
                z0,
                method="RK45",
                rtol=1e-6,
                atol=1e-9,
                dense_output=True,
            )

            if not sol.success:
                return np.ones(7) * 1e10

            # 提取终端状态
            zf = sol.y[:, -1]
            r_tf = zf[0:3]
            v_tf = zf[3:6]
            lam_m_tf = zf[13]

            # 计算误差
            pos_err = r_tf - rf
            vel_err = v_tf - vf

            # 归一化误差（位置误差更重要）
            error = np.concatenate(
                [
                    pos_err / 1000.0,  # 归一化到km
                    vel_err / 10.0,  # 归一化到10m/s
                    [lam_m_tf],  # 横截条件
                ]
            )

            return error

        except Exception as e:
            return np.ones(7) * 1e10

    def stage2_shooting_refinement(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        lam0_guess: np.ndarray,
        max_iter: int = 100,
    ) -> Dict:
        """
        阶段2：打靶法精细优化

        使用伪谱法的结果作为初始猜测，通过打靶法获得高精度解。
        这是混合算法的第二阶段，目标是提高解的精度。

        Parameters:
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_span: 时间区间
            lam0_guess: 初始协态猜测
            max_iter: 最大迭代次数

        Returns:
            result: 打靶法优化结果
        """
        if self.verbose:
            print(f"\n{'=' * 70}")
            print("🎯 阶段2: 打靶法精细优化 (局部优化)")
            print(f"{'=' * 70}")
            print("目标: 提高轨迹精度，获得燃料最优解")

        start_time = time.time()

        # 定义打靶方程
        def shooting_eq(lam0):
            return self.shooting_function(lam0, t_span, r0, v0, m0, rf, vf)

        # 使用多种求解方法
        methods = [
            (
                "fsolve",
                lambda: fsolve(
                    shooting_eq,
                    lam0_guess,
                    full_output=True,
                    xtol=1e-6,
                    maxfev=max_iter,
                ),
            ),
            (
                "minimize",
                lambda: self._minimize_shooting(shooting_eq, lam0_guess, max_iter),
            ),
        ]

        best_result = None
        best_error = float("inf")

        for method_name, method_func in methods:
            try:
                if self.verbose:
                    print(f"\n尝试 {method_name}...")

                result = method_func()

                if method_name == "fsolve":
                    lam0_sol, info, ier, msg = result
                    if ier == 1:
                        error = shooting_eq(lam0_sol)
                        error_norm = np.linalg.norm(error[:6])
                        if error_norm < best_error:
                            best_error = error_norm
                            best_result = (lam0_sol, error_norm, "fsolve")
                            if self.verbose:
                                print(
                                    f"  ✅ {method_name} 成功，误差: {error_norm:.6f}"
                                )
                    else:
                        if self.verbose:
                            print(f"  ⚠️ {method_name} 未收敛: {msg}")
                else:
                    lam0_sol, error_norm, success = result
                    if success and error_norm < best_error:
                        best_error = error_norm
                        best_result = (lam0_sol, error_norm, "minimize")
                        if self.verbose:
                            print(f"  ✅ {method_name} 成功，误差: {error_norm:.6f}")

            except Exception as e:
                if self.verbose:
                    print(f"  ❌ {method_name} 失败: {str(e)}")
                continue

        elapsed = time.time() - start_time

        # 获取最终轨迹
        if best_result is not None:
            lam0_sol, final_error, method_used = best_result

            # 积分最终轨迹
            z0 = np.concatenate([r0, v0, [m0], lam0_sol])
            sol = solve_ivp(
                self.dynamics_full,
                t_span,
                z0,
                method="RK45",
                rtol=1e-6,
                atol=1e-9,
                t_eval=np.linspace(t_span[0], t_span[1], 200),
                dense_output=True,
            )

            if sol.success:
                zf = sol.y[:, -1]
                final_mass = zf[6]
                fuel_consumption = m0 - final_mass

                result = {
                    "success": final_error < 100,  # 100m误差视为成功
                    "initial_costate": lam0_sol,
                    "trajectory": sol,
                    "final_error": final_error,
                    "final_mass": final_mass,
                    "fuel_consumption": fuel_consumption,
                    "method": method_used,
                    "time": elapsed,
                    "stage": 2,
                }

                if self.verbose:
                    print(f"\n✅ 打靶法优化成功")
                    print(f"   使用算法: {method_used}")
                    print(f"   终端误差: {final_error:.6f}")
                    print(f"   燃料消耗: {fuel_consumption:.2f} kg")
                    print(f"   耗时: {elapsed:.2f}秒")

                self.stage2_result = result
                return result

        # 如果都失败，返回阶段1的结果
        if self.verbose:
            print(f"\n⚠️ 打靶法优化失败，使用伪谱法结果")

        return {
            "success": False,
            "message": "打靶法优化失败",
            "stage": 2,
            "time": elapsed,
        }

    def _minimize_shooting(
        self, shooting_eq: Callable, lam0_guess: np.ndarray, max_iter: int
    ) -> Tuple[np.ndarray, float, bool]:
        """使用最小化方法求解打靶方程"""

        def objective(lam0):
            error = shooting_eq(lam0)
            return np.sum(error**2)

        result = minimize(
            objective,
            lam0_guess,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": 1e-8},
        )

        if result.success:
            final_error = np.linalg.norm(shooting_eq(result.x)[:6])
            return result.x, final_error, True
        else:
            return lam0_guess, float("inf"), False

    def optimize(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        use_stage2: bool = True,
        ps_max_iter: int = 200,
        shooting_max_iter: int = 100,
    ) -> Dict:
        """
        执行混合优化（完整流程）

        Parameters:
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_span: 时间区间
            use_stage2: 是否使用阶段2（打靶法）
            ps_max_iter: 伪谱法最大迭代次数
            shooting_max_iter: 打靶法最大迭代次数

        Returns:
            result: 优化结果字典
        """
        if self.verbose:
            print(f"\n{'=' * 70}")
            print("🚀 混合轨迹优化算法")
            print(f"{'=' * 70}")
            print("策略: 伪谱法(全局) → 打靶法(局部)")

        total_start_time = time.time()

        # ===== 阶段1: 伪谱法粗搜索 =====
        stage1_result = self.stage1_pseudospectral_search(
            r0, v0, m0, rf, vf, t_span, max_iter=ps_max_iter
        )

        # 如果伪谱法失败，直接返回
        if (
            not stage1_result["success"]
            and stage1_result.get("pos_error", 1e10) > 10000
        ):
            if self.verbose:
                print("\n❌ 伪谱法粗搜索失败，混合优化终止")
            return {
                "success": False,
                "stage1": stage1_result,
                "message": "伪谱法粗搜索失败",
            }

        # ===== 阶段2: 打靶法精细优化（可选） =====
        stage2_result = None
        if use_stage2:
            # 转换初始猜测
            lam0_guess = self.convert_ps_to_shooting_guess(
                stage1_result, r0, v0, m0, rf, vf, t_span
            )

            # 执行打靶法优化
            stage2_result = self.stage2_shooting_refinement(
                r0, v0, m0, rf, vf, t_span, lam0_guess, max_iter=shooting_max_iter
            )

        total_time = time.time() - total_start_time

        # 整合结果
        final_result = self._combine_results(stage1_result, stage2_result, total_time)

        if self.verbose:
            print(f"\n{'=' * 70}")
            print("📊 混合优化完成")
            print(f"{'=' * 70}")
            print(f"总耗时: {total_time:.2f}秒")
            if final_result["success"]:
                print(f"✅ 优化成功")
                print(f"   最优燃料: {final_result['fuel_consumption']:.2f} kg")
                print(f"   位置误差: {final_result['pos_error']:.2f} m")
            else:
                print(f"⚠️ 优化部分成功")

        return final_result

    def _combine_results(
        self, stage1: Dict, stage2: Optional[Dict], total_time: float
    ) -> Dict:
        """整合两个阶段的优化结果"""

        result = {
            "success": stage1["success"],
            "stage1": stage1,
            "total_time": total_time,
        }

        if stage2 is not None and stage2.get("success", False):
            # 使用阶段2的结果（精度更高）
            result["stage2"] = stage2
            result["success"] = True
            result["method"] = "hybrid (ps+shooting)"
            result["fuel_consumption"] = stage2["fuel_consumption"]
            result["final_mass"] = stage2["final_mass"]
            result["final_error"] = stage2["final_error"]
            result["trajectory"] = stage2["trajectory"]
            result["stage"] = 2
        else:
            # 使用阶段1的结果
            result["method"] = "pseudospectral only"
            result["fuel_consumption"] = stage1["fuel_consumption"]
            result["final_mass"] = stage1["m"][-1] if "m" in stage1 else None
            result["final_error"] = stage1["pos_error"]
            result["trajectory"] = stage1
            result["stage"] = 1

        # 添加基本轨迹数据
        if result["stage"] == 2 and stage2 is not None:
            sol = stage2["trajectory"]
            result["t"] = sol.t
            result["r"] = sol.y[0:3, :].T
            result["v"] = sol.y[3:6, :].T
            result["m"] = sol.y[6, :]
            # 计算控制历史
            result["u"] = self._compute_control_history(sol)
        else:
            result["t"] = stage1["t"]
            result["r"] = stage1["r"]
            result["v"] = stage1["v"]
            result["m"] = stage1["m"]
            result["u"] = stage1["u"]

        result["pos_error"] = result["final_error"]
        result["vel_error"] = (
            np.linalg.norm(result["v"][-1] - result["v"][0])
            if len(result["v"]) > 0
            else 0
        )

        return result

    def _compute_control_history(self, sol) -> np.ndarray:
        """从轨迹解计算控制历史"""
        n = len(sol.t)
        u_history = np.zeros(n)

        for i in range(n):
            z = sol.y[:, i]
            lam_v = z[10:13]
            lam_m = z[13]
            m = z[6]
            _, u = self._optimal_control_bang_bang(lam_v, lam_m, m)
            u_history[i] = u

        return u_history

    def plot_comparison(
        self,
        save_path: Optional[str] = None,
    ):
        """
        绘制两个阶段结果的对比图
        """
        if self.stage1_result is None:
            print("没有阶段1结果可供对比")
            return

        fig = plt.figure(figsize=(16, 12))

        # 获取数据
        t1 = self.stage1_result["t"]
        r1 = self.stage1_result["r"]
        v1 = self.stage1_result["v"]
        m1 = self.stage1_result["m"]
        u1 = self.stage1_result["u"]

        # 3D轨迹对比
        ax = fig.add_subplot(3, 3, 1, projection="3d")
        ax.plot(r1[:, 0], r1[:, 1], r1[:, 2], "b-", linewidth=2, label="阶段1: 伪谱法")

        if self.stage2_result is not None and self.stage2_result.get("trajectory"):
            sol = self.stage2_result["trajectory"]
            r2 = sol.y[0:3, :].T
            ax.plot(
                r2[:, 0], r2[:, 1], r2[:, 2], "r--", linewidth=2, label="阶段2: 打靶法"
            )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        ax.set_title("3D轨迹对比")

        # 位置对比
        for i, coord in enumerate(["X", "Y", "Z"]):
            ax = fig.add_subplot(3, 3, 2 + i)
            ax.plot(t1, r1[:, i], "b-", linewidth=2, label="伪谱法")
            if self.stage2_result is not None and self.stage2_result.get("trajectory"):
                sol = self.stage2_result["trajectory"]
                ax.plot(sol.t, sol.y[i, :], "r--", linewidth=2, label="打靶法")
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel(f"{coord} (m)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{coord}位置对比")

        # 速度对比
        for i, coord in enumerate(["Vx", "Vy", "Vz"]):
            ax = fig.add_subplot(3, 3, 5 + i)
            ax.plot(t1, v1[:, i], "b-", linewidth=2, label="伪谱法")
            if self.stage2_result is not None and self.stage2_result.get("trajectory"):
                sol = self.stage2_result["trajectory"]
                ax.plot(sol.t, sol.y[3 + i, :], "r--", linewidth=2, label="打靶法")
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel(f"{coord} (m/s)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{coord}速度对比")

        # 质量对比
        ax = fig.add_subplot(3, 3, 8)
        ax.plot(t1, m1, "b-", linewidth=2, label="伪谱法")
        if self.stage2_result is not None and self.stage2_result.get("trajectory"):
            sol = self.stage2_result["trajectory"]
            ax.plot(sol.t, sol.y[6, :], "r--", linewidth=2, label="打靶法")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("质量 (kg)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("质量对比")

        # 推力对比
        ax = fig.add_subplot(3, 3, 9)
        ax.plot(t1, u1 * self.sc.T_max, "b-", linewidth=2, label="伪谱法")
        if self.stage2_result is not None and self.stage2_result.get("trajectory"):
            sol = self.stage2_result["trajectory"]
            u2 = self._compute_control_history(sol)
            ax.plot(sol.t, u2 * self.sc.T_max, "r--", linewidth=2, label="打靶法")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("推力 (N)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("推力对比")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 对比图已保存: {save_path}")

        plt.show()


# 示例使用
if __name__ == "__main__":

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

    print("=" * 70)
    print("混合轨迹优化算法测试")
    print("=" * 70)

    asteroid = ExampleAsteroid()
    spacecraft = ExampleSpacecraft()
    optimizer = HybridTrajectoryOptimizer(asteroid, spacecraft, n_nodes_ps=20)

    # 边界条件（433 Eros示例）
    r0 = np.array([10177, 6956, 8256])
    v0 = np.array([-25, -12, -17])
    m0 = spacecraft.m0
    rf = np.array([676, 5121, 449])
    vf = np.array([0, 0, 0])
    t_span = [0, 770]

    # 执行混合优化
    result = optimizer.optimize(
        r0,
        v0,
        m0,
        rf,
        vf,
        t_span,
        use_stage2=True,
        ps_max_iter=150,
        shooting_max_iter=100,
    )

    if result["success"]:
        print("\n✅ 混合优化成功完成！")
        print(f"最终使用阶段: {result['stage']}")
        print(f"燃料消耗: {result['fuel_consumption']:.2f} kg")
        print(f"位置误差: {result['pos_error']:.2f} m")

        # 绘制对比图
        optimizer.plot_comparison()
    else:
        print("\n❌ 优化未完成")
        print(f"消息: {result.get('message', '未知错误')}")
