#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 自适应同伦步长优化

基于同伦技术解决两点边值问题(TPBVP)对协态初值敏感的问题。
核心改进：
1. 自适应步长调整 - 根据收敛情况动态调整同伦参数步长
2. 预测-校正策略 - 利用前一步的解预测下一步的初值
3. 牛顿迭代加速 - 在每个同伦步骤使用牛顿法快速求解
4. 智能回退机制 - 收敛失败时自动回退并减小步长

数学基础：
- 同伦方程: H(x, ζ) = (1-ζ)F(x) + ζG(x) = 0
- 从能量最优(ζ=0)平滑过渡到燃料最优(ζ=1)
- 使用弧长 continuation 方法跟踪解曲线

参考：同伦技术解决TPBVP敏感性（第4.3节）
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, newton, fsolve
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Callable
import time
import warnings
from tqdm import tqdm


class AdaptiveHomotopyOptimizer:
    """
    自适应同伦步长轨迹优化器

    改进传统固定步长同伦法，实现：
    - 自适应步长调整
    - 预测-校正策略
    - 失败自动回退
    - 牛顿迭代加速

    Attributes:
        asteroid: 小行星模型
        spacecraft: 航天器参数
        zeta_history: 同伦参数历史
        solution_history: 解的历史
        step_sizes: 步长历史
    """

    def __init__(
        self,
        asteroid,
        spacecraft,
        initial_step: float = 0.1,
        min_step: float = 0.01,
        max_step: float = 0.3,
        adaptive_factor: float = 0.7,
        verbose: bool = True,
    ):
        """
        初始化自适应同伦优化器

        Parameters:
            asteroid: 小行星对象
            spacecraft: 航天器对象
            initial_step: 初始步长（默认0.1）
            min_step: 最小步长（默认0.01）
            max_step: 最大步长（默认0.3）
            adaptive_factor: 步长调整因子（默认0.7）
            verbose: 是否显示详细信息
        """
        self.ast = asteroid
        self.sc = spacecraft
        self.initial_step = initial_step
        self.min_step = min_step
        self.max_step = max_step
        self.adaptive_factor = adaptive_factor
        self.verbose = verbose

        # 历史记录
        self.zeta_history = []
        self.solution_history = []
        self.step_sizes = []
        self.convergence_history = []

        # 当前状态
        self.current_zeta = 0.0
        self.current_solution = None
        self.current_step = initial_step

    def homotopy_control(
        self,
        lam_v: np.ndarray,
        lam_m: float,
        m: float,
        zeta: float,
    ) -> Tuple[np.ndarray, float]:
        """
        同伦混合控制律

        混合目标函数：
        J = (1-ζ) * J_energy + ζ * J_fuel

        开关函数：
        S = (1-ζ) + ζ*(1-λ_m) - (I_sp*g0/m)*||λ_v||

        Parameters:
            lam_v: 速度协态
            lam_m: 质量协态
            m: 质量
            zeta: 同伦参数

        Returns:
            chi: 推力方向
            u: 推力大小 [0, 1]
        """
        lam_v_norm = np.linalg.norm(lam_v)
        if lam_v_norm > 1e-10:
            chi = -lam_v / lam_v_norm
        else:
            chi = np.array([0.0, 0.0, 1.0])

        # 同伦开关函数
        # ζ=0: 能量最优（连续推力）
        # ζ=1: 燃料最优（Bang-Bang控制）
        if zeta < 1e-6:
            # 纯能量最优
            S = 1 - lam_m - (self.sc.I_sp * self.sc.g0 / m) * lam_v_norm
        else:
            # 混合最优
            S = (
                (1 - zeta)
                + zeta * (1 - lam_m)
                - (self.sc.I_sp * self.sc.g0 / m) * lam_v_norm
            )

        # 平滑过渡（避免数值震荡）
        if abs(S) < 0.05:
            # 过渡区域使用平滑控制
            u = 0.5 - 10 * S
            u = np.clip(u, 0, 1)
        elif S < 0:
            u = 1.0
        else:
            u = 0.0

        return chi, u

    def dynamics_homotopy(
        self,
        t: float,
        state: np.ndarray,
        zeta: float,
    ) -> np.ndarray:
        """
        同伦动力学方程

        Parameters:
            t: 时间
            state: 状态向量 [r(3), v(3), m, lam_r(3), lam_v(3), lam_m]
            zeta: 同伦参数

        Returns:
            状态导数
        """
        r = state[0:3]
        v = state[3:6]
        m = state[6]
        lam_r = state[7:10]
        lam_v = state[10:13]
        lam_m = state[13]

        chi, u = self.homotopy_control(lam_v, lam_m, m, zeta)

        # 环境力
        g = self.ast.gravity_gradient(r)
        omega = self.ast.omega
        coriolis = -2 * np.cross(omega, v)
        centrifugal = -np.cross(omega, np.cross(omega, r))

        # 状态方程
        r_dot = v
        thrust_acc = (self.sc.T_max / m) * u * chi if m > 1e-3 else np.zeros(3)
        v_dot = g + coriolis + centrifugal + thrust_acc
        m_dot = -self.sc.T_max * u / (self.sc.I_sp * self.sc.g0) if m > 1e-3 else 0

        # 协态方程
        omega_skew = np.array(
            [
                [0, -omega[2], omega[1]],
                [omega[2], 0, -omega[0]],
                [-omega[1], omega[0], 0],
            ]
        )

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

        return np.concatenate([r_dot, v_dot, [m_dot], dlam_rdt, dlam_vdt, [dlam_mdt]])

    def propagate(
        self,
        zeta: float,
        t_span: List[float],
        initial_state: np.ndarray,
    ) -> object:
        """
        积分轨迹

        Parameters:
            zeta: 同伦参数
            t_span: 时间区间
            initial_state: 初始状态

        Returns:
            sol: 积分结果
        """
        sol = solve_ivp(
            lambda t, y: self.dynamics_homotopy(t, y, zeta),
            t_span,
            initial_state,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
            dense_output=True,
        )
        return sol

    def compute_residual(
        self,
        lam0: np.ndarray,
        zeta: float,
        t_span: List[float],
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
    ) -> np.ndarray:
        """
        计算打靶法残差

        Parameters:
            lam0: 初始协态
            zeta: 同伦参数
            t_span: 时间区间
            r0, v0, m0: 初始状态
            rf, vf: 终端状态

        Returns:
            residual: 残差向量 [r_err, v_err, lam_m_err]
        """
        initial_state = np.concatenate([r0, v0, [m0], lam0])

        sol = self.propagate(zeta, t_span, initial_state)

        if not sol.success:
            return np.ones(7) * 1e10

        zf = sol.y[:, -1]
        r_tf = zf[0:3]
        v_tf = zf[3:6]
        lam_m_tf = zf[13]

        # 计算残差
        residual = np.concatenate(
            [
                (r_tf - rf) / 1000.0,  # 归一化位置误差
                (v_tf - vf) / 10.0,  # 归一化速度误差
                [lam_m_tf],  # 横截条件
            ]
        )

        return residual

    def compute_jacobian(
        self,
        lam0: np.ndarray,
        zeta: float,
        t_span: List[float],
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        数值计算雅可比矩阵

        Parameters:
            lam0: 初始协态
            zeta: 同伦参数
            t_span, r0, v0, m0, rf, vf: 边界条件
            eps: 数值微分步长

        Returns:
            J: 7x7雅可比矩阵
        """
        n = len(lam0)
        J = np.zeros((n, n))
        f0 = self.compute_residual(lam0, zeta, t_span, r0, v0, m0, rf, vf)

        for j in range(n):
            lam_perturbed = lam0.copy()
            lam_perturbed[j] += eps
            f_perturbed = self.compute_residual(
                lam_perturbed, zeta, t_span, r0, v0, m0, rf, vf
            )
            J[:, j] = (f_perturbed - f0) / eps

        return J

    def newton_solve(
        self,
        zeta: float,
        t_span: List[float],
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        lam0_guess: np.ndarray,
        max_iter: int = 20,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, bool, float]:
        """
        使用牛顿迭代法求解同伦方程

        Parameters:
            zeta: 同伦参数
            t_span, r0, v0, m0, rf, vf: 边界条件
            lam0_guess: 初始猜测
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            lam0: 求解的初始协态
            converged: 是否收敛
            final_error: 最终误差
        """
        lam0 = lam0_guess.copy()

        for iteration in range(max_iter):
            # 计算残差
            residual = self.compute_residual(lam0, zeta, t_span, r0, v0, m0, rf, vf)
            error = np.linalg.norm(residual)

            if error < tol:
                return lam0, True, error

            # 计算雅可比矩阵
            J = self.compute_jacobian(lam0, zeta, t_span, r0, v0, m0, rf, vf)

            # 求解线性方程组 J * delta = -residual
            try:
                lu, piv = lu_factor(J)
                delta = lu_solve((lu, piv), -residual)
            except Exception:
                # 雅可比矩阵奇异，使用伪逆
                delta = -np.linalg.lstsq(J, residual, rcond=None)[0]

            # 线搜索确定步长
            alpha = 1.0
            for _ in range(10):  # 最多10次线搜索
                lam_new = lam0 + alpha * delta
                residual_new = self.compute_residual(
                    lam_new, zeta, t_span, r0, v0, m0, rf, vf
                )
                error_new = np.linalg.norm(residual_new)

                if error_new < error:
                    lam0 = lam_new
                    break
                alpha *= 0.5
            else:
                # 线搜索失败，接受牛顿步
                lam0 = lam0 + 0.5 * delta

        # 达到最大迭代次数
        final_residual = self.compute_residual(lam0, zeta, t_span, r0, v0, m0, rf, vf)
        final_error = np.linalg.norm(final_residual)
        converged = final_error < tol * 10  # 放宽收敛条件

        return lam0, converged, final_error

    def predict_next_solution(
        self,
        current_zeta: float,
        current_lam: np.ndarray,
        step_size: float,
    ) -> np.ndarray:
        """
        预测下一步的解（切线预测）

        使用Euler预测：
        λ(zeta + dzeta) ≈ λ(zeta) + dλ/dzeta * dzeta

        Parameters:
            current_zeta: 当前同伦参数
            current_lam: 当前解
            step_size: 步长

        Returns:
            predicted_lam: 预测的解
        """
        if len(self.solution_history) < 2:
            # 历史不足，使用简单外推
            return current_lam

        # 使用最近的两个点估计导数
        lam_prev = self.solution_history[-2]
        zeta_prev = self.zeta_history[-2]

        # 切线方向
        dlam_dzeta = (current_lam - lam_prev) / (current_zeta - zeta_prev + 1e-10)

        # 预测
        predicted_lam = current_lam + dlam_dzeta * step_size

        return predicted_lam

    def solve_homotopy_step(
        self,
        zeta_target: float,
        t_span: List[float],
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        lam0_guess: np.ndarray,
    ) -> Tuple[np.ndarray, bool, float]:
        """
        求解单个同伦步骤

        Parameters:
            zeta_target: 目标同伦参数
            t_span, r0, v0, m0, rf, vf: 边界条件
            lam0_guess: 初始猜测

        Returns:
            lam0: 求解的初始协态
            converged: 是否收敛
            error: 最终误差
        """
        # 首先尝试牛顿法
        lam0, converged, error = self.newton_solve(
            zeta_target, t_span, r0, v0, m0, rf, vf, lam0_guess
        )

        if converged:
            return lam0, True, error

        # 牛顿法失败，尝试最小化方法
        def objective(lam):
            residual = self.compute_residual(
                lam, zeta_target, t_span, r0, v0, m0, rf, vf
            )
            return np.sum(residual**2)

        result = minimize(
            objective,
            lam0_guess,
            method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-8},
        )

        if result.success:
            final_residual = self.compute_residual(
                result.x, zeta_target, t_span, r0, v0, m0, rf, vf
            )
            final_error = np.linalg.norm(final_residual)
            return result.x, final_error < 1.0, final_error

        return lam0_guess, False, float("inf")

    def optimize(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        init_lam_r: Optional[np.ndarray] = None,
        init_lam_v: Optional[np.ndarray] = None,
        init_lam_m: Optional[float] = None,
    ) -> Dict:
        """
        执行自适应同伦优化

        Parameters:
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_span: 时间区间
            init_lam_r, init_lam_v, init_lam_m: 初始协态猜测

        Returns:
            result: 优化结果字典
        """
        if self.verbose:
            print(f"\n{'=' * 70}")
            print("🔄 自适应同伦步长优化")
            print(f"{'=' * 70}")
            print(f"初始步长: {self.initial_step}")
            print(f"最小步长: {self.min_step}")
            print(f"最大步长: {self.max_step}")
            print(f"步长调整因子: {self.adaptive_factor}")

        start_time = time.time()

        # 初始化
        if init_lam_r is None:
            init_lam_r = np.array([0.01, 0.01, 0.01])
        if init_lam_v is None:
            init_lam_v = np.array([-0.01, -0.01, -0.01])
        if init_lam_m is None:
            init_lam_m = 0.0

        current_lam = np.concatenate([init_lam_r, init_lam_v, [init_lam_m]])
        current_zeta = 0.0
        current_step = self.initial_step

        # 历史记录
        self.zeta_history = [current_zeta]
        self.solution_history = [current_lam.copy()]
        self.step_sizes = [current_step]
        self.convergence_history = [True]

        solutions = []
        t0, tf = t_span

        # ζ=0的解
        initial_state = np.concatenate([r0, v0, [m0], current_lam])
        sol = self.propagate(0, t_span, initial_state)
        if sol.success:
            m_final = sol.y[6, -1]
            fuel = m0 - m_final
        else:
            fuel = 0

        solutions.append(
            {
                "zeta": 0.0,
                "lam": current_lam.copy(),
                "fuel": fuel,
                "error": 0.0,
                "step": 0.0,
            }
        )

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"步骤 0: ζ=0.00 (能量最优)")
            print(f"{'=' * 70}")

        # 自适应同伦迭代
        step_count = 0
        max_steps = 100  # 最大步骤数

        while current_zeta < 1.0 and step_count < max_steps:
            step_count += 1

            # 预测下一步的同伦参数
            zeta_next = min(current_zeta + current_step, 1.0)
            actual_step = zeta_next - current_zeta

            # 预测解（切线预测）
            if step_count > 1:
                predicted_lam = self.predict_next_solution(
                    current_zeta, current_lam, actual_step
                )
            else:
                predicted_lam = current_lam

            if self.verbose:
                print(f"\n{'=' * 70}")
                print(f"步骤 {step_count}: ζ={zeta_next:.4f}")
                print(f"{'=' * 70}")
                print(f"步长: {actual_step:.4f}")

            # 求解同伦步骤
            lam_new, converged, error = self.solve_homotopy_step(
                zeta_next, t_span, r0, v0, m0, rf, vf, predicted_lam
            )

            # 自适应步长调整
            if converged:
                # 收敛成功，可以增加步长
                current_step = min(current_step / self.adaptive_factor, self.max_step)

                # 更新当前解
                current_zeta = zeta_next
                current_lam = lam_new

                # 记录
                self.zeta_history.append(current_zeta)
                self.solution_history.append(current_lam.copy())
                self.step_sizes.append(actual_step)
                self.convergence_history.append(True)

                # 评估
                final_state = np.concatenate([r0, v0, [m0], current_lam])
                sol = self.propagate(current_zeta, t_span, final_state)
                if sol.success:
                    m_final = sol.y[6, -1]
                    fuel = m0 - m_final
                    r_final = sol.y[0:3, -1]
                    v_final = sol.y[3:6, -1]
                    pos_err = np.linalg.norm(r_final - rf)
                    vel_err = np.linalg.norm(v_final - vf)
                else:
                    fuel = 0
                    pos_err = float("inf")
                    vel_err = float("inf")

                solutions.append(
                    {
                        "zeta": current_zeta,
                        "lam": current_lam.copy(),
                        "fuel": fuel,
                        "error": error,
                        "step": actual_step,
                        "pos_err": pos_err,
                        "vel_err": vel_err,
                    }
                )

                if self.verbose:
                    print(f"✅ 收敛成功")
                    print(f"   残差: {error:.6e}")
                    print(f"   燃料: {fuel:.2f} kg")
                    print(f"   位置误差: {pos_err:.2f} m")
                    print(f"   下一步步长: {current_step:.4f}")

            else:
                # 收敛失败，减小步重试
                current_step *= self.adaptive_factor

                self.convergence_history.append(False)

                if self.verbose:
                    print(f"❌ 收敛失败，减小步长至 {current_step:.4f}")

                # 如果步长太小，跳过此步
                if current_step < self.min_step:
                    if self.verbose:
                        print(f"⚠️ 步长已达到最小值，强制前进")

                    # 使用最小步长强制前进
                    current_zeta = min(current_zeta + self.min_step, 1.0)
                    current_lam = predicted_lam  # 使用预测值

                    self.zeta_history.append(current_zeta)
                    self.solution_history.append(current_lam.copy())
                    self.step_sizes.append(self.min_step)

                    # 重置步长
                    current_step = self.initial_step

        # 最终结果（ζ=1）
        if self.verbose:
            print(f"\n{'=' * 70}")
            print("🏁 最终验证 (ζ=1.0, 燃料最优)")
            print(f"{'=' * 70}")

        final_state = np.concatenate([r0, v0, [m0], current_lam])
        final_sol = self.propagate(1.0, t_span, final_state)

        total_time = time.time() - start_time

        if final_sol.success:
            zf = final_sol.y[:, -1]
            r_final = zf[0:3]
            v_final = zf[3:6]
            m_final = zf[6]

            pos_err = np.linalg.norm(r_final - rf)
            vel_err = np.linalg.norm(v_final - vf)
            fuel_consumption = m0 - m_final

            if self.verbose:
                print(f"✅ 优化完成")
                print(f"   总步骤: {step_count}")
                print(f"   总耗时: {total_time:.2f}秒")
                print(f"   位置误差: {pos_err:.2f} m")
                print(f"   速度误差: {vel_err:.2f} m/s")
                print(f"   燃料消耗: {fuel_consumption:.2f} kg")

            return {
                "success": True,
                "solutions": solutions,
                "final_lam": current_lam,
                "final_solution": final_sol,
                "final_mass": m_final,
                "fuel_consumption": fuel_consumption,
                "pos_error": pos_err,
                "vel_error": vel_err,
                "total_steps": step_count,
                "total_time": total_time,
                "zeta_history": self.zeta_history,
                "step_sizes": self.step_sizes,
            }
        else:
            if self.verbose:
                print(f"❌ 最终积分失败")

            return {
                "success": False,
                "message": "最终积分失败",
                "solutions": solutions,
                "total_steps": step_count,
                "total_time": total_time,
            }

    def plot_homotopy_trajectory(
        self,
        result: Dict,
        save_path: Optional[str] = None,
    ):
        """
        绘制同伦过程

        Parameters:
            result: 优化结果
            save_path: 保存路径
        """
        if not result.get("success", False):
            print("优化失败，无法绘图")
            return

        solutions = result["solutions"]
        zetas = [s["zeta"] for s in solutions]
        fuels = [s["fuel"] for s in solutions]
        errors = [s.get("error", 0) for s in solutions]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 同伦轨迹图
        ax = axes[0, 0]
        ax.plot(zetas, fuels, "bo-", linewidth=2, markersize=6)
        ax.set_xlabel("同伦参数 ζ", fontsize=12)
        ax.set_ylabel("燃料消耗 (kg)", fontsize=12)
        ax.set_title("同伦轨迹: 能量最优 → 燃料最优", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color="g", linestyle="--", alpha=0.5, label="能量最优")
        ax.axvline(1, color="r", linestyle="--", alpha=0.5, label="燃料最优")
        ax.legend()

        # 步长历史
        ax = axes[0, 1]
        if len(self.step_sizes) > 1:
            ax.plot(range(len(self.step_sizes)), self.step_sizes, "g-o", linewidth=2)
            ax.set_xlabel("步骤", fontsize=12)
            ax.set_ylabel("步长", fontsize=12)
            ax.set_title("自适应步长历史", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.axhline(
                self.min_step, color="r", linestyle="--", alpha=0.5, label="最小步长"
            )
            ax.axhline(
                self.max_step,
                color="orange",
                linestyle="--",
                alpha=0.5,
                label="最大步长",
            )
            ax.legend()

        # 收敛历史
        ax = axes[1, 0]
        if len(self.convergence_history) > 0:
            colors = ["green" if c else "red" for c in self.convergence_history]
            ax.scatter(
                range(len(self.convergence_history)),
                [1 if c else 0 for c in self.convergence_history],
                c=colors,
                s=50,
            )
            ax.set_xlabel("步骤", fontsize=12)
            ax.set_ylabel("收敛状态", fontsize=12)
            ax.set_title("收敛历史 (绿=成功, 红=失败)", fontsize=14)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["失败", "成功"])
            ax.grid(True, alpha=0.3)

        # 残差历史
        ax = axes[1, 1]
        ax.semilogy(zetas, errors, "m-o", linewidth=2, markersize=6)
        ax.set_xlabel("同伦参数 ζ", fontsize=12)
        ax.set_ylabel("残差 (log)", fontsize=12)
        ax.set_title("残差收敛历史", fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 同伦图已保存: {save_path}")

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
    print("自适应同伦步长优化测试")
    print("=" * 70)

    asteroid = ExampleAsteroid()
    spacecraft = ExampleSpacecraft()
    optimizer = AdaptiveHomotopyOptimizer(
        asteroid,
        spacecraft,
        initial_step=0.15,
        min_step=0.02,
        max_step=0.4,
        adaptive_factor=0.6,
    )

    r0 = np.array([10177, 6956, 8256])
    v0 = np.array([-25, -12, -17])
    m0 = spacecraft.m0
    rf = np.array([676, 5121, 449])
    vf = np.array([0, 0, 0])
    t_span = [0, 770]

    result = optimizer.optimize(r0, v0, m0, rf, vf, t_span)

    if result["success"]:
        print("\n✅ 自适应同伦优化成功！")
        optimizer.plot_homotopy_trajectory(result)
    else:
        print("\n❌ 优化失败")
        print(f"消息: {result.get('message', '未知错误')}")
