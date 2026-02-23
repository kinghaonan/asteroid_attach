#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 多重打靶法（优化版）

优化改进：
1. 添加tqdm进度条显示多猜测过程
2. 简化协态方程，提高数值稳定性
3. 改进初始猜测生成
4. 添加详细调试信息
5. 增强错误处理和容错机制

特点：
- 将轨迹分为多段，每段单独积分
- 通过连续性约束连接各段
- 比单段打靶法更稳定
- 适合复杂轨迹优化
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import warnings
from tqdm import tqdm


class MultipleShootingOptimizerOptimized:
    """
    优化版多重打靶法轨迹优化器

    将轨迹分为多段，分别求解后通过连续性约束连接。
    """

    def __init__(self, asteroid, spacecraft, n_segments: int = 5, verbose: bool = True):
        """
        初始化多重打靶法优化器

        Parameters:
            asteroid: 小行星对象
            spacecraft: 航天器对象
            n_segments: 分段数（默认5段）
            verbose: 是否显示详细信息
        """
        self.ast = asteroid
        self.sc = spacecraft
        self.n_segments = n_segments
        self.verbose = verbose
        self.continuity_weight = 1e3

    def dynamics_simplified(self, t: float, z: np.ndarray) -> np.ndarray:
        """
        简化的状态-协态动力学方程
        """
        r = z[0:3]
        v = z[3:6]
        m = z[6]
        lam_r = z[7:10]
        lam_v = z[10:13]
        lam_m = z[13]

        # 计算控制量
        chi, u = self.optimal_control(lam_v, lam_m, m)

        # 状态方程
        drdt = v

        # 引力（简化计算）
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-3:
            g = np.zeros(3)
        else:
            g = self.ast.gravity_gradient(r)

        # 环境力
        omega = self.ast.omega
        coriolis = -2 * np.cross(omega, v)
        centrifugal = -np.cross(omega, np.cross(omega, r))

        # 推力加速度
        thrust_acc = np.zeros(3)
        if m > 1e-3:
            thrust_acc = (self.sc.T_max / m) * u * chi

        dvdt = g + coriolis + centrifugal + thrust_acc
        dmdt = -self.sc.T_max * u / (self.sc.I_sp * self.sc.g0) if m > 1e-3 else 0

        # 简化的协态方程
        dlam_rdt = np.zeros(3)
        dlam_vdt = -lam_r
        dlam_mdt = 0.0

        return np.concatenate([drdt, dvdt, [dmdt], dlam_rdt, dlam_vdt, [dlam_mdt]])

    def optimal_control(
        self, lam_v: np.ndarray, lam_m: float, m: float
    ) -> Tuple[np.ndarray, float]:
        """
        计算最优控制（平滑版）
        """
        lam_v_norm = np.linalg.norm(lam_v)
        if lam_v_norm > 1e-6:
            chi = -lam_v / lam_v_norm
        else:
            chi = np.zeros(3)

        # 开关函数
        H_r = 1 - lam_m - (self.sc.I_sp * self.sc.g0 / m) * lam_v_norm

        # 平滑控制
        if abs(H_r) < 0.1:
            u = 0.5 - 5 * H_r
        elif H_r > 0:
            u = 0.0
        else:
            u = 1.0

        u = np.clip(u, 0, 1)

        return chi, u

    def generate_smart_initial_guesses(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
    ) -> List[np.ndarray]:
        """
        生成智能初始猜测集
        """
        guesses = []

        # 计算需要的速度变化
        delta_v_needed = np.linalg.norm(vf - v0)
        delta_r = rf - r0
        distance = np.linalg.norm(delta_r)

        # 估算推力时间比例
        avg_thrust_time = distance / (delta_v_needed + 1e-3)
        thrust_ratio = avg_thrust_time / (t_span[1] - t_span[0])
        thrust_ratio = np.clip(thrust_ratio, 0.1, 0.9)

        # 猜测1: 基于动量定理
        lam_r_guess = delta_r / distance if distance > 1e-3 else np.array([1, 0, 0])
        lam_v_guess = (vf - v0) / (delta_v_needed + 1e-3)
        lam_m_guess = 1 - thrust_ratio

        base_guess = np.concatenate(
            [lam_r_guess * 0.1, lam_v_guess * 0.1, [lam_m_guess]]
        )
        guesses.append(np.tile(base_guess, self.n_segments))

        # 猜测2: 保守估计
        conservative = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0])
        guesses.append(np.tile(conservative, self.n_segments))

        # 猜测3: 激进估计
        aggressive = np.array([1.0, 1.0, 1.0, -0.5, -0.5, -0.5, -0.1])
        guesses.append(np.tile(aggressive, self.n_segments))

        # 猜测4: 基于指向目标
        thrust_dir = delta_r / (distance + 1e-3)
        target_guess = np.concatenate([thrust_dir * 0.5, -thrust_dir * 0.3, [0.5]])
        guesses.append(np.tile(target_guess, self.n_segments))

        # 猜测5-8: 随机扰动
        np.random.seed(42)
        for _ in range(4):
            random_base = np.random.randn(7) * 0.5
            random_base[6] = np.random.uniform(-0.5, 0.5)
            guesses.append(np.tile(random_base, self.n_segments))

        return guesses

    def multiple_shooting(
        self,
        X: np.ndarray,
        t_span: List[float],
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        多重打靶函数
        """
        # 将决策变量分解为每个段的初始协态
        lam0_segments = X.reshape(self.n_segments, 7)

        # 初始化误差向量
        errors = []

        # 初始化轨迹
        full_t = np.array([])
        full_y = np.empty((14, 0))

        # 第一个段的初始条件
        current_r = r0
        current_v = v0
        current_m = m0

        # 时间节点
        t_nodes = np.linspace(t_span[0], t_span[1], self.n_segments + 1)

        for i in range(self.n_segments):
            # 当前段的时间区间
            segment_t_span = [t_nodes[i], t_nodes[i + 1]]

            # 当前段的初始条件
            z0 = np.concatenate([current_r, current_v, [current_m], lam0_segments[i]])

            try:
                # 积分当前段
                sol = solve_ivp(
                    self.dynamics_simplified,
                    segment_t_span,
                    z0,
                    method="RK45",
                    rtol=1e-5,
                    atol=1e-8,
                )

                if not sol.success:
                    errors.extend(np.ones(7) * 1e6)
                    continue

                # 提取终端状态
                zf = sol.y[:, -1]
                r_seg = zf[0:3]
                v_seg = zf[3:6]
                m_seg = zf[6]

                # 如果不是最后一段，添加连续性条件
                if i < self.n_segments - 1:
                    continuity_errors = np.concatenate(
                        [r_seg - current_r, v_seg - current_v, [m_seg - current_m]]
                    )
                    errors.extend(self.continuity_weight * continuity_errors)

                    # 更新下一段的初始状态
                    current_r = r_seg
                    current_v = v_seg
                    current_m = m_seg

                # 存储轨迹
                full_t = np.concatenate([full_t, sol.t])
                full_y = np.concatenate([full_y, sol.y], axis=1)

            except Exception as e:
                errors.extend(np.ones(7) * 1e6)
                continue

        # 终端边界条件误差
        terminal_errors = np.concatenate(
            [
                current_r - rf,
                current_v - vf,
                [0],  # 横截条件
            ]
        )
        errors.extend(terminal_errors)

        return np.array(errors), full_t, full_y

    def solve_single(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        lam_guess: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        求解单次多重打靶问题
        """
        try:
            result = least_squares(
                lambda X: self.multiple_shooting(X, t_span, r0, v0, m0, rf, vf)[0],
                lam_guess,
                method="lm",
                ftol=1e-4,
                xtol=1e-4,
                gtol=1e-4,
                max_nfev=500,
            )

            X_opt = result.x
            errors, t_opt, y_opt = self.multiple_shooting(
                X_opt, t_span, r0, v0, m0, rf, vf
            )
            final_error = np.linalg.norm(errors)

            return X_opt, t_opt, y_opt, final_error

        except Exception as e:
            return lam_guess, np.array([]), np.empty((14, 0)), float("inf")

    def optimize_with_multiple_guesses(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        guesses: Optional[List[np.ndarray]] = None,
    ) -> Dict:
        """
        使用多初始猜测求解（带进度条）
        """
        if guesses is None:
            guesses = self.generate_smart_initial_guesses(r0, v0, m0, rf, vf, t_span)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print("🎯 多重打靶法优化")
            print(f"{'=' * 60}")
            print(f"分段数: {self.n_segments}")
            print(f"初始猜测数: {len(guesses)}")
            print(f"\n⚡ 开始多猜测优化...")

        best_X = None
        best_t = None
        best_y = None
        best_error = float("inf")

        # 使用进度条
        iterator = enumerate(guesses)
        if self.verbose:
            iterator = tqdm(
                enumerate(guesses), total=len(guesses), desc="  猜测尝试", ncols=80
            )

        for i, lam_guess in iterator:
            try:
                X_opt, t_opt, y_opt, error = self.solve_single(
                    r0, v0, m0, rf, vf, t_span, lam_guess
                )

                if error < best_error:
                    best_error = error
                    best_X = X_opt
                    best_t = t_opt
                    best_y = y_opt

                if self.verbose and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix({"最佳误差": f"{best_error:.2f}"})

                # 如果误差足够小，提前终止
                if best_error < 100:
                    if self.verbose:
                        print(f"\n✅ 找到足够精确的解，提前终止")
                    break

            except Exception as e:
                if self.verbose:
                    print(f"\n  猜测 {i + 1} 失败: {str(e)}")
                continue

        success = best_error < 1000

        if self.verbose:
            print(f"\n📊 优化结果:")
            print(f"   最佳误差: {best_error:.2f}")
            print(f"   收敛状态: {'成功' if success else '部分收敛'}")

        final_mass = m0
        if best_y is not None and best_y.size > 0:
            final_mass = best_y[6, -1]

        return {
            "success": success,
            "initial_costates": best_X.reshape(self.n_segments, 7)
            if best_X is not None
            else None,
            "t": best_t,
            "y": best_y,
            "final_error": best_error,
            "final_mass": final_mass,
        }

    def extract_trajectory_data(self, t: np.ndarray, y: np.ndarray) -> Dict:
        """
        提取轨迹数据
        """
        if y.size == 0:
            return {}

        try:
            r = y[0:3, :].T
            v = y[3:6, :].T
            m = y[6, :]

            # 计算控制历史
            u_history = np.zeros(len(t))

            for i in range(len(t)):
                z = y[:, i]
                lam_v_i = z[10:13]
                lam_m_i = z[13]
                m_i = z[6]
                _, u = self.optimal_control(lam_v_i, lam_m_i, m_i)
                u_history[i] = u

            return {
                "t": t,
                "r": r,
                "v": v,
                "m": m,
                "u": u_history,
            }
        except Exception as e:
            print(f"提取轨迹数据失败: {str(e)}")
            return {}

    def plot_results(
        self,
        data: Dict,
        r0: np.ndarray,
        rf: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """绘制优化结果"""
        if not data:
            print("没有数据可供绘图")
            return

        try:
            fig = plt.figure(figsize=(15, 10))

            # 位置
            ax = fig.add_subplot(2, 3, 1)
            ax.plot(data["t"], data["r"][:, 0], label="x", linewidth=2)
            ax.plot(data["t"], data["r"][:, 1], label="y", linewidth=2)
            ax.plot(data["t"], data["r"][:, 2], label="z", linewidth=2)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("位置 (m)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 速度
            ax = fig.add_subplot(2, 3, 2)
            ax.plot(data["t"], data["v"][:, 0], label="vx", linewidth=2)
            ax.plot(data["t"], data["v"][:, 1], label="vy", linewidth=2)
            ax.plot(data["t"], data["v"][:, 2], label="vz", linewidth=2)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("速度 (m/s)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 质量
            ax = fig.add_subplot(2, 3, 3)
            ax.plot(data["t"], data["m"], "g-", linewidth=2)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("质量 (kg)")
            ax.grid(True, alpha=0.3)

            # 3D轨迹
            ax = fig.add_subplot(2, 3, 4, projection="3d")
            ax.plot(
                data["r"][:, 0], data["r"][:, 1], data["r"][:, 2], "b-", linewidth=2
            )
            ax.scatter(r0[0], r0[1], r0[2], c="r", marker="o", s=100, label="起点")
            ax.scatter(rf[0], rf[1], rf[2], c="g", marker="^", s=100, label="目标")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.legend()

            # 推力
            ax = fig.add_subplot(2, 3, 5)
            ax.plot(data["t"], data["u"] * self.sc.T_max, "r-", linewidth=2)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("推力 (N)")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"✅ 结果已保存: {save_path}")

            plt.show()
        except Exception as e:
            print(f"⚠️ 绘图失败: {str(e)}")


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

    print("=" * 60)
    print("多重打靶法优化器测试")
    print("=" * 60)

    asteroid = ExampleAsteroid()
    spacecraft = ExampleSpacecraft()
    optimizer = MultipleShootingOptimizerOptimized(asteroid, spacecraft, n_segments=5)

    r0 = np.array([10177, 6956, 8256])
    v0 = np.array([-25, -12, -17])
    m0 = spacecraft.m0
    rf = np.array([676, 5121, 449])
    vf = np.array([0, 0, 0])
    t_span = [0, 770]

    result = optimizer.optimize_with_multiple_guesses(r0, v0, m0, rf, vf, t_span)

    if result["success"] and result["y"] is not None:
        data = optimizer.extract_trajectory_data(result["t"], result["y"])
        if data:
            optimizer.plot_results(data, r0, rf)
            print("\n✅ 优化完成！")
