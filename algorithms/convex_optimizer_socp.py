#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 凸优化方法（SOCP）

基于NASA论文: "Trajectory Design Employing Convex Optimization for Landing
on Irregularly Shaped Asteroids" by Robin M. Pinson and Ping Lu

核心方法:
1. Lossless Convexification - 无损凸化，引入松弛变量Tm
2. 变量变换: at = T/m (推力加速度), q = ln(m) (对数质量)
3. 逐次求解法 (Successive Solution Method) - 线性化非线性引力
4. 二阶锥规划 (SOCP) - 凸优化问题求解

优点:
- 无需初始猜测
- 全局收敛
- 可靠高效
- 适合星载计算
"""

import numpy as np
import cvxpy as cp
from typing import Dict, Tuple, List, Optional
import time
from tqdm import tqdm


class SOCPTrajectoryOptimizer:
    """
    基于SOCP的凸优化轨迹规划器

    使用二阶锥规划求解燃料最优着陆问题。
    实现了论文中的无损凸化和逐次求解方法。

    Attributes:
        asteroid: 小行星模型
        spacecraft: 航天器参数
        n_nodes: 离散节点数
        dt: 时间步长
        verbose: 详细输出模式
    """

    def __init__(self, asteroid, spacecraft, n_nodes: int = 50, verbose: bool = True):
        """
        初始化凸优化器

        Parameters:
            asteroid: 小行星对象，需提供 gravity_gradient(r) 和 omega 属性
            spacecraft: 航天器对象，需提供 T_max, T_min, I_sp, g0, m0 属性
            n_nodes: 离散节点数
            verbose: 是否显示详细信息
        """
        self.ast = asteroid
        self.sc = spacecraft
        self.n_nodes = n_nodes
        self.verbose = verbose

        self.T_max = spacecraft.T_max
        self.T_min = getattr(spacecraft, "T_min", 0.0)
        self.I_sp = spacecraft.I_sp
        self.g0 = spacecraft.g0
        self.m0 = spacecraft.m0
        self.v_ex = self.I_sp * self.g0

        self.omega = np.array(asteroid.omega)
        self.mu = getattr(asteroid, "mu", 4.463e5)

        self.scaling = None
        self.convergence_tol = 0.5
        self.max_iterations = 20

    def setup_scaling(self, r0: np.ndarray, rf: np.ndarray):
        """
        设置变量缩放因子

        论文建议使用最小半轴长度作为距离缩放因子，
        保持缩放后的位置不小于1，避免数值问题。
        """
        R_sc = min(
            np.linalg.norm(r0),
            np.linalg.norm(rf),
            np.abs(r0).min() + 1e-6,
            np.abs(rf).min() + 1e-6,
        )
        R_sc = max(R_sc, 500.0)

        g_sc = self.mu / R_sc**2
        v_sc = np.sqrt(R_sc * g_sc)
        t_sc = np.sqrt(R_sc / g_sc)

        self.scaling = {
            "R_sc": R_sc,
            "g_sc": g_sc,
            "v_sc": v_sc,
            "t_sc": t_sc,
            "T_sc": self.T_max,
            "m_sc": self.m0,
        }

    def compute_gravity_terms(self, r: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        计算引力项：主项(dom)和高阶项

        将引力分解为主项（牛顿引力）和高阶项，
        主项放入矩阵A，高阶项放入向量c

        Parameters:
            r: 位置向量

        Returns:
            dom: 主项标量（用于矩阵A）
            higher_order: 高阶项向量（用于向量c）
        """
        r_norm = np.linalg.norm(r)

        if r_norm < 1e-3:
            return 0.0, np.zeros(3)

        dom = -self.mu / r_norm**3

        try:
            g_full = self.ast.gravity_gradient(r)
            g_newton = dom * r
            higher_order = g_full - g_newton
        except:
            higher_order = np.zeros(3)

        return dom, higher_order

    def build_linearized_dynamics(
        self, r_ref: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        构建线性化动力学矩阵

        基于参考轨迹r_ref，构建线性时变系统：
        dx/dt = A(r_ref) * x + B * u + c(r_ref)

        Parameters:
            r_ref: 参考位置轨迹 (n_nodes, 3)
            dt: 时间步长

        Returns:
            A_matrices: 状态矩阵列表
            B_matrix: 控制矩阵
            c_vectors: 常数向量列表
        """
        A_matrices = []
        c_vectors = []

        omega = self.omega
        omega_z = omega[2] if len(omega) > 2 else 0.0

        for i in range(self.n_nodes):
            r = r_ref[i]
            dom, higher_order = self.compute_gravity_terms(r)

            omega_sq = omega_z**2

            A = np.array(
                [
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [omega_sq + dom, 0, 0, 0, 2 * omega_z, 0, 0],
                    [0, omega_sq + dom, 0, -2 * omega_z, 0, 0, 0],
                    [0, 0, dom, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            )
            A_matrices.append(A)

            c = np.zeros(7)
            c[3:6] = higher_order
            c_vectors.append(c)

        B = np.zeros((7, 4))
        B[3:6, 0:3] = np.eye(3)
        B[6, 3] = -1.0 / self.v_ex

        return A_matrices, B, c_vectors

    def solve_socp_problem(
        self,
        r_ref: np.ndarray,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        tf: float,
        q0: float,
    ) -> Optional[Dict]:
        """
        求解单次SOCP问题

        基于参考轨迹r_ref，构建并求解凸优化问题

        Parameters:
            r_ref: 参考位置轨迹
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            tf: 飞行时间
            q0: 初始对数质量

        Returns:
            优化结果字典，失败返回None
        """
        dt = tf / (self.n_nodes - 1)

        A_matrices, B, c_vectors = self.build_linearized_dynamics(r_ref, dt)

        r = cp.Variable((self.n_nodes, 3))
        v = cp.Variable((self.n_nodes, 3))
        q = cp.Variable(self.n_nodes)
        at = cp.Variable((self.n_nodes, 3))
        atm = cp.Variable(self.n_nodes)

        constraints = []

        constraints.append(r[0] == r0)
        constraints.append(v[0] == v0)
        constraints.append(q[0] == q0)

        constraints.append(r[-1] == rf)
        constraints.append(v[-1] == vf)

        q_dry = np.log(0.1 * self.m0)
        constraints.append(q >= q_dry)

        for i in range(self.n_nodes - 1):
            A = A_matrices[i]
            c = c_vectors[i]
            A_next = A_matrices[i + 1]
            c_next = c_vectors[i + 1]

            x_i = cp.hstack([r[i], v[i], q[i]])
            x_ip1 = cp.hstack([r[i + 1], v[i + 1], q[i + 1]])
            u_i = cp.hstack([at[i], atm[i]])
            u_ip1 = cp.hstack([at[i + 1], atm[i + 1]])

            rhs_i = A @ x_i + B @ u_i + c
            rhs_ip1 = A_next @ x_ip1 + B @ u_ip1 + c_next

            dynamics = x_ip1 - x_i - 0.5 * dt * (rhs_i + rhs_ip1)
            constraints.append(dynamics == np.zeros(7))

        for i in range(self.n_nodes):
            constraints.append(cp.norm(at[i]) <= atm[i])

            q_ref = q0 - 0.3 * (i / self.n_nodes)
            T_min_ratio = self.T_min / self.m0 * np.exp(-q_ref)
            T_max_ratio = self.T_max / self.m0 * np.exp(-q_ref)

            constraints.append(atm[i] >= T_min_ratio * (1 - (q[i] - q_ref)))
            constraints.append(atm[i] <= T_max_ratio * (1 - (q[i] - q_ref)))
            constraints.append(atm[i] >= 0)
            constraints.append(atm[i] <= self.T_max / (0.1 * self.m0))

        objective = cp.Maximize(q[-1])

        problem = cp.Problem(objective, constraints)

        solvers = [cp.SCS, cp.CLARABEL, cp.ECOS]
        last_error = None

        for solver in solvers:
            try:
                problem.solve(solver=solver, verbose=False)

                if problem.status in ["optimal", "optimal_inaccurate"]:
                    return {
                        "r": r.value,
                        "v": v.value,
                        "q": q.value,
                        "at": at.value,
                        "atm": atm.value,
                        "status": problem.status,
                    }
                else:
                    last_error = f"Status: {problem.status}"
            except Exception as e:
                last_error = str(e)
                continue

        return None

    def optimize(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        max_iter: int = 15,
    ) -> Dict:
        """
        执行凸优化轨迹规划

        使用逐次求解方法，迭代求解SOCP问题

        Parameters:
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_span: 时间区间 [t0, tf]
            max_iter: 最大迭代次数

        Returns:
            优化结果字典
        """
        if self.verbose:
            print(f"\n{'=' * 60}")
            print("📐 凸优化轨迹规划 (SOCP)")
            print(f"{'=' * 60}")
            print(f"节点数: {self.n_nodes}")
            print(f"最大迭代: {max_iter}")

        self.setup_scaling(r0, rf)

        t0, tf = t_span
        dt = (tf - t0) / (self.n_nodes - 1)
        q0 = np.log(m0)

        r_ref = np.zeros((self.n_nodes, 3))
        for i in range(3):
            r_ref[:, i] = np.linspace(r0[i], rf[i], self.n_nodes)

        best_result = None
        best_fuel = float("inf")
        prev_r = r_ref.copy()

        start_time = time.time()

        iterator = range(max_iter)
        if self.verbose:
            iterator = tqdm(iterator, desc="  逐次求解", ncols=80)

        for iteration in iterator:
            result = self.solve_socp_problem(r_ref, r0, v0, m0, rf, vf, tf, q0)

            if result is None:
                if self.verbose:
                    print(f"\n迭代 {iteration + 1}: SOCP求解失败")
                continue

            r_new = result["r"]

            diff = np.max(np.abs(r_new - prev_r))

            m_final = np.exp(result["q"][-1])
            fuel = m0 - m_final

            if fuel > 0 and fuel < m0 * 0.9 and fuel < best_fuel:
                best_fuel = fuel
                best_result = result

            if diff < self.convergence_tol:
                if self.verbose:
                    print(f"\n✅ 收敛于迭代 {iteration + 1}, 误差: {diff:.4f} m")
                break

            prev_r = r_new.copy()
            r_ref = r_new

        end_time = time.time()

        if best_result is None:
            if self.verbose:
                print("\n⚠️ 凸优化未能找到可行解")
            return {"success": False, "message": "SOCP未能收敛到可行解"}

        m_final = np.exp(best_result["q"][-1])
        fuel_consumption = m0 - m_final

        pos_err = np.linalg.norm(best_result["r"][-1] - rf)
        vel_err = np.linalg.norm(best_result["v"][-1] - vf)

        t_nodes = np.linspace(t0, tf, self.n_nodes)
        m_nodes = np.exp(best_result["q"])

        u_norm = np.linalg.norm(best_result["at"], axis=1)
        u_ratio = np.clip(u_norm / (best_result["atm"] + 1e-10), 0, 1)

        if self.verbose:
            print(f"\n⏱️ 优化耗时: {end_time - start_time:.2f}秒")
            print(f"\n📊 优化结果:")
            print(f"   燃料消耗: {fuel_consumption:.2f} kg")
            print(f"   终端质量: {m_final:.2f} kg")
            print(f"   位置误差: {pos_err:.4f} m")
            print(f"   速度误差: {vel_err:.4f} m/s")

        return {
            "success": True,
            "t": t_nodes,
            "r": best_result["r"],
            "v": best_result["v"],
            "m": m_nodes,
            "u": u_ratio,
            "at": best_result["at"],
            "atm": best_result["atm"],
            "fuel_consumption": fuel_consumption,
            "pos_error": pos_err,
            "vel_error": vel_err,
            "iterations": iteration + 1,
            "time": end_time - start_time,
        }

    def optimize_flight_time(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_range: Tuple[float, float] = (300, 1500),
        n_candidates: int = 5,
    ) -> Dict:
        """
        优化飞行时间

        使用候选点搜索方法找到最优飞行时间

        Parameters:
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_range: 飞行时间搜索范围
            n_candidates: 候选点数量

        Returns:
            最优结果字典
        """
        if self.verbose:
            print(f"\n{'=' * 60}")
            print("🎯 飞行时间优化")
            print(f"{'=' * 60}")
            print(f"搜索范围: {t_range[0]} - {t_range[1]} s")
            print(f"候选点数: {n_candidates}")

        tf_candidates = np.linspace(t_range[0], t_range[1], n_candidates)

        best_result = None
        best_fuel = float("inf")
        best_tf = t_range[0]

        for tf in tf_candidates:
            result = self.optimize(r0, v0, m0, rf, vf, [0, tf], max_iter=10)

            if result.get("success", False):
                fuel = result["fuel_consumption"]
                if 0 < fuel < best_fuel:
                    best_fuel = fuel
                    best_result = result
                    best_tf = tf

        if best_result is None:
            if self.verbose:
                print("\n⚠️ 未能找到可行解")
            return self.optimize(r0, v0, m0, rf, vf, [0, t_range[1]])

        if self.verbose:
            print(f"\n✅ 最优飞行时间: {best_tf:.1f} s")
            print(f"   最优燃料消耗: {best_fuel:.2f} kg")

        return best_result


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
            self.T_min = 0.0
            self.I_sp = 400.0
            self.g0 = 9.81
            self.m0 = 1000.0

    print("=" * 60)
    print("凸优化轨迹规划器测试 (SOCP)")
    print("=" * 60)

    asteroid = ExampleAsteroid()
    spacecraft = ExampleSpacecraft()
    optimizer = SOCPTrajectoryOptimizer(asteroid, spacecraft, n_nodes=30)

    r0 = np.array([10177, 6956, 8256])
    v0 = np.array([-25, -12, -17])
    m0 = spacecraft.m0
    rf = np.array([676, 5121, 449])
    vf = np.array([0, 0, 0])
    t_span = [0, 770]

    result = optimizer.optimize(r0, v0, m0, rf, vf, t_span)

    if result["success"]:
        print("\n✅ 优化成功！")
    else:
        print("\n❌ 优化失败")
