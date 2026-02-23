#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 约束处理模块

提供完整的轨迹约束处理功能：
1. 推力约束：幅值约束、方向约束、变化率约束
2. 路径约束：障碍物规避、安全走廊
3. 状态约束：速度限制、高度限制、质量约束
4. 终端约束：位置、速度、姿态精确约束
5. 约束软化和罚函数处理

参考：小天体附着多约束轨迹优化（第3章）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum


class ConstraintType(Enum):
    """约束类型枚举"""

    EQUALITY = "equality"  # 等式约束
    INEQUALITY = "inequality"  # 不等式约束
    BOUNDS = "bounds"  # 边界约束
    SOFT = "soft"  # 软约束


@dataclass
class ConstraintViolation:
    """约束违反信息"""

    constraint_name: str
    violation_value: float
    constraint_type: ConstraintType
    is_violated: bool


class ThrustConstraints:
    """
    推力约束处理器

    处理推力相关的各种约束：
    - 幅值约束: T_min <= |T| <= T_max
    - 方向约束: 推力方向限制
    - 变化率约束: |dT/dt| <= T_dot_max
    """

    def __init__(
        self,
        T_max: float,
        T_min: float = 0.0,
        T_dot_max: Optional[float] = None,
        max_steering_rate: Optional[float] = None,
    ):
        """
        初始化推力约束

        Parameters:
            T_max: 最大推力 (N)
            T_min: 最小推力 (N)，默认0
            T_dot_max: 最大推力变化率 (N/s)，默认None
            max_steering_rate: 最大方向变化率 (rad/s)，默认None
        """
        self.T_max = T_max
        self.T_min = T_min
        self.T_dot_max = T_dot_max
        self.max_steering_rate = max_steering_rate

    def check_magnitude(self, thrust: np.ndarray) -> ConstraintViolation:
        """
        检查推力幅值约束

        Parameters:
            thrust: 推力向量 [Tx, Ty, Tz]

        Returns:
            ConstraintViolation: 约束违反信息
        """
        T_mag = np.linalg.norm(thrust)

        if T_mag > self.T_max:
            violation = T_mag - self.T_max
            return ConstraintViolation(
                constraint_name="max_thrust",
                violation_value=violation,
                constraint_type=ConstraintType.INEQUALITY,
                is_violated=True,
            )
        elif T_mag < self.T_min and T_mag > 1e-10:
            violation = self.T_min - T_mag
            return ConstraintViolation(
                constraint_name="min_thrust",
                violation_value=violation,
                constraint_type=ConstraintType.INEQUALITY,
                is_violated=True,
            )

        return ConstraintViolation(
            constraint_name="thrust_magnitude",
            violation_value=0.0,
            constraint_type=ConstraintType.INEQUALITY,
            is_violated=False,
        )

    def check_rate_constraint(
        self, thrust_current: np.ndarray, thrust_prev: np.ndarray, dt: float
    ) -> ConstraintViolation:
        """
        检查推力变化率约束

        Parameters:
            thrust_current: 当前推力
            thrust_prev: 上一时刻推力
            dt: 时间步长

        Returns:
            ConstraintViolation: 约束违反信息
        """
        if self.T_dot_max is None:
            return ConstraintViolation(
                constraint_name="thrust_rate",
                violation_value=0.0,
                constraint_type=ConstraintType.INEQUALITY,
                is_violated=False,
            )

        T_dot = np.linalg.norm(thrust_current - thrust_prev) / dt

        if T_dot > self.T_dot_max:
            return ConstraintViolation(
                constraint_name="thrust_rate",
                violation_value=T_dot - self.T_dot_max,
                constraint_type=ConstraintType.INEQUALITY,
                is_violated=True,
            )

        return ConstraintViolation(
            constraint_name="thrust_rate",
            violation_value=0.0,
            constraint_type=ConstraintType.INEQUALITY,
            is_violated=False,
        )

    def project_to_feasible(self, thrust: np.ndarray) -> np.ndarray:
        """
        将推力投影到可行域

        Parameters:
            thrust: 推力向量

        Returns:
            np.ndarray: 投影后的推力
        """
        T_mag = np.linalg.norm(thrust)

        if T_mag > self.T_max:
            return thrust * (self.T_max / T_mag)
        elif T_mag < self.T_min and T_mag > 1e-10:
            return thrust * (self.T_min / T_mag)

        return thrust


class PathConstraints:
    """
    路径约束处理器

    处理轨迹路径相关的约束：
    - 障碍物规避：避开小行星表面、障碍物
    - 安全走廊：保持在安全区域内
    - 视线约束：保持对目标的可见性
    """

    def __init__(
        self,
        asteroid_radius: float,
        min_altitude: float = 100.0,
        max_altitude: Optional[float] = None,
        obstacle_list: Optional[List[Tuple[np.ndarray, float]]] = None,
    ):
        """
        初始化路径约束

        Parameters:
            asteroid_radius: 小行星半径 (m)
            min_altitude: 最小安全高度 (m)，默认100
            max_altitude: 最大高度限制 (m)，默认None
            obstacle_list: 障碍物列表 [(center, radius), ...]
        """
        self.asteroid_radius = asteroid_radius
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.obstacle_list = obstacle_list or []

    def check_altitude(self, position: np.ndarray) -> ConstraintViolation:
        """
        检查高度约束

        Parameters:
            position: 位置向量 [x, y, z]

        Returns:
            ConstraintViolation: 约束违反信息
        """
        r_norm = np.linalg.norm(position)
        altitude = r_norm - self.asteroid_radius

        if altitude < self.min_altitude:
            return ConstraintViolation(
                constraint_name="min_altitude",
                violation_value=self.min_altitude - altitude,
                constraint_type=ConstraintType.INEQUALITY,
                is_violated=True,
            )

        if self.max_altitude is not None and altitude > self.max_altitude:
            return ConstraintViolation(
                constraint_name="max_altitude",
                violation_value=altitude - self.max_altitude,
                constraint_type=ConstraintType.INEQUALITY,
                is_violated=True,
            )

        return ConstraintViolation(
            constraint_name="altitude",
            violation_value=0.0,
            constraint_type=ConstraintType.INEQUALITY,
            is_violated=False,
        )

    def check_obstacle_avoidance(
        self, position: np.ndarray
    ) -> List[ConstraintViolation]:
        """
        检查障碍物规避约束

        Parameters:
            position: 位置向量

        Returns:
            List[ConstraintViolation]: 约束违反信息列表
        """
        violations = []

        for i, (center, radius) in enumerate(self.obstacle_list):
            distance = np.linalg.norm(position - center)
            safety_margin = 50.0  # 安全余量

            if distance < radius + safety_margin:
                violations.append(
                    ConstraintViolation(
                        constraint_name=f"obstacle_{i}",
                        violation_value=radius + safety_margin - distance,
                        constraint_type=ConstraintType.INEQUALITY,
                        is_violated=True,
                    )
                )

        return violations

    def check_line_of_sight(
        self, position: np.ndarray, target: np.ndarray, asteroid_center: np.ndarray
    ) -> ConstraintViolation:
        """
        检查视线约束（避免小行星遮挡）

        Parameters:
            position: 当前位置
            target: 目标位置
            asteroid_center: 小行星中心

        Returns:
            ConstraintViolation: 约束违反信息
        """
        # 计算视线方向
        los = target - position
        los_norm = np.linalg.norm(los)

        if los_norm < 1e-10:
            return ConstraintViolation(
                constraint_name="line_of_sight",
                violation_value=0.0,
                constraint_type=ConstraintType.INEQUALITY,
                is_violated=False,
            )

        los_dir = los / los_norm

        # 计算到小行星中心的垂直距离
        to_center = asteroid_center - position
        projection = np.dot(to_center, los_dir)

        if projection > 0 and projection < los_norm:
            # 垂足在线段内
            perpendicular = to_center - projection * los_dir
            perp_dist = np.linalg.norm(perpendicular)

            if perp_dist < self.asteroid_radius:
                return ConstraintViolation(
                    constraint_name="line_of_sight",
                    violation_value=self.asteroid_radius - perp_dist,
                    constraint_type=ConstraintType.INEQUALITY,
                    is_violated=True,
                )

        return ConstraintViolation(
            constraint_name="line_of_sight",
            violation_value=0.0,
            constraint_type=ConstraintType.INEQUALITY,
            is_violated=False,
        )


class StateConstraints:
    """
    状态约束处理器

    处理航天器状态的约束：
    - 速度约束: |v| <= v_max
    - 质量约束: m >= m_dry
    - 角速度约束
    """

    def __init__(
        self,
        v_max: Optional[float] = None,
        m_dry: float = 0.0,
        omega_max: Optional[float] = None,
    ):
        """
        初始化状态约束

        Parameters:
            v_max: 最大速度 (m/s)，默认None
            m_dry: 干质量 (kg)，默认0
            omega_max: 最大角速度 (rad/s)，默认None
        """
        self.v_max = v_max
        self.m_dry = m_dry
        self.omega_max = omega_max

    def check_velocity(self, velocity: np.ndarray) -> ConstraintViolation:
        """检查速度约束"""
        if self.v_max is None:
            return ConstraintViolation(
                constraint_name="velocity",
                violation_value=0.0,
                constraint_type=ConstraintType.INEQUALITY,
                is_violated=False,
            )

        v_mag = np.linalg.norm(velocity)

        if v_mag > self.v_max:
            return ConstraintViolation(
                constraint_name="max_velocity",
                violation_value=v_mag - self.v_max,
                constraint_type=ConstraintType.INEQUALITY,
                is_violated=True,
            )

        return ConstraintViolation(
            constraint_name="velocity",
            violation_value=0.0,
            constraint_type=ConstraintType.INEQUALITY,
            is_violated=False,
        )

    def check_mass(self, mass: float) -> ConstraintViolation:
        """检查质量约束"""
        if mass < self.m_dry:
            return ConstraintViolation(
                constraint_name="min_mass",
                violation_value=self.m_dry - mass,
                constraint_type=ConstraintType.INEQUALITY,
                is_violated=True,
            )

        return ConstraintViolation(
            constraint_name="mass",
            violation_value=0.0,
            constraint_type=ConstraintType.INEQUALITY,
            is_violated=False,
        )


class TerminalConstraints:
    """
    终端约束处理器

    处理终端状态的精确约束：
    - 位置约束: r(tf) = rf
    - 速度约束: v(tf) = vf
    - 姿态约束（可选）
    """

    def __init__(
        self,
        position_tol: float = 100.0,
        velocity_tol: float = 1.0,
        attitude_tol: Optional[float] = None,
    ):
        """
        初始化终端约束

        Parameters:
            position_tol: 位置容差 (m)，默认100
            velocity_tol: 速度容差 (m/s)，默认1.0
            attitude_tol: 姿态容差 (deg)，默认None
        """
        self.position_tol = position_tol
        self.velocity_tol = velocity_tol
        self.attitude_tol = attitude_tol

    def check_position(
        self, position: np.ndarray, target: np.ndarray
    ) -> ConstraintViolation:
        """检查终端位置约束"""
        error = np.linalg.norm(position - target)

        if error > self.position_tol:
            return ConstraintViolation(
                constraint_name="terminal_position",
                violation_value=error - self.position_tol,
                constraint_type=ConstraintType.EQUALITY,
                is_violated=True,
            )

        return ConstraintViolation(
            constraint_name="terminal_position",
            violation_value=0.0,
            constraint_type=ConstraintType.EQUALITY,
            is_violated=False,
        )

    def check_velocity(
        self, velocity: np.ndarray, target: np.ndarray
    ) -> ConstraintViolation:
        """检查终端速度约束"""
        error = np.linalg.norm(velocity - target)

        if error > self.velocity_tol:
            return ConstraintViolation(
                constraint_name="terminal_velocity",
                violation_value=error - self.velocity_tol,
                constraint_type=ConstraintType.EQUALITY,
                is_violated=True,
            )

        return ConstraintViolation(
            constraint_name="terminal_velocity",
            violation_value=0.0,
            constraint_type=ConstraintType.EQUALITY,
            is_violated=False,
        )


class ConstraintManager:
    """
    约束管理器

    统一管理所有类型的约束，提供：
    - 约束检查
    - 罚函数计算
    - 约束违反统计
    - 可行性判断
    """

    def __init__(
        self,
        thrust_constraints: Optional[ThrustConstraints] = None,
        path_constraints: Optional[PathConstraints] = None,
        state_constraints: Optional[StateConstraints] = None,
        terminal_constraints: Optional[TerminalConstraints] = None,
        penalty_weights: Optional[Dict[str, float]] = None,
    ):
        """
        初始化约束管理器

        Parameters:
            thrust_constraints: 推力约束
            path_constraints: 路径约束
            state_constraints: 状态约束
            terminal_constraints: 终端约束
            penalty_weights: 罚函数权重
        """
        self.thrust = thrust_constraints
        self.path = path_constraints
        self.state = state_constraints
        self.terminal = terminal_constraints

        # 默认罚函数权重
        self.penalty_weights = penalty_weights or {
            "thrust": 1.0,
            "path": 10.0,
            "state": 5.0,
            "terminal": 100.0,
        }

    def check_all_constraints(
        self,
        state: np.ndarray,
        control: np.ndarray,
        time: float,
        is_terminal: bool = False,
        target_state: Optional[np.ndarray] = None,
    ) -> List[ConstraintViolation]:
        """
        检查所有约束

        Parameters:
            state: 状态向量 [r(3), v(3), m]
            control: 控制向量 [Tx, Ty, Tz] 或推力比
            time: 时间
            is_terminal: 是否为终端状态
            target_state: 目标状态（用于终端约束）

        Returns:
            List[ConstraintViolation]: 所有约束违反信息
        """
        violations = []

        r = state[0:3]
        v = state[3:6]
        m = state[6]

        # 推力约束
        if self.thrust is not None:
            violations.append(self.thrust.check_magnitude(control))

        # 路径约束
        if self.path is not None:
            violations.append(self.path.check_altitude(r))
            violations.extend(self.path.check_obstacle_avoidance(r))

        # 状态约束
        if self.state is not None:
            violations.append(self.state.check_velocity(v))
            violations.append(self.state.check_mass(m))

        # 终端约束
        if is_terminal and target_state is not None and self.terminal is not None:
            rf = target_state[0:3]
            vf = target_state[3:6]
            violations.append(self.terminal.check_position(r, rf))
            violations.append(self.terminal.check_velocity(v, vf))

        return violations

    def compute_penalty(self, violations: List[ConstraintViolation]) -> float:
        """
        计算罚函数值

        Parameters:
            violations: 约束违反信息列表

        Returns:
            float: 罚函数值
        """
        penalty = 0.0

        for v in violations:
            if not v.is_violated:
                continue

            weight = self.penalty_weights.get(v.constraint_name.split("_")[0], 1.0)

            if v.constraint_type == ConstraintType.EQUALITY:
                penalty += weight * v.violation_value**2
            else:
                penalty += weight * max(0, v.violation_value) ** 2

        return penalty

    def is_feasible(
        self,
        trajectory: np.ndarray,
        controls: np.ndarray,
        times: np.ndarray,
        tol: float = 1e-6,
    ) -> bool:
        """
        判断轨迹是否可行

        Parameters:
            trajectory: 轨迹状态 (N, 7)
            controls: 控制序列 (N, 3)
            times: 时间序列 (N,)
            tol: 容差

        Returns:
            bool: 是否可行
        """
        for i, (state, control, t) in enumerate(zip(trajectory, controls, times)):
            is_terminal = i == len(trajectory) - 1
            violations = self.check_all_constraints(state, control, t, is_terminal)

            for v in violations:
                if v.is_violated and v.violation_value > tol:
                    return False

        return True

    def get_violation_summary(self, violations: List[ConstraintViolation]) -> Dict:
        """
        获取约束违反统计

        Returns:
            Dict: 违反统计信息
        """
        summary = {
            "total_violations": 0,
            "equality_violations": 0,
            "inequality_violations": 0,
            "max_violation": 0.0,
            "violated_constraints": [],
        }

        for v in violations:
            if v.is_violated:
                summary["total_violations"] += 1
                summary["max_violation"] = max(
                    summary["max_violation"], v.violation_value
                )
                summary["violated_constraints"].append(v.constraint_name)

                if v.constraint_type == ConstraintType.EQUALITY:
                    summary["equality_violations"] += 1
                else:
                    summary["inequality_violations"] += 1

        return summary


# 示例使用
if __name__ == "__main__":
    print("=" * 70)
    print("轨迹约束处理模块示例")
    print("=" * 70)

    # 创建约束处理器
    thrust_con = ThrustConstraints(
        T_max=20.0,
        T_min=0.0,
        T_dot_max=5.0,
    )

    path_con = PathConstraints(
        asteroid_radius=8000.0,
        min_altitude=100.0,
        obstacle_list=[
            (np.array([1000, 0, 0]), 500.0),
        ],
    )

    state_con = StateConstraints(
        v_max=100.0,
        m_dry=800.0,
    )

    terminal_con = TerminalConstraints(
        position_tol=100.0,
        velocity_tol=1.0,
    )

    # 创建约束管理器
    manager = ConstraintManager(
        thrust_constraints=thrust_con,
        path_constraints=path_con,
        state_constraints=state_con,
        terminal_constraints=terminal_con,
    )

    # 测试状态
    state = np.array([8500.0, 0.0, 0.0, 50.0, 0.0, 0.0, 900.0])
    control = np.array([15.0, 0.0, 0.0])

    print("\n📋 约束检查")
    print("-" * 70)

    # 推力约束
    thrust_v = thrust_con.check_magnitude(control)
    print(f"推力约束: {'✅ 满足' if not thrust_v.is_violated else '❌ 违反'}")
    if thrust_v.is_violated:
        print(f"   违反值: {thrust_v.violation_value:.2f}")

    # 高度约束
    alt_v = path_con.check_altitude(state[0:3])
    print(f"高度约束: {'✅ 满足' if not alt_v.is_violated else '❌ 违反'}")
    if alt_v.is_violated:
        print(f"   违反值: {alt_v.violation_value:.2f}")

    # 速度约束
    vel_v = state_con.check_velocity(state[3:6])
    print(f"速度约束: {'✅ 满足' if not vel_v.is_violated else '❌ 违反'}")
    if vel_v.is_violated:
        print(f"   违反值: {vel_v.violation_value:.2f}")

    # 检查所有约束
    violations = manager.check_all_constraints(state, control, 0.0)
    penalty = manager.compute_penalty(violations)

    print(f"\n📊 罚函数值: {penalty:.4f}")
    print(f"约束违反数: {sum(1 for v in violations if v.is_violated)}")

    print("\n✅ 约束处理模块测试完成！")
