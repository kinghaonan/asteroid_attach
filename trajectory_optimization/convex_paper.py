#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Successive convexification trajectory optimization (paper-inspired).

Implements a simplified version of the convex SOCP approach in
Pinson & Lu (2018): fixed-time SOCP with lossless convexification
and successive linearization of gravity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import cvxpy as cp


@dataclass
class ConvexParams:
    dt: float = 5.0
    max_iterations: int = 8
    position_tolerance: float = 1.0
    fd_eps: float = 1e-3
    solver: str = "SCS"
    solver_max_iters: int = 5000
    solver_eps: float = 1e-5
    glide_slope_deg: float = 20.0
    vertical_window_s: float = 20.0
    vertical_eps: float = 2.0
    min_radius_margin: float = 0.0
    trust_radius_m: float = 1500.0
    min_radius_weight: float = 1000.0
    smooth_weight: float = 1.0
    max_delta_a: float = 0.2


def _finite_diff_jacobian(asteroid, r: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (A, c) such that g(r) ≈ A r + c."""
    g0 = asteroid.compute_gravity(r)
    A = np.zeros((3, 3))
    for i in range(3):
        dr = np.zeros(3)
        dr[i] = eps
        gp = asteroid.compute_gravity(r + dr)
        gm = asteroid.compute_gravity(r - dr)
        A[:, i] = (gp - gm) / (2.0 * eps)
    c = g0 - A @ r
    return A, c


def _linearize_gravity_along(asteroid, r_ref: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Linearize gravity at each node of reference trajectory."""
    Np1 = r_ref.shape[0]
    A_list = np.zeros((Np1, 3, 3))
    c_list = np.zeros((Np1, 3))
    for k in range(Np1):
        A, c = _finite_diff_jacobian(asteroid, r_ref[k], eps)
        A_list[k] = A
        c_list[k] = c
    return A_list, c_list


def _solve_fixed_time(
    asteroid,
    spacecraft,
    r0: np.ndarray,
    v0: np.ndarray,
    m0: float,
    rf: np.ndarray,
    vf: np.ndarray,
    t_span: List[float],
    params: ConvexParams,
    r_ref: Optional[np.ndarray] = None,
) -> Dict:
    t0, tf = t_span
    N = int(np.ceil((tf - t0) / params.dt))
    dt = (tf - t0) / N
    t_nodes = np.linspace(t0, tf, N + 1)

    if r_ref is None:
        r_ref = np.linspace(r0, rf, N + 1)

    center = getattr(asteroid, "center", None)
    radius = getattr(asteroid, "radius", None)
    if center is not None and radius is not None:
        center = np.asarray(center, dtype=float)
        radius = float(radius)
        for k in range(N + 1):
            vec = r_ref[k] - center
            dist = np.linalg.norm(vec)
            if dist < radius:
                if dist < 1e-9:
                    vec = rf - center
                    dist = np.linalg.norm(vec)
                    if dist < 1e-9:
                        vec = np.array([1.0, 0.0, 0.0])
                        dist = 1.0
                r_ref[k] = center + vec / dist * radius

    # Linearize gravity at reference
    A_list, c_list = _linearize_gravity_along(asteroid, r_ref, params.fd_eps)

    # Variables
    r = cp.Variable((N + 1, 3))
    v = cp.Variable((N + 1, 3))
    q = cp.Variable(N + 1)  # log mass
    a_t = cp.Variable((N, 3))  # thrust accel
    a_tm = cp.Variable(N)  # thrust accel magnitude slack

    constraints = []

    # Boundary conditions
    constraints += [r[0] == r0, v[0] == v0, q[0] == np.log(m0)]
    constraints += [r[-1] == rf, v[-1] == vf]

    # Dynamics (trapezoidal with linearized gravity at midpoints)
    omega = getattr(asteroid, "omega", np.zeros(3))
    for k in range(N):
        r_mid = 0.5 * (r[k] + r[k + 1])
        v_mid = 0.5 * (v[k] + v[k + 1])

        A_mid = 0.5 * (A_list[k] + A_list[k + 1])
        c_mid = 0.5 * (c_list[k] + c_list[k + 1])
        g_lin = A_mid @ r_mid + c_mid

        coriolis = -2 * cp.hstack([
            omega[1] * v_mid[2] - omega[2] * v_mid[1],
            omega[2] * v_mid[0] - omega[0] * v_mid[2],
            omega[0] * v_mid[1] - omega[1] * v_mid[0],
        ])
        # centrifugal = -omega x (omega x r)
        oxr = cp.hstack([
            omega[1] * r_mid[2] - omega[2] * r_mid[1],
            omega[2] * r_mid[0] - omega[0] * r_mid[2],
            omega[0] * r_mid[1] - omega[1] * r_mid[0],
        ])
        centrifugal = -cp.hstack([
            omega[1] * oxr[2] - omega[2] * oxr[1],
            omega[2] * oxr[0] - omega[0] * oxr[2],
            omega[0] * oxr[1] - omega[1] * oxr[0],
        ])

        constraints += [r[k + 1] == r[k] + dt * v_mid]
        constraints += [v[k + 1] == v[k] + dt * (a_t[k] + g_lin + coriolis + centrifugal)]

        # log-mass dynamics: q_{k+1} = q_k - dt * a_tm/(Isp*g0)
        constraints += [q[k + 1] == q[k] - dt * a_tm[k] / (spacecraft.I_sp * spacecraft.g0)]

    # Thrust magnitude SOC
    for k in range(N):
        constraints += [cp.norm(a_t[k]) <= a_tm[k]]

    # Thrust bounds with linearized exp(-q) about q0
    q0 = np.log(m0)
    T_max = spacecraft.T_max
    T_min = getattr(spacecraft, "T_min", 0.0)
    Tmax_exp = T_max * np.exp(-q0)
    Tmin_exp = T_min * np.exp(-q0)
    for k in range(N):
        constraints += [a_tm[k] <= Tmax_exp * (1 - (q[k] - q0))]
        constraints += [a_tm[k] >= Tmin_exp * (1 - (q[k] - q0))]

    # Mass lower bound
    m_dry = getattr(spacecraft, "m_dry", 0.8 * m0)
    constraints += [q >= np.log(m_dry)]

    # Glide slope constraint
    center = getattr(asteroid, "center", np.zeros(3))
    n_vec = rf - center
    n_norm = np.linalg.norm(n_vec)
    if n_norm > 1e-9:
        n_hat = n_vec / n_norm
        cos_theta = np.cos(np.deg2rad(params.glide_slope_deg))
        for k in range(N + 1):
            r_rel = r[k] - rf
            constraints += [cp.norm(r_rel) * cos_theta - r_rel @ n_hat <= 0]

        # Near-vertical constraint (last window)
        if params.vertical_window_s > 0:
            mask = t_nodes >= (tf - params.vertical_window_s)
            for k in range(N + 1):
                if mask[k]:
                    r_rel = r[k] - rf
                    dot = r_rel @ n_hat
                    r_perp = r_rel - dot * n_hat
                    constraints += [cp.norm(r_perp) <= params.vertical_eps]

    # Minimum radius constraint (linearized with reference, soft with slack)
    s_rad = None
    if center is not None and radius is not None:
        s_rad = cp.Variable(N + 1, nonneg=True)
        radius_req = radius + params.min_radius_margin
        for k in range(N + 1):
            vec = r_ref[k] - center
            dist = np.linalg.norm(vec)
            if dist < 1e-9:
                vec = rf - center
                dist = np.linalg.norm(vec)
                if dist < 1e-9:
                    vec = np.array([1.0, 0.0, 0.0])
                    dist = 1.0
            n_hat = vec / dist
            constraints += [n_hat @ (r[k] - center) + s_rad[k] >= radius_req]

    # Trust region around reference to prevent runaway
    if params.trust_radius_m and params.trust_radius_m > 0:
        for k in range(N + 1):
            constraints += [cp.norm(r[k] - r_ref[k]) <= params.trust_radius_m]

    # Smoothness constraints and penalties (control-friendly)
    if params.max_delta_a and params.max_delta_a > 0:
        for k in range(N - 1):
            constraints += [cp.norm(a_t[k + 1] - a_t[k]) <= params.max_delta_a]

    # Objective: maximize final mass => minimize -q[-1] + penalties
    obj_terms = [-q[-1]]
    if s_rad is not None:
        obj_terms.append(params.min_radius_weight * cp.sum(s_rad))
    if params.smooth_weight and params.smooth_weight > 0:
        obj_terms.append(params.smooth_weight * cp.sum_squares(a_t[1:] - a_t[:-1]))
    objective = cp.Minimize(cp.sum(obj_terms))
    problem = cp.Problem(objective, constraints)
    solver_name = str(params.solver).upper()
    if solver_name == "CLARABEL":
        # Clarabel option names vary by version; avoid passing unsupported options.
        problem.solve(
            solver=solver_name,
            verbose=True,
        )
    elif solver_name == "ECOS":
        problem.solve(
            solver=solver_name,
            max_iters=params.solver_max_iters,
            abstol=params.solver_eps,
            reltol=params.solver_eps,
            verbose=True,
        )
    else:
        problem.solve(
            solver=solver_name,
            max_iters=params.solver_max_iters,
            eps=params.solver_eps,
            verbose=True,
        )

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(f"SOCP failed: {problem.status}")

    r_sol = r.value
    v_sol = v.value
    q_sol = q.value
    m_sol = np.exp(q_sol)
    a_t_sol = a_t.value
    t_vec = t_nodes

    # Reconstruct thrust using midpoint mass
    m_mid = 0.5 * (m_sol[:-1] + m_sol[1:])
    U = a_t_sol * m_mid[:, None]

    return {
        "t": t_vec,
        "r": r_sol,
        "v": v_sol,
        "m": m_sol,
        "U": U,
        "fuel_consumption": m_sol[0] - m_sol[-1],
        "pos_error": float(np.linalg.norm(r_sol[-1] - rf)),
        "vel_error": float(np.linalg.norm(v_sol[-1] - vf)),
        "success": True,
    }


def solve_successive_convex(
    asteroid,
    spacecraft,
    r0: np.ndarray,
    v0: np.ndarray,
    m0: float,
    rf: np.ndarray,
    vf: np.ndarray,
    t_span: List[float],
    params: ConvexParams,
) -> Dict:
    # Initial reference
    N = int(np.ceil((t_span[1] - t_span[0]) / params.dt))
    r_ref = np.linspace(r0, rf, N + 1)

    # If min-radius margin is requested, allow first iteration to be relaxed
    relax_margin_once = params.min_radius_margin > 0.0

    for _ in range(params.max_iterations):
        attempt = 0
        while True:
            attempt += 1
            try:
                params_work = params
                if relax_margin_once:
                    params_work = ConvexParams(**{**params.__dict__, "min_radius_margin": 0.0})
                result = _solve_fixed_time(
                    asteroid, spacecraft, r0, v0, m0, rf, vf, t_span, params_work, r_ref=r_ref
                )
                break
            except RuntimeError as exc:
                if "infeasible" not in str(exc).lower() or attempt >= 3:
                    raise
                if relax_margin_once:
                    relax_margin_once = False
                    continue
                params = ConvexParams(**{**params.__dict__, "trust_radius_m": params.trust_radius_m * 2.0})
        r_new = result["r"]
        err = np.max(np.linalg.norm(r_new - r_ref, axis=1))
        r_ref = r_new
        if err < params.position_tolerance:
            break

    return result
