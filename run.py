#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asteroid landing pipeline (convex optimization + PID + Monte Carlo).
"""

import os
import sys
import time
import argparse
import pickle
from typing import Dict, List, Optional

import numpy as np
import yaml

from gravity_learning import (
    PLYAsteroidModel,
    PolyhedralGravitySampler,
    GravityAndGradientDNN,
    GravityGradientTrainer,
)
from trajectory_optimization.convex_paper import ConvexParams, solve_successive_convex
from control_simulation import TrajectoryTracker, TrackingMonteCarloSimulator, simulate_tracking

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str = "config/config.yaml") -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_directories(config: Dict) -> None:
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/samples", exist_ok=True)
    os.makedirs("results/phase1", exist_ok=True)
    os.makedirs("results/phase2", exist_ok=True)
    os.makedirs("results/phase3", exist_ok=True)


class Asteroid:
    def __init__(self, dnn_model, config: Dict):
        self.dnn_model = dnn_model
        self.omega = np.array(config["phase2"]["asteroid"]["omega"], dtype=float)
        self.mu = float(config["phase2"]["asteroid"]["mu"])
        self.center = None
        self.radius = None

    def compute_gravity(self, position: np.ndarray) -> np.ndarray:
        if self.dnn_model is not None:
            gravity, _ = self.dnn_model.predict(position.reshape(1, -1))
            return gravity.flatten()
        r = np.linalg.norm(position)
        if r < 1e-10:
            return np.zeros(3)
        return -self.mu * position / (r ** 3)


class Spacecraft:
    def __init__(self, config: Dict):
        sc_cfg = config["phase2"]["spacecraft"]
        self.T_max = float(sc_cfg["T_max"])
        self.I_sp = float(sc_cfg["I_sp"])
        self.g0 = float(sc_cfg["g0"])
        self.m0 = float(sc_cfg["m0"])


def attach_asteroid_geometry(asteroid: Asteroid, ply_model: Optional[PLYAsteroidModel]) -> None:
    if ply_model is None:
        return
    vertices = ply_model.vertices
    center = np.mean(vertices, axis=0)
    radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
    asteroid.center = center
    asteroid.radius = radius


def run_phase1(config: Dict, skip_training: bool = False):
    print("\n" + "=" * 70)
    print("Phase 1: gravity field modeling")
    print("=" * 70)

    phase1_cfg = config["phase1"]
    ply_model = PLYAsteroidModel(phase1_cfg["ply_file"])
    ply_model.scale_to_real_size(phase1_cfg["target_diameter_m"])

    model_path = phase1_cfg["output"]["model_file"]
    if skip_training and os.path.exists(model_path):
        print(f"[skip training] load model: {model_path}")
        dnn_model = GravityAndGradientDNN.load_model(model_path)
        return dnn_model, ply_model

    sampler = PolyhedralGravitySampler(ply_model, asteroid_density=phase1_cfg["asteroid_density"])
    pkl_file = phase1_cfg["sampling"]["pkl_file"]

    if phase1_cfg["sampling"].get("reuse_existing", False) and os.path.exists(pkl_file):
        samples = np.load(pkl_file, allow_pickle=True)
        positions = samples["positions"]
        gravity = samples["gravity"]
        gradient = samples.get("gradient", None)
    else:
        positions, gravity, gradient = sampler.generate_samples(
            n_samples=phase1_cfg["sampling"]["num_samples"],
            min_r_ratio=phase1_cfg["sampling"]["min_r_ratio"],
            max_r_ratio=phase1_cfg["sampling"]["max_r_ratio"],
        )
        np.savez(pkl_file, positions=positions, gravity=gravity, gradient=gradient)

    dnn_model = GravityAndGradientDNN()
    trainer = GravityGradientTrainer(dnn_model)
    training_cfg = phase1_cfg["training"]
    train_loader, val_loader = trainer.prepare_data(
        positions,
        gravity,
        gradient,
        batch_size=training_cfg.get("batch_size", 1024),
        val_split=training_cfg.get("val_split", 0.2),
    )
    trainer.train(
        train_loader,
        val_loader,
        epochs=training_cfg.get("epochs", 200),
        lr=training_cfg.get("learning_rate", 1e-3),
        save_path=model_path,
    )
    return dnn_model, ply_model


def run_phase2_convex(config: Dict, asteroid: Asteroid, spacecraft: Spacecraft) -> Dict:
    print("\n" + "=" * 70)
    print("Phase 2: convex trajectory optimization")
    print("=" * 70)

    bc = config["phase2"]["boundary_conditions"]
    r0 = np.array(bc["r0"], dtype=float)
    v0 = np.array(bc["v0"], dtype=float)
    rf = np.array(bc["rf"], dtype=float)
    vf = np.array(bc["vf"], dtype=float)
    t_span = bc["t_span"]

    cp_cfg = config.get("phase2", {}).get("convex_paper", {})
    params = ConvexParams(
        dt=float(cp_cfg.get("dt", 5.0)),
        max_iterations=int(cp_cfg.get("max_iterations", 8)),
        position_tolerance=float(cp_cfg.get("position_tolerance", 1.0)),
        fd_eps=float(cp_cfg.get("fd_eps", 1e-3)),
        solver=str(cp_cfg.get("solver", "ECOS")),
        solver_max_iters=int(cp_cfg.get("solver_max_iters", 5000)),
        solver_eps=float(cp_cfg.get("solver_eps", 1e-6)),
        glide_slope_deg=float(config["phase2"]["asteroid"].get("glide_slope_deg", 20.0)),
        vertical_window_s=float(config["phase2"]["asteroid"].get("vertical_window_s", 20.0)),
        vertical_eps=float(cp_cfg.get("vertical_eps", 2.0)),
        min_radius_margin=float(cp_cfg.get("min_radius_margin", 0.0)),
        trust_radius_m=float(cp_cfg.get("trust_radius_m", 3000.0)),
        min_radius_weight=float(cp_cfg.get("min_radius_weight", 1000.0)),
        smooth_weight=float(cp_cfg.get("smooth_weight", 5.0)),
        max_delta_a=float(cp_cfg.get("max_delta_a", 0.1)),
    )

    result = solve_successive_convex(
        asteroid, spacecraft, r0, v0, spacecraft.m0, rf, vf, t_span, params
    )
    result["method"] = "convex_paper"
    print(
        f"[ok] convex_paper: fuel={result['fuel_consumption']:.2f} kg, "
        f"pos_err={result['pos_error']:.4f} m, vel_err={result['vel_error']:.4f} m/s"
    )
    return result


def run_phase3_tracking(config: Dict, asteroid: Asteroid, spacecraft: Spacecraft, trajectory_result: Dict) -> Dict:
    print("\n" + "=" * 70)
    print("Phase 3: PID tracking + Monte Carlo")
    print("=" * 70)

    t_ref = trajectory_result["t"]
    r_ref = trajectory_result["r"]
    v_ref = trajectory_result.get("v", None)

    pid_cfg = config["phase3"]["pid"]
    tracker = TrajectoryTracker(
        spacecraft,
        asteroid,
        Kp=float(pid_cfg.get("Kp", 2.0)),
        Ki=float(pid_cfg.get("Ki", 0.2)),
        Kd=float(pid_cfg.get("Kd", 1.0)),
        use_feedforward=True,
    )
    # Build kinematic reference acceleration from v_ref or r_ref
    a_ref = None
    if v_ref is not None and len(t_ref) > 1:
        a_ref = np.gradient(v_ref, t_ref, axis=0)
    elif len(t_ref) > 2:
        v_tmp = np.gradient(r_ref, t_ref, axis=0)
        a_ref = np.gradient(v_tmp, t_ref, axis=0)

    # Diagnostics for reference trajectory sanity
    t_ref = np.array(t_ref, dtype=float)
    r_ref = np.array(r_ref, dtype=float)
    if v_ref is not None:
        v_ref = np.array(v_ref, dtype=float)

    if len(t_ref) > 1:
        dt_stats = np.diff(t_ref)
        print(f"[diag] t_ref: n={len(t_ref)}, dt_med={np.median(dt_stats):.4f}, dt_min={np.min(dt_stats):.4f}, dt_max={np.max(dt_stats):.4f}")
    r_norm = np.linalg.norm(r_ref, axis=1)
    print(f"[diag] r_ref: min={np.min(r_norm):.3f} m, max={np.max(r_norm):.3f} m")
    if v_ref is not None:
        v_norm = np.linalg.norm(v_ref, axis=1)
        print(f"[diag] v_ref: min={np.min(v_norm):.4f} m/s, max={np.max(v_norm):.4f} m/s")
    if a_ref is not None:
        a_norm = np.linalg.norm(a_ref, axis=1)
        print(f"[diag] a_ref: min={np.min(a_norm):.6f} m/s^2, max={np.max(a_norm):.6f} m/s^2")

    if a_ref is None:
        tracker.set_reference_trajectory(t_ref, r_ref, v_ref)
    else:
        tracker.set_reference_trajectory(t_ref, r_ref, v_ref, a_ref)

    bc = config["phase2"]["boundary_conditions"]
    r0 = np.array(bc["r0"], dtype=float)
    v0 = np.array(bc["v0"], dtype=float)
    m0 = spacecraft.m0
    t_span = bc["t_span"]

    tracking_cfg = config["phase3"]["tracking"]
    dt_cfg = float(tracking_cfg.get("dt", 1.0))
    if len(t_ref) > 1:
        dt_ref = float(np.median(np.diff(t_ref)))
        dt = min(dt_cfg, dt_ref)
    else:
        dt = dt_cfg

    tracking_result = simulate_tracking(
        tracker, asteroid, spacecraft, r0, v0, m0, t_span, dt
    )

    eval_cfg = config["phase3"]["evaluation"]
    max_pos_err = float(eval_cfg.get("max_position_error", 1.0))
    max_vel_err = float(eval_cfg.get("max_velocity_error", 1.0))

    tracking_success = (tracking_result["pos_error"] <= max_pos_err) and (
        tracking_result["vel_error"] <= max_vel_err
    )

    mc_cfg = config["phase3"]["monte_carlo"]
    mc_results = None
    if mc_cfg.get("enabled", True):
        mc_sim = TrackingMonteCarloSimulator(
            tracker,
            asteroid,
            spacecraft,
            n_simulations=int(mc_cfg.get("n_simulations", 20)),
        )
        mc_results = mc_sim.run_monte_carlo(
            r0,
            v0,
            m0,
            t_span,
            dt,
            position_noise=float(mc_cfg.get("position_noise", 20.0)),
            velocity_noise=float(mc_cfg.get("velocity_noise", 1.0)),
            max_position_error=max_pos_err,
            max_velocity_error=max_vel_err,
        )

    print(
        f"[ok] tracking: pos_err={tracking_result['pos_error']:.4f} m, "
        f"vel_err={tracking_result['vel_error']:.4f} m/s, success={tracking_success}"
    )
    if mc_results is not None:
        print(
            f"[ok] monte_carlo: success_rate={mc_results['success_rate']:.1f}%, "
            f"pos_err_max={mc_results['pos_error_max']:.4f} m, "
            f"vel_err_max={mc_results['vel_error_max']:.4f} m/s"
        )

    output_cfg = config["phase3"]["output"]
    results_path = output_cfg.get("results_file", "results/phase3/control_simulation.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(
            {
                "tracking": tracking_result,
                "tracking_success": tracking_success,
                "monte_carlo": mc_results,
            },
            f,
        )

    return {
        "tracking": tracking_result,
        "tracking_success": tracking_success,
        "monte_carlo": mc_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Asteroid landing pipeline (convex + PID + MC)")
    parser.add_argument("--skip-training", action="store_true", help="skip DNN training")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="run a single phase")
    args = parser.parse_args()

    print("=" * 70)
    print("Asteroid landing pipeline - convex only")
    print("=" * 70)

    start_time = time.time()
    config = load_config()
    create_directories(config)

    spacecraft = Spacecraft(config)

    if args.phase is None or args.phase == 1:
        dnn_model, ply_model = run_phase1(config, skip_training=args.skip_training)
        asteroid = Asteroid(dnn_model, config)
    else:
        dnn_model = GravityAndGradientDNN.load_model(config["phase1"]["output"]["model_file"])
        asteroid = Asteroid(dnn_model, config)
        ply_model = None

    attach_asteroid_geometry(asteroid, ply_model)

    trajectory_result = None
    if args.phase is None or args.phase == 2:
        trajectory_result = run_phase2_convex(config, asteroid, spacecraft)

    if args.phase is None or args.phase == 3:
        if trajectory_result is None:
            print("ERROR: phase 3 needs phase 2 trajectory")
            return
        run_phase3_tracking(config, asteroid, spacecraft, trajectory_result)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)
    print(f"Total time: {total_time:.2f} s")

    if trajectory_result:
        print("\nTrajectory result:")
        print("  method: convex_paper")
        print(f"  fuel: {trajectory_result['fuel_consumption']:.2f} kg")
        print(f"  pos_err: {trajectory_result['pos_error']:.4f} m")
        print(f"  vel_err: {trajectory_result['vel_error']:.4f} m/s")


if __name__ == "__main__":
    main()

