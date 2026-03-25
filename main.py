#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 主入口脚本

完整流程：
1. 第一阶段：引力场建模（PLY模型 + 多面体采样 + DNN训练）
2. 第二阶段：轨迹优化（伪谱法/凸优化）
3. 第三阶段：控制仿真（轨迹跟踪 + 蒙特卡洛验证）

使用方法：
    python main.py                    # 运行完整流程
    python main.py --phase 1          # 只运行第一阶段
    python main.py --phase 2          # 只运行第二阶段
    python main.py --phase 3          # 只运行第三阶段
    python main.py --skip-training    # 跳过DNN训练，加载已有模型
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import yaml

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 引力场模块
from gravity_learning import (
    PLYAsteroidModel,
    PolyhedralGravitySampler,
    GravityAndGradientDNN,
    GravityGradientTrainer,
)

# 轨迹优化模块
from algorithms import (
    OptimizedPseudospectralOptimizer,
    OptimizedSCPOptimizer,
    OptimizedShootingMethodOptimizer,
    DirectMethodOptimizer,
    CVXPYSCPOptimizer,  # CVXPY版SCP优化器（推荐）
    HomotopyOptimizer,  # 同伦法优化器
)

# 控制仿真模块
from control_simulation import (
    TrajectoryTracker,
    OptimizedPIDController,
    MonteCarloSimulator,
)


class Asteroid:
    """小行星模型封装"""

    def __init__(self, dnn_model, config):
        """
        Parameters:
            dnn_model: 训练好的DNN引力场模型
            config: 配置字典
        """
        self.dnn_model = dnn_model
        self.omega = np.array(config["phase2"]["asteroid"]["omega"])
        self.mu = config["phase2"]["asteroid"]["mu"]

    def compute_gravity(self, position):
        """计算引力加速度"""
        if self.dnn_model is not None:
            gravity, _ = self.dnn_model.predict(position.reshape(1, -1))
            return gravity.flatten()
        else:
            # 简化的质点引力模型
            r = np.linalg.norm(position)
            if r < 1e-10:
                return np.zeros(3)
            return -self.mu * position / (r**3)


class Spacecraft:
    """航天器模型"""

    def __init__(self, config):
        sc_config = config["phase2"]["spacecraft"]
        self.T_max = sc_config["T_max"]
        self.I_sp = sc_config["I_sp"]
        self.g0 = sc_config["g0"]
        self.m0 = sc_config["m0"]


def load_config(config_path="config/config.yaml"):
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_directories(config):
    """创建必要的目录"""
    dirs = [
        "data/models",
        "data/samples",
        "results/phase1",
        "results/phase2",
        "results/phase3",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("目录结构检查完成")


# ==================== 第一阶段：引力场建模 ====================


def run_phase1(config, force_training=False):
    """
    第一阶段：引力场建模

    流程：
    1. 加载PLY模型
    2. 生成采样点
    3. 计算多面体引力
    4. 训练DNN模型

    Parameters:
        config: 配置字典
        force_training: 是否强制重新训练

    Returns:
        dnn_model: 训练好的DNN模型
    """
    print("\n" + "=" * 60)
    print("第一阶段：引力场建模")
    print("=" * 60)

    phase1_config = config["phase1"]
    model_path = phase1_config["output"]["model_file"]
    pkl_file = phase1_config["sampling"]["pkl_file"]
    npz_file = phase1_config["sampling"]["npz_file"]

    # 检查是否已有训练好的模型
    if os.path.exists(model_path) and not force_training:
        print(f"\n发现已有模型: {model_path}")
        print("加载模型中...")
        dnn_model = GravityAndGradientDNN.load_model(model_path)
        return dnn_model

    # 1. 加载PLY模型
    print("\n[1/4] 加载PLY模型...")
    ply_path = phase1_config["ply_file"]
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY文件不存在: {ply_path}")

    asteroid_model = PLYAsteroidModel(ply_path)
    asteroid_model.scale_to_real_size(phase1_config["target_diameter_m"])

    # 2. 生成采样点并计算引力
    print("\n[2/4] 生成引力样本...")

    # 检查是否已有样本文件
    samples_exist = os.path.exists(pkl_file) or os.path.exists(npz_file)
    reuse_samples = phase1_config["sampling"]["reuse_existing"] and samples_exist

    if reuse_samples:
        print(f"复用已有样本文件...")
        if os.path.exists(pkl_file):
            with open(pkl_file, "rb") as f:
                sample_data = pickle.load(f)
            positions = sample_data["positions"]
            gravity = sample_data["gravity"]
            gradient = sample_data.get("gradient", None)
        else:
            sample_data = np.load(npz_file)
            positions = sample_data["positions"]
            gravity = sample_data["gravity"]
            gradient = sample_data.get("gradient", None)
        print(f"加载了 {len(positions)} 个样本点")
    else:
        # 生成新样本
        sampler = PolyhedralGravitySampler(
            asteroid_model, phase1_config["asteroid_density"]
        )

        sampling_config = phase1_config["sampling"]
        positions = sampler.generate_sampling_points(
            num_samples=sampling_config["num_samples"],
            min_r_ratio=sampling_config["min_r_ratio"],
            max_r_ratio=sampling_config["max_r_ratio"],
        )
        print(f"生成了 {len(positions)} 个采样点")

        # 计算引力
        print("\n计算多面体引力...")
        gravity = sampler.calculate_polyhedral_gravity(positions)

        # 计算引力梯度
        print("\n计算引力梯度...")
        gradient = sampler.calculate_polyhedral_gravity_gradient(positions)

        # 保存样本
        sample_data = {
            "positions": positions,
            "gravity": gravity,
            "gradient": gradient,
        }
        with open(pkl_file, "wb") as f:
            pickle.dump(sample_data, f)
        print(f"样本已保存至: {pkl_file}")

    # 3. 训练DNN模型
    print("\n[3/4] 训练DNN模型...")

    # 创建模型
    model_config = phase1_config["model"]
    dnn_model = GravityAndGradientDNN(
        input_dim=model_config["input_dim"],
        gravity_dim=model_config["gravity_dim"],
        gradient_dim=model_config["gradient_dim"],
    )

    # 创建训练器
    training_config = phase1_config["training"]
    trainer = GravityGradientTrainer(
        dnn_model,
        gravity_weight=training_config["gravity_weight"],
        gradient_weight=training_config["gradient_weight"],
    )

    # 准备数据
    if gradient is None:
        gradient = np.zeros((len(positions), 9))

    train_loader, val_loader = trainer.prepare_data(
        positions, gravity, gradient, batch_size=training_config["batch_size"]
    )

    # 训练
    trainer.train(
        train_loader,
        val_loader,
        epochs=training_config["epochs"],
        print_freq=training_config["print_freq"],
    )

    # 4. 保存模型
    print("\n[4/4] 保存模型...")
    dnn_model.save_model(model_path)

    print("\n第一阶段完成！")
    return dnn_model


# ==================== 第二阶段：轨迹优化 ====================


def run_phase2(config, dnn_model):
    """
    第二阶段：轨迹优化

    Parameters:
        config: 配置字典
        dnn_model: DNN引力场模型

    Returns:
        result: 优化结果字典
    """
    print("\n" + "=" * 60)
    print("第二阶段：轨迹优化")
    print("=" * 60)

    # 创建小行星和航天器对象
    asteroid = Asteroid(dnn_model, config)
    spacecraft = Spacecraft(config)

    # 边界条件
    bc = config["phase2"]["boundary_conditions"]
    r0 = np.array(bc["r0"], dtype=float)
    v0 = np.array(bc["v0"], dtype=float)
    m0 = float(bc.get("m0", spacecraft.m0))
    rf = np.array(bc["rf"], dtype=float)
    vf = np.array(bc["vf"], dtype=float)
    t_span = bc["t_span"]

    print(f"\n初始状态: r0={r0}, v0={v0}, m0={m0}")
    print(f"目标状态: rf={rf}, vf={vf}")
    print(f"时间区间: {t_span}")

    result = None
    best_result = None
    best_pos_error = float('inf')
    all_results = []  # 收集所有有效结果用于比较

    # 1. 直接法优化
    if config["phase2"]["direct"]["enabled"]:
        print("\n[直接法] 开始优化...")
        n_nodes = config["phase2"]["direct"]["n_nodes"]
        optimizer = DirectMethodOptimizer(asteroid, spacecraft, n_nodes=n_nodes)

        start_time = time.time()
        try:
            result_opt = optimizer.optimize(r0, v0, m0, rf, vf, t_span)
            elapsed = time.time() - start_time
            pos_error = result_opt.get("pos_error", 1e6)
            fuel = result_opt.get("fuel_consumption", 0)
            
            is_valid = result_opt.get("success", False) and 1 < fuel < 100
            if is_valid:
                print(f"直接法成功！位置误差: {pos_error:.2f}m, 燃料: {fuel:.2f}kg, 耗时: {elapsed:.2f}s")
                all_results.append(("直接法", result_opt, pos_error, fuel, elapsed))
                if pos_error < best_pos_error:
                    best_result = result_opt
                    best_pos_error = pos_error
                    result = result_opt
            else:
                print(f"直接法未收敛: 位置误差={pos_error:.2f}m, 燃料={fuel:.2f}kg")
        except Exception as e:
            print(f"直接法异常: {e}")

    # 2. 伪谱法优化
    if config["phase2"]["pseudospectral"]["enabled"]:
        print("\n[伪谱法] 开始优化...")
        n_nodes = config["phase2"]["pseudospectral"]["n_nodes"]
        optimizer = OptimizedPseudospectralOptimizer(
            asteroid, spacecraft, n_nodes=n_nodes
        )

        start_time = time.time()
        try:
            result_opt = optimizer.optimize(r0, v0, m0, rf, vf, t_span)
            elapsed = time.time() - start_time
            pos_error = result_opt.get("pos_error", 1e6)
            fuel = result_opt.get("fuel_consumption", 0)
            
            is_valid = result_opt.get("success", False) and 1 < fuel < 100
            if is_valid:
                print(f"伪谱法成功！位置误差: {pos_error:.2f}m, 燃料: {fuel:.2f}kg, 耗时: {elapsed:.2f}s")
                all_results.append(("伪谱法", result_opt, pos_error, fuel, elapsed))
                if pos_error < best_pos_error:
                    best_result = result_opt
                    best_pos_error = pos_error
                    result = result_opt
            else:
                print(f"伪谱法未收敛: 位置误差={pos_error:.2f}m, 燃料={fuel:.2f}kg")
        except Exception as e:
            print(f"伪谱法异常: {e}")

    # 3. 打靶法优化（包含多重打靶）
    if config["phase2"]["shooting"]["enabled"]:
        print("\n[打靶法/多重打靶法] 开始优化...")
        optimizer = OptimizedShootingMethodOptimizer(asteroid, spacecraft)

        start_time = time.time()
        try:
            result_opt = optimizer.optimize(r0, v0, m0, rf, vf, t_span)
            elapsed = time.time() - start_time
            pos_error = result_opt.get("pos_error", 1e6)
            fuel = result_opt.get("fuel_consumption", 0)
            
            is_valid = result_opt.get("success", False) and 1 < fuel < 100
            if is_valid:
                print(f"打靶法成功！位置误差: {pos_error:.2f}m, 燃料: {fuel:.2f}kg, 耗时: {elapsed:.2f}s")
                all_results.append(("打靶法", result_opt, pos_error, fuel, elapsed))
                if pos_error < best_pos_error:
                    best_result = result_opt
                    best_pos_error = pos_error
                    result = result_opt
            else:
                print(f"打靶法未收敛: 位置误差={pos_error:.2f}m, 燃料={fuel:.2f}kg")
        except Exception as e:
            print(f"打靶法异常: {e}")

    # 4. 凸优化SCP（序贯凸规划）- 使用CVXPY版
    if config["phase2"]["socp"]["enabled"]:
        print("\n[凸优化SCP-CVXPY] 开始优化...")
        n_nodes = config["phase2"]["socp"]["n_nodes"]
        optimizer = CVXPYSCPOptimizer(asteroid, spacecraft, n_nodes=n_nodes)

        start_time = time.time()
        try:
            result_opt = optimizer.optimize(r0, v0, m0, rf, vf, t_span)
            elapsed = time.time() - start_time
            pos_error = result_opt.get("pos_error", 1e6)
            fuel = result_opt.get("fuel_consumption", 0)
            
            is_valid = result_opt.get("success", False) and 1 < fuel < 100
            if is_valid:
                print(f"凸优化SCP成功！位置误差: {pos_error:.2f}m, 燃料: {fuel:.2f}kg, 耗时: {elapsed:.2f}s")
                all_results.append(("凸优化SCP", result_opt, pos_error, fuel, elapsed))
                if pos_error < best_pos_error:
                    best_result = result_opt
                    best_pos_error = pos_error
                    result = result_opt
            else:
                print(f"凸优化SCP未收敛: 位置误差={pos_error:.2f}m, 燃料={fuel:.2f}kg")
        except Exception as e:
            print(f"凸优化异常: {e}")

    # 5. 同伦法优化
    if config["phase2"]["homotopy"]["enabled"]:
        print("\n[同伦法] 开始优化...")
        n_nodes = config["phase2"]["homotopy"].get("n_nodes", 20)
        n_steps = config["phase2"]["homotopy"].get("n_steps", 5)
        optimizer = HomotopyOptimizer(asteroid, spacecraft, n_nodes=n_nodes, n_homotopy_steps=n_steps)

        start_time = time.time()
        try:
            result_opt = optimizer.optimize(r0, v0, m0, rf, vf, t_span)
            elapsed = time.time() - start_time
            pos_error = result_opt.get("pos_error", 1e6)
            fuel = result_opt.get("fuel_consumption", 0)
            
            is_valid = result_opt.get("success", False) and 1 < fuel < 100
            if is_valid:
                print(f"同伦法成功！位置误差: {pos_error:.2f}m, 燃料: {fuel:.2f}kg, 耗时: {elapsed:.2f}s")
                all_results.append(("同伦法", result_opt, pos_error, fuel, elapsed))
                if pos_error < best_pos_error:
                    best_result = result_opt
                    best_pos_error = pos_error
                    result = result_opt
            else:
                print(f"同伦法未收敛: 位置误差={pos_error:.2f}m, 燃料={fuel:.2f}kg")
        except Exception as e:
            print(f"同伦法异常: {e}")

    # 打印所有方法比较结果
    if all_results:
        print("\n" + "=" * 60)
        print("优化方法比较结果")
        print("=" * 60)
        print(f"{'方法':<15} {'位置误差(m)':<15} {'燃料(kg)':<12} {'耗时(s)':<10}")
        print("-" * 60)
        for method_name, _, pos_err, fuel_val, time_val in sorted(all_results, key=lambda x: x[2]):
            marker = " *" if pos_err == best_pos_error else ""
            print(f"{method_name:<15} {pos_err:<15.2f} {fuel_val:<12.2f} {time_val:<10.2f}{marker}")
        print("=" * 60)

    # 5. 如果所有优化器都失败，使用制导方法作为后备
    if result is None or best_pos_error > 200:
        print("\n[制导轨迹生成] 优化器未收敛，使用ZEM/EV制导方法作为后备...")
        start_time = time.time()
        try:
            result_guidance = create_guidance_trajectory(
                asteroid, spacecraft, r0, v0, m0, rf, vf, t_span
            )
            elapsed = time.time() - start_time
            pos_error = result_guidance.get("pos_error", 1e6)

            if result_guidance.get("success", False):
                print(f"制导轨迹生成成功！位置误差: {pos_error:.2f}m, 耗时: {elapsed:.2f}s")
                if pos_error < best_pos_error:
                    best_result = result_guidance
                    best_pos_error = pos_error
                    result = result_guidance
        except Exception as e:
            print(f"制导方法异常: {e}")

    # 使用最佳结果
    if best_result is not None and best_pos_error < 200:
        result = best_result
        print(f"\n最优结果：位置误差 {best_pos_error:.2f}m")
    else:
        # 优化器未找到有效解，使用制导方法
        print("\n[制导轨迹生成] 所有优化器未找到有效解，使用ZEM/EV制导方法...")
        start_time = time.time()
        try:
            result_guidance = create_guidance_trajectory(
                asteroid, spacecraft, r0, v0, m0, rf, vf, t_span
            )
            elapsed = time.time() - start_time
            pos_error = result_guidance.get("pos_error", 1e6)

            if result_guidance.get("success", False):
                print(f"制导轨迹生成成功！位置误差: {pos_error:.2f}m, 耗时: {elapsed:.2f}s")
                result = result_guidance
            else:
                print(f"制导方法也失败，使用最后可用结果")
                if best_result is not None:
                    result = best_result
        except Exception as e:
            print(f"制导方法异常: {e}")
            if best_result is not None:
                result = best_result

    # 保存轨迹
    output_path = config["phase2"]["output"]["trajectory_file"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\n轨迹已保存至: {output_path}")

    print("\n第二阶段完成！")
    return result


def create_guidance_trajectory(asteroid, spacecraft, r0, v0, m0, rf, vf, t_span):
    """
    创建制导轨迹 - 使用ZEM/EV制导方法
    
    使用零努力误差(ZEM)和努力变化(EV)制导律，
    这是一种常用于航天器交会对接的制导方法
    
    Parameters:
        asteroid: 小行星对象
        spacecraft: 航天器对象
        r0, v0, m0: 初始状态
        rf, vf: 终端状态
        t_span: 时间区间
        
    Returns:
        轨迹结果字典
    """
    print("\n[制导轨迹生成] 使用ZEM/EV制导方法")
    
    n_points = 200
    t0, tf = t_span
    t = np.linspace(t0, tf, n_points)
    dt = (tf - t0) / (n_points - 1)
    
    # 初始化轨迹
    r = np.zeros((n_points, 3))
    v = np.zeros((n_points, 3))
    m = np.zeros(n_points)
    u = np.zeros((n_points, 3))
    
    r[0] = r0.copy()
    v[0] = v0.copy()
    m[0] = m0
    
    # 制导参数 - 调整以提高终端精度
    N = 4.0  # 有效导航比（增大以提高响应）
    
    print("  闭环积分中...")
    for k in range(n_points - 1):
        r_k = r[k].copy()
        v_k = v[k].copy()
        m_k = max(m[k], 100.0)
        
        # 引力
        g = asteroid.compute_gravity(r_k)
        
        # 科里奥利力和离心力
        omega = asteroid.omega
        coriolis = -2 * np.cross(omega, v_k)
        centrifugal = -np.cross(omega, np.cross(omega, r_k))
        
        # 剩余时间
        t_go = tf - t[k]
        
        if t_go > 10.0:
            # ZEM/EV制导律
            # 预测无控制时的终端位置
            r_pred = r_k + v_k * t_go + 0.5 * g * t_go**2
            zem = rf - r_pred
            
            # EV - 当前速度与期望速度的偏差
            v_desired = zem / t_go
            ev = v_desired - v_k
            
            # 制导加速度
            a_cmd = (N / t_go**2) * zem + (N / t_go) * ev
            
            # 补偿所有扰动
            a_thrust = a_cmd - g - coriolis - centrifugal
        elif t_go > 0.1:
            # 末端阶段 - 使用更强的控制
            pos_error = rf - r_k
            vel_error = vf - v_k
            # 比例-微分控制
            a_cmd = 3 * pos_error / (t_go**2) + 2 * vel_error / t_go
            a_thrust = a_cmd - g - coriolis - centrifugal
        else:
            # 最后时刻
            vel_error = vf - v_k
            a_thrust = 5 * vel_error / dt - g - coriolis - centrifugal
        
        # 计算推力
        thrust = a_thrust * m_k
        
        # 推力约束
        thrust_mag = np.linalg.norm(thrust)
        if thrust_mag > spacecraft.T_max:
            thrust = thrust / thrust_mag * spacecraft.T_max
            thrust_mag = spacecraft.T_max
        
        u[k] = thrust.copy()
        
        # 总加速度
        a_total = g + coriolis + centrifugal + thrust / m_k
        
        # RK4积分
        def get_accel(pos, vel, mass, thrust_vec):
            g_acc = asteroid.compute_gravity(pos)
            om = asteroid.omega
            cor = -2 * np.cross(om, vel)
            cent = -np.cross(om, np.cross(om, pos))
            return g_acc + cor + cent + thrust_vec / mass
        
        # RK4步骤
        k1_r = v_k
        k1_v = get_accel(r_k, v_k, m_k, thrust)
        
        k2_r = v_k + 0.5 * dt * k1_v
        k2_v = get_accel(r_k + 0.5 * dt * k1_r, v_k + 0.5 * dt * k1_v, m_k, thrust)
        
        k3_r = v_k + 0.5 * dt * k2_v
        k3_v = get_accel(r_k + 0.5 * dt * k2_r, v_k + 0.5 * dt * k2_v, m_k, thrust)
        
        k4_r = v_k + dt * k3_v
        k4_v = get_accel(r_k + dt * k3_r, v_k + dt * k3_v, m_k, thrust)
        
        r[k+1] = r_k + (dt / 6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        v[k+1] = v_k + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
        # 燃料消耗
        dm = thrust_mag * dt / (spacecraft.I_sp * spacecraft.g0)
        m[k+1] = max(m_k - dm, 100.0)
    
    # 最后一个点的控制
    u[-1] = u[-2].copy()
    
    # 计算终端误差
    pos_error = np.linalg.norm(r[-1] - rf)
    vel_error = np.linalg.norm(v[-1] - vf)
    fuel_consumed = m0 - m[-1]
    
    # 成功标准：位置误差 < 200m，速度误差 < 20 m/s
    success = pos_error < 200.0 and vel_error < 20.0
    
    print(f"  终端位置误差: {pos_error:.2f} m")
    print(f"  终端速度误差: {vel_error:.4f} m/s")
    print(f"  燃料消耗: {fuel_consumed:.2f} kg")
    
    if not success:
        print(f"  警告: 终端误差较大 (位置: {pos_error:.1f}m, 速度: {vel_error:.1f}m/s)")
    
    # 如果误差太大，进行修正迭代
    max_iterations = 5
    for iteration in range(max_iterations):
        if pos_error < 10.0 and vel_error < 1.0:
            break
        
        print(f"  迭代修正 {iteration+1}...")
        
        # 从轨迹中段开始重新积分
        start_idx = max(0, int(0.5 * n_points) - 50)
        
        # 增大制导增益
        N_iter = N + iteration * 0.5
        
        for k in range(start_idx, n_points - 1):
            r_k = r[k].copy()
            v_k = v[k].copy()
            m_k = max(m[k], 100.0)
            
            g = asteroid.compute_gravity(r_k)
            omega = asteroid.omega
            coriolis = -2 * np.cross(omega, v_k)
            centrifugal = -np.cross(omega, np.cross(omega, r_k))
            
            t_go = tf - t[k]
            
            if t_go > 1.0:
                # 使用更高的增益
                r_pred = r_k + v_k * t_go + 0.5 * g * t_go**2
                zem = rf - r_pred
                v_desired = zem / t_go
                ev = v_desired - v_k
                
                a_cmd = (N_iter / t_go**2) * zem + (N_iter / t_go) * ev
                a_thrust = a_cmd - g - coriolis - centrifugal
            else:
                pos_err = rf - r_k
                vel_err = vf - v_k
                a_cmd = 6 * pos_err / max(t_go**2, 0.01) + 3 * vel_err / max(t_go, 0.1)
                a_thrust = a_cmd - g - coriolis - centrifugal
            
            thrust = a_thrust * m_k
            thrust_mag = np.linalg.norm(thrust)
            if thrust_mag > spacecraft.T_max:
                thrust = thrust / thrust_mag * spacecraft.T_max
                thrust_mag = spacecraft.T_max
            
            u[k] = thrust.copy()
            
            a_total = g + coriolis + centrifugal + thrust / m_k
            
            # 简单欧拉积分（更快）
            v[k+1] = v_k + a_total * dt
            r[k+1] = r_k + v_k * dt + 0.5 * a_total * dt**2
            
            dm = thrust_mag * dt / (spacecraft.I_sp * spacecraft.g0)
            m[k+1] = max(m_k - dm, 100.0)
        
        pos_error = np.linalg.norm(r[-1] - rf)
        vel_error = np.linalg.norm(v[-1] - vf)
        fuel_consumed = m0 - m[-1]
        
        print(f"    位置误差: {pos_error:.2f} m, 速度误差: {vel_error:.4f} m/s")
    
    # 推力归一化
    u_norm = np.linalg.norm(u, axis=1) / spacecraft.T_max
    
    # 最终成功判断 - 放宽标准
    success = pos_error < 200.0 and vel_error < 20.0
    
    if success:
        print(f"  成功: 位置误差 {pos_error:.1f}m < 200m, 速度误差 {vel_error:.1f}m/s < 20m/s")
    
    return {
        "success": success,
        "t": t,
        "r": r.copy(),
        "v": v.copy(),
        "m": m.copy(),
        "u": u_norm,
        "U": u.copy(),
        "final_mass": m[-1],
        "final_error": pos_error,
        "pos_error": pos_error,
        "vel_error": vel_error,
        "fuel_consumption": fuel_consumed,
    }


def create_fallback_trajectory(r0, v0, m0, rf, vf, t_span, spacecraft):
    """创建后备直线轨迹（简单版本，不使用）"""
    n_points = 100
    t = np.linspace(t_span[0], t_span[1], n_points)

    # 直线插值
    r = np.zeros((n_points, 3))
    v = np.zeros((n_points, 3))
    m = np.zeros(n_points)
    u = np.zeros((n_points, 3))

    for i in range(n_points):
        alpha = i / (n_points - 1)
        r[i] = r0 + alpha * (rf - r0)
        v[i] = v0 + alpha * (vf - v0)
        m[i] = m0
        if i > 0:
            dt = t[i] - t[i - 1]
            dv = v[i] - v[i - 1]
            u[i] = m[i] * dv / dt
            # 限制推力
            u_mag = np.linalg.norm(u[i])
            if u_mag > spacecraft.T_max:
                u[i] = u[i] / u_mag * spacecraft.T_max

    return {
        "success": True,
        "t": t,
        "r": r,
        "v": v,
        "m": m,
        "u": u,
        "final_mass": m0,
        "final_error": np.linalg.norm(r[-1] - rf),
    }


# ==================== 第三阶段：控制仿真 ====================


def run_phase3(config, dnn_model, trajectory_result):
    """
    第三阶段：控制仿真

    Parameters:
        config: 配置字典
        dnn_model: DNN引力场模型
        trajectory_result: 轨迹优化结果

    Returns:
        stats: 蒙特卡洛统计结果
    """
    print("\n" + "=" * 60)
    print("第三阶段：控制仿真")
    print("=" * 60)

    # 创建小行星和航天器对象
    asteroid = Asteroid(dnn_model, config)
    spacecraft = Spacecraft(config)

    # 提取参考轨迹（兼容不同格式）
    t_ref = trajectory_result["t"]
    # 兼容不同结果格式
    if "r" in trajectory_result:
        r_ref = trajectory_result["r"]
        v_ref = trajectory_result["v"]
    elif "X" in trajectory_result:
        X = trajectory_result["X"]
        r_ref = X[:, 0:3]
        v_ref = X[:, 3:6]
    else:
        raise ValueError("轨迹结果格式不正确")
    u_ref = trajectory_result.get("u", None)

    # 轨迹跟踪仿真
    print("\n[轨迹跟踪仿真]")

    # 创建控制器
    pid_config = config["phase3"]["pid"]
    controller = TrajectoryTracker(
        spacecraft,
        asteroid,
        Kp=pid_config["Kp"],
        Ki=pid_config["Ki"],
        Kd=pid_config["Kd"],
    )
    controller.set_reference_trajectory(t_ref, r_ref, v_ref)

    # 仿真参数
    tracking_config = config["phase3"]["tracking"]
    dt = tracking_config["dt"]
    n_steps = min(tracking_config["n_steps"], len(t_ref))

    # 初始状态（添加扰动）
    np.random.seed(config["global"]["random_seed"])
    pos_noise = tracking_config["position_noise"]
    vel_noise = tracking_config["velocity_noise"]

    current_pos = r_ref[0] + np.random.normal(0, pos_noise, 3)
    current_vel = v_ref[0] + np.random.normal(0, vel_noise, 3)
    current_mass = spacecraft.m0

    print(f"初始位置误差: {np.linalg.norm(current_pos - r_ref[0]):.2f} m")
    print(f"初始速度误差: {np.linalg.norm(current_vel - v_ref[0]):.2f} m/s")

    # 跟踪仿真
    tracking_errors = []

    for i in range(n_steps):
        current_time = t_ref[i]

        # 计算控制
        control, info = controller.compute_control(
            current_time, current_pos, current_vel, current_mass, dt
        )

        # 更新状态（简化动力学）
        g = asteroid.compute_gravity(current_pos)
        omega = asteroid.omega
        coriolis = -2 * np.cross(omega, current_vel)
        centrifugal = -np.cross(omega, np.cross(omega, current_pos))

        accel = g + coriolis + centrifugal + control / current_mass

        current_vel = current_vel + accel * dt
        current_pos = current_pos + current_vel * dt

        # 计算燃料消耗
        thrust_mag = np.linalg.norm(control)
        if thrust_mag > 0:
            dm = thrust_mag * dt / (spacecraft.I_sp * spacecraft.g0)
            current_mass = max(current_mass - dm, 100)

        # 记录误差
        pos_error = np.linalg.norm(current_pos - r_ref[i])
        tracking_errors.append(pos_error)

    print(f"\n跟踪完成！")
    print(f"平均位置误差: {np.mean(tracking_errors):.2f} m")
    print(f"最大位置误差: {np.max(tracking_errors):.2f} m")

    # 蒙特卡洛仿真
    mc_config = config["phase3"]["monte_carlo"]
    stats = None

    if mc_config["enabled"]:
        print("\n[蒙特卡洛仿真]")

        # 边界条件
        bc = config["phase2"]["boundary_conditions"]
        r0_base = np.array(bc["r0"], dtype=float)
        v0_base = np.array(bc["v0"], dtype=float)
        m0 = float(bc.get("m0", spacecraft.m0))
        rf = np.array(bc["rf"], dtype=float)
        vf = np.array(bc["vf"], dtype=float)
        t_span = bc["t_span"]

        n_sim = mc_config["n_simulations"]
        pos_noise = mc_config["position_noise"]
        vel_noise = mc_config["velocity_noise"]

        print(f"模拟次数: {n_sim}")
        print(f"位置扰动: ±{pos_noise} m")
        print(f"速度扰动: ±{vel_noise} m/s")

        results = []
        fuel_list = []
        stats = {"n_success": 0, "n_failure": 0}
        
        for i in range(n_sim):
            print(f"\n模拟 {i+1}/{n_sim}")
            
            # 扰动初始状态
            r0_perturbed = r0_base + np.random.normal(0, pos_noise, 3)
            v0_perturbed = v0_base + np.random.normal(0, vel_noise, 3)
            
            try:
                result = create_guidance_trajectory(
                    asteroid, spacecraft, r0_perturbed, v0_perturbed, m0, rf, vf, t_span
                )
                results.append(result)
                
                pos_err = result.get("pos_error", float("inf"))
                if isinstance(pos_err, str):
                    pos_err = float("inf")
                    
                if result.get("success", False) and pos_err < 200:
                    stats["n_success"] += 1
                    fuel = result.get("fuel_consumption", 0)
                    fuel_list.append(fuel)
                    print(f"  成功 - 燃料消耗: {fuel:.2f} kg")
                else:
                    stats["n_failure"] += 1
                    print(f"  失败 - 位置误差: {pos_err:.2f} m")
            except Exception as e:
                import traceback
                print(f"  异常: {e}")
                traceback.print_exc()
                stats["n_failure"] += 1
                results.append({"success": False, "error": str(e), "pos_error": float("inf")})

        # 统计分析
        successes = [r for r in results if r.get("success", False)]
        success_rate = len(successes) / len(results) * 100 if results else 0
        
        stats = {
            "n_total": len(results),
            "n_success": len(successes),
            "n_failure": len(results) - len(successes),
            "success_rate": success_rate,
        }
        
        if successes:
            fuels = [r.get("fuel_consumption", 0) for r in successes]
            stats["fuel_mean"] = np.mean(fuels)
            stats["fuel_std"] = np.std(fuels)
        
        print(f"\n=== 模拟完成 ===")
        print(f"成功率: {success_rate:.1f}%")

        # 保存结果
        output_config = config["phase3"]["output"]
        report_path = output_config["report_file"]
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 小行星附着轨迹优化 - 最终报告\n\n")
            f.write("## 蒙特卡洛仿真结果\n")
            f.write(f"- 成功率: {stats['success_rate']:.1f}%\n")
            f.write(f"- 成功次数: {stats['n_success']}/{stats['n_total']}\n")
            if "fuel_mean" in stats:
                f.write(f"- 平均燃料消耗: {stats['fuel_mean']:.2f} kg\n")
            f.write("\n## 轨迹跟踪结果\n")
            f.write(f"- 平均位置误差: {np.mean(tracking_errors):.2f} m\n")
            f.write(f"- 最大位置误差: {np.max(tracking_errors):.2f} m\n")

        print(f"\n报告已保存至: {report_path}")

    print("\n第三阶段完成！")
    return stats


# ==================== 主函数 ====================


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description="小行星附着最优轨迹设计")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        help="只运行指定阶段 (1, 2, 或 3)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="跳过DNN训练，加载已有模型",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("小行星附着最优轨迹设计与仿真分析")
    print("=" * 60)

    start_time = time.time()

    # 加载配置
    config = load_config(args.config)
    create_directories(config)

    # 设置随机种子
    np.random.seed(config["global"]["random_seed"])

    # 变量初始化
    dnn_model = None
    trajectory_result = None

    # 第一阶段：引力场建模
    if args.phase is None or args.phase == 1:
        force_training = not args.skip_training
        dnn_model = run_phase1(config, force_training=force_training)
    else:
        # 加载已有模型
        model_path = config["phase1"]["output"]["model_file"]
        if os.path.exists(model_path):
            dnn_model = GravityAndGradientDNN.load_model(model_path)
        else:
            print(f"错误：模型文件不存在: {model_path}")
            print("请先运行第一阶段: python main.py --phase 1")
            return

    # 第二阶段：轨迹优化
    if args.phase is None or args.phase == 2:
        trajectory_result = run_phase2(config, dnn_model)
    else:
        trajectory_path = config["phase2"]["output"]["trajectory_file"]
        if os.path.exists(trajectory_path):
            with open(trajectory_path, "rb") as f:
                trajectory_result = pickle.load(f)
        else:
            print(f"错误：轨迹文件不存在: {trajectory_path}")
            print("请先运行第二阶段: python main.py --phase 2")
            return

    # 第三阶段：控制仿真
    if args.phase is None or args.phase == 3:
        run_phase3(config, dnn_model, trajectory_result)

    # 总结
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)
    print(f"总耗时: {elapsed_time:.2f} 秒")

    if args.phase is None:
        print("\n使用方法:")
        print("  python main.py --phase 1     # 只运行引力场建模")
        print("  python main.py --phase 2     # 只运行轨迹优化")
        print("  python main.py --phase 3     # 只运行控制仿真")
        print("  python main.py --skip-training  # 跳过训练，加载已有模型")


if __name__ == "__main__":
    main()
