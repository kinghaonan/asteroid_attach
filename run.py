#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 完整流程

流程：
1. 第一阶段：引力场建模（PLY模型 + DNN训练）
2. 第二阶段：轨迹优化（多种优化方法）
3. 第三阶段：控制仿真（轨迹跟踪 + 蒙特卡洛验证）

使用方法：
    python run.py                    # 运行完整流程
    python run.py --skip-training    # 跳过DNN训练
    python run.py --phase 2          # 只运行第二阶段
    python run.py --method homotopy  # 指定优化方法
"""

import os
import sys
import time
import argparse
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 延迟导入模块（避免numpy/torch兼容性问题）
_PLYAsteroidModel = None
_PolyhedralGravitySampler = None
_GravityAndGradientDNN = None
_GravityGradientTrainer = None
_DirectMethodOptimizer = None
_CVXPYSCPOptimizer = None
_FastPseudospectralOptimizer = None
_FastShootingOptimizer = None
_FastHomotopyOptimizer = None


def _import_gravity_learning():
    """延迟导入引力场模块"""
    global _PLYAsteroidModel, _PolyhedralGravitySampler, _GravityAndGradientDNN, _GravityGradientTrainer
    if _PLYAsteroidModel is None:
        from gravity_learning import (
            PLYAsteroidModel,
            PolyhedralGravitySampler,
            GravityAndGradientDNN,
            GravityGradientTrainer,
        )
        _PLYAsteroidModel = PLYAsteroidModel
        _PolyhedralGravitySampler = PolyhedralGravitySampler
        _GravityAndGradientDNN = GravityAndGradientDNN
        _GravityGradientTrainer = GravityGradientTrainer
    return _PLYAsteroidModel, _PolyhedralGravitySampler, _GravityAndGradientDNN, _GravityGradientTrainer


def _import_algorithms():
    """延迟导入算法模块"""
    global _DirectMethodOptimizer, _CVXPYSCPOptimizer, _FastPseudospectralOptimizer, _FastShootingOptimizer, _FastHomotopyOptimizer
    if _DirectMethodOptimizer is None:
        from algorithms import (
            DirectMethodOptimizer,
            CVXPYSCPOptimizer,
            FastPseudospectralOptimizer,
            FastShootingOptimizer,
            FastHomotopyOptimizer,
        )
        _DirectMethodOptimizer = DirectMethodOptimizer
        _CVXPYSCPOptimizer = CVXPYSCPOptimizer
        _FastPseudospectralOptimizer = FastPseudospectralOptimizer
        _FastShootingOptimizer = FastShootingOptimizer
        _FastHomotopyOptimizer = FastHomotopyOptimizer
    return _DirectMethodOptimizer, _CVXPYSCPOptimizer, _FastPseudospectralOptimizer, _FastShootingOptimizer, _FastHomotopyOptimizer


class Asteroid:
    """小行星模型封装"""
    
    def __init__(self, dnn_model, config):
        self.dnn_model = dnn_model
        self.omega = np.array(config["phase2"]["asteroid"]["omega"])
        self.mu = config["phase2"]["asteroid"]["mu"]
    
    def compute_gravity(self, position):
        """计算引力加速度"""
        if self.dnn_model is not None:
            gravity, _ = self.dnn_model.predict(position.reshape(1, -1))
            return gravity.flatten()
        else:
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
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/samples", exist_ok=True)
    os.makedirs("results/phase1", exist_ok=True)
    os.makedirs("results/phase2", exist_ok=True)
    os.makedirs("results/phase3", exist_ok=True)


# ==================== 第一阶段：引力场建模 ====================

def run_phase1(config, skip_training=False):
    """
    第一阶段：引力场建模
    
    流程：
    1. 加载PLY模型
    2. 生成引力样本（多面体法）
    3. 训练DNN模型
    """
    # 延迟导入
    PLYAsteroidModel, PolyhedralGravitySampler, GravityAndGradientDNN, GravityGradientTrainer = _import_gravity_learning()
    
    print("\n" + "="*70)
    print("第一阶段：引力场建模")
    print("="*70)
    
    phase1_config = config["phase1"]
    
    # 1. 加载PLY模型
    print("\n[1/3] 加载PLY模型...")
    ply_model = PLYAsteroidModel(phase1_config["ply_file"])
    ply_model.scale_to_real_size(phase1_config["target_diameter_m"])
    
    print(f"  顶点数: {len(ply_model.vertices)}")
    print(f"  面数: {len(ply_model.faces)}")
    
    # 2. 检查是否已有训练好的模型
    model_path = phase1_config["output"]["model_file"]
    
    if skip_training and os.path.exists(model_path):
        print(f"\n[跳过训练] 加载已有模型: {model_path}")
        dnn_model = GravityAndGradientDNN.load_model(model_path)
        return dnn_model, ply_model
    
    # 3. 生成引力样本
    print("\n[2/3] 生成引力样本...")
    sampler = PolyhedralGravitySampler(ply_model, asteroid_density=phase1_config["asteroid_density"])
    
    # 检查是否已有样本
    pkl_file = phase1_config["sampling"]["pkl_file"]
    npz_file = phase1_config["sampling"]["npz_file"]
    
    if phase1_config["sampling"]["reuse_existing"] and os.path.exists(pkl_file):
        print(f"  加载已有样本: {pkl_file}")
        samples = np.load(pkl_file, allow_pickle=True)
        positions = samples["positions"]
        gravity = samples["gravity"]
        if "gradient" in samples:
            gradient = samples["gradient"]
        else:
            gradient = None
    else:
        print(f"  生成新样本...")
        positions, gravity, gradient = sampler.generate_samples(
            n_samples=phase1_config["sampling"]["num_samples"],
            min_r_ratio=phase1_config["sampling"]["min_r_ratio"],
            max_r_ratio=phase1_config["sampling"]["max_r_ratio"],
        )
        # 保存样本
        np.savez(pkl_file, positions=positions, gravity=gravity, gradient=gradient)
        print(f"  样本已保存: {pkl_file}")
    
    print(f"  样本数: {len(positions)}")
    
    # 4. 训练DNN模型
    print("\n[3/3] 训练DNN模型...")
    dnn_model = GravityAndGradientDNN()
    trainer = GravityGradientTrainer(dnn_model)
    
    training_config = phase1_config["training"]
    train_loader, val_loader = trainer.prepare_data(
        positions, gravity, gradient, 
        batch_size=training_config["batch_size"]
    )
    history = trainer.train(
        train_loader, val_loader,
        epochs=training_config["epochs"],
    )
    
    # 保存模型
    dnn_model.save_model(model_path)
    print(f"  模型已保存: {model_path}")
    
    return trainer.model, ply_model


# ==================== 第二阶段：轨迹优化 ====================

def run_phase2(config, asteroid, spacecraft, method="homotopy"):
    """
    第二阶段：轨迹优化
    
    支持方法：
    - direct: 直接法
    - cvxpy: CVXPY凸优化
    - pseudospectral: 快速伪谱法
    - shooting: 快速打靶法
    - homotopy: 快速同伦法（推荐）
    """
    # 延迟导入
    DirectMethodOptimizer, CVXPYSCPOptimizer, FastPseudospectralOptimizer, FastShootingOptimizer, FastHomotopyOptimizer = _import_algorithms()
    
    print("\n" + "="*70)
    print("第二阶段：轨迹优化")
    print("="*70)
    
    phase2_config = config["phase2"]
    
    # 问题设置
    bc = phase2_config["boundary_conditions"]
    r0 = np.array(bc["r0"])
    v0 = np.array(bc["v0"])
    m0 = spacecraft.m0
    rf = np.array(bc["rf"])
    vf = np.array(bc["vf"])
    t_span = bc["t_span"]
    
    print(f"\n问题设置:")
    print(f"  初始位置: {r0} m")
    print(f"  初始速度: {v0} m/s")
    print(f"  初始质量: {m0} kg")
    print(f"  目标位置: {rf} m")
    print(f"  目标速度: {vf} m/s")
    print(f"  飞行时间: {t_span[1]} s")
    print(f"  优化方法: {method}")
    
    # 创建优化器
    optimizers = {
        "direct": lambda: DirectMethodOptimizer(asteroid, spacecraft, n_nodes=20),
        "cvxpy": lambda: CVXPYSCPOptimizer(asteroid, spacecraft, n_nodes=20, max_iterations=50, verbose=False),
        "pseudospectral": lambda: FastPseudospectralOptimizer(asteroid, spacecraft, n_nodes=25, verbose=False),
        "shooting": lambda: FastShootingOptimizer(asteroid, spacecraft, n_nodes=25, verbose=False),
        "homotopy": lambda: FastHomotopyOptimizer(asteroid, spacecraft, n_nodes=25, n_homotopy_steps=3, verbose=False),
    }
    
    if method not in optimizers:
        print(f"错误: 未知方法 '{method}'")
        print(f"可用方法: {list(optimizers.keys())}")
        return None
    
    # 执行优化
    print(f"\n开始优化...")
    start_time = time.time()
    
    optimizer = optimizers[method]()
    result = optimizer.optimize(r0, v0, m0, rf, vf, t_span)
    
    elapsed = time.time() - start_time
    
    # 显示结果
    print("\n" + "-"*50)
    print("优化结果:")
    print("-"*50)
    
    if result and 'r' in result:
        pos_err = np.linalg.norm(result['r'][-1] - rf)
        vel_err = np.linalg.norm(result['v'][-1] - vf)
        fuel = m0 - result['m'][-1]
        success = result.get('success', pos_err < 10 and vel_err < 5)
        
        print(f"  成功: {success}")
        print(f"  位置误差: {pos_err:.4f} m")
        print(f"  速度误差: {vel_err:.4f} m/s")
        print(f"  燃料消耗: {fuel:.2f} kg")
        print(f"  计算时间: {elapsed:.2f} s")
        
        # 保存结果
        result['method'] = method
        result['elapsed_time'] = elapsed
        result['success'] = success
        
        return result
    else:
        print(f"  优化失败")
        return None


# ==================== 第三阶段：控制与仿真 ====================

def run_phase3(config, asteroid, spacecraft, trajectory_result):
    """
    第三阶段：控制与仿真
    
    流程：
    1. 轨迹跟踪控制器
    2. 闭环仿真
    3. 蒙特卡洛验证
    """
    print("\n" + "="*70)
    print("第三阶段：控制与仿真")
    print("="*70)
    
    if trajectory_result is None:
        print("错误: 无有效轨迹")
        return None
    
    phase3_config = config.get("phase3", {})
    
    # 提取参考轨迹
    t_ref = trajectory_result['t']
    r_ref = trajectory_result['r']
    v_ref = trajectory_result['v']
    m_ref = trajectory_result['m']
    
    # 插值加密轨迹（控制需要更密集的时间步）
    t0, tf = t_ref[0], t_ref[-1]
    dt_control = 1.0  # 控制时间步长 1秒
    n_control = int((tf - t0) / dt_control) + 1
    t_control = np.linspace(t0, tf, n_control)
    
    # 线性插值位置、速度、质量
    from scipy.interpolate import interp1d
    interp_r = interp1d(t_ref, r_ref, axis=0, kind='linear', fill_value='extrapolate')
    interp_v = interp1d(t_ref, v_ref, axis=0, kind='linear', fill_value='extrapolate')
    interp_m = interp1d(t_ref, m_ref, axis=0, kind='linear', fill_value='extrapolate')
    
    r_ref_fine = interp_r(t_control)
    v_ref_fine = interp_v(t_control)
    m_ref_fine = interp_m(t_control)
    
    print(f"  轨迹插值: {len(t_ref)} -> {len(t_control)} 节点 (dt={dt_control}s)")
    
    # 提取或生成控制序列（前馈控制）
    if 'U' in trajectory_result:
        u_ref_orig = trajectory_result['U']
    elif 'u' in trajectory_result:
        u_norm = trajectory_result['u']
        u_ref_orig = np.zeros((len(u_norm), 3))
        for i in range(len(u_norm) - 1):
            if i < len(v_ref) - 1:
                dv = v_ref[i+1] - v_ref[i]
                dt = t_ref[i+1] - t_ref[i] if i < len(t_ref) - 1 else 1.0
                if dt > 0:
                    a = dv / dt
                    m_avg = (m_ref[i] + m_ref[min(i+1, len(m_ref)-1)]) / 2
                    u_ref_orig[i] = a * m_avg
                    u_mag = np.linalg.norm(u_ref_orig[i])
                    if u_mag > spacecraft.T_max:
                        u_ref_orig[i] = u_ref_orig[i] / u_mag * spacecraft.T_max
    else:
        u_ref_orig = np.zeros((len(t_ref), 3))
    
    # 插值控制序列
    interp_u = interp1d(t_ref, u_ref_orig, axis=0, kind='linear', fill_value='extrapolate')
    u_ref_fine = interp_u(t_control)
    
    # 控制参数
    # 对于大尺度问题（距离~10km，质量1000kg，推力20N）
    # 最大加速度约0.02 m/s²，需要精心设计控制策略
    max_accel = spacecraft.T_max / m_ref[0]  # 约0.02 m/s²
    
    print(f"\n[1/3] 创建轨迹跟踪控制器...")
    print(f"  控制策略: 前馈控制 + PD跟踪 + 终端制导")
    print(f"  巡航: Kp=0.5, Kd=30 (t>500s)")
    print(f"  中段: Kp=1.0, Kd=50 (100s<t<500s)")
    print(f"  接近: Kp=2.0, Kd=90 (t<100s)")
    print(f"  终端: ZEM-ZEV制导 (t<20s)")
    print(f"  最大加速度: {max_accel:.4f} m/s²")
    
    def state_derivative(r, v, m, control):
        """状态导数"""
        g = asteroid.compute_gravity(r)
        omega = asteroid.omega
        coriolis = -2 * np.cross(omega, v)
        centrifugal = -np.cross(omega, np.cross(omega, r))
        a_total = g + coriolis + centrifugal + control / m
        return a_total
    
    def rk4_step(r, v, m, control, dt):
        """RK4积分"""
        k1_v = state_derivative(r, v, m, control)
        k1_r = v
        
        k2_v = state_derivative(r + 0.5*dt*k1_r, v + 0.5*dt*k1_v, m, control)
        k2_r = v + 0.5*dt*k1_v
        
        k3_v = state_derivative(r + 0.5*dt*k2_r, v + 0.5*dt*k2_v, m, control)
        k3_r = v + 0.5*dt*k2_v
        
        k4_v = state_derivative(r + dt*k3_r, v + dt*k3_v, m, control)
        k4_r = v + dt*k3_v
        
        v_new = v + dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        r_new = r + dt/6 * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        
        return r_new, v_new
    
    def run_closed_loop(r0, v0, m0, r_ref, v_ref, m_ref, u_ref, t_ref):
        """闭环仿真 - 轨迹跟踪 + 终端制导"""
        n_steps = len(t_ref)
        r_sim = [r0.copy()]
        v_sim = [v0.copy()]
        m_sim = [m0]
        
        r_curr, v_curr, m_curr = r0.copy(), v0.copy(), m0
        
        # 目标状态
        r_target = r_ref[-1]
        v_target = v_ref[-1]
        
        for i in range(n_steps - 1):
            dt_i = t_ref[i+1] - t_ref[i]
            t_remain = t_ref[-1] - t_ref[i]
            
            # 前馈控制
            u_ff = u_ref[i] if i < len(u_ref) else np.zeros(3)
            
            if t_remain < 15:  # 最后15秒：纯速度修正
                v_err = v_target - v_curr
                
                # 直接速度修正
                a_cmd = v_err / max(t_remain, 0.5)
                
                # 限制加速度
                a_max = spacecraft.T_max / m_curr
                a_mag = np.linalg.norm(a_cmd)
                if a_mag > a_max:
                    a_cmd = a_cmd / a_mag * a_max
                
                control = m_curr * a_cmd
            elif t_remain < 50:  # 50s内：ZEM-ZEV制导
                r_err = r_target - r_curr
                v_err = v_target - v_curr
                
                # ZEM-ZEV制导律 - 平衡位置和速度
                t_go = max(t_remain, 1.0)
                a_cmd = 4 * r_err / (t_go**2) + 3 * v_err / t_go
                
                # 限制加速度
                a_max = spacecraft.T_max / m_curr
                a_mag = np.linalg.norm(a_cmd)
                if a_mag > a_max:
                    a_cmd = a_cmd / a_mag * a_max
                
                control = m_curr * a_cmd
            else:  # 轨迹跟踪 + 反馈
                r_err = r_ref[i] - r_curr
                v_err = v_ref[i] - v_curr
                
                # PD控制
                if t_remain < 100:
                    Kp, Kd = 2.0, 90.0
                elif t_remain < 500:
                    Kp, Kd = 1.0, 50.0
                else:
                    Kp, Kd = 0.5, 30.0
                
                u_fb = Kp * r_err + Kd * v_err
                control = u_ff + u_fb
            
            # 限制推力
            u_mag = np.linalg.norm(control)
            if u_mag > spacecraft.T_max:
                control = control / u_mag * spacecraft.T_max
            
            # RK4积分
            r_curr, v_curr = rk4_step(r_curr, v_curr, m_curr, control, dt_i)
            
            # 燃料消耗
            thrust_mag = np.linalg.norm(control)
            if thrust_mag > 0:
                m_curr = m_curr - thrust_mag * dt_i / (spacecraft.I_sp * spacecraft.g0)
                m_curr = max(m_curr, 0.3 * m0)
            
            r_sim.append(r_curr.copy())
            v_sim.append(v_curr.copy())
            m_sim.append(m_curr)
        
        return np.array(r_sim), np.array(v_sim), np.array(m_sim)
    
    # 2. 闭环仿真
    print("\n[2/3] 闭环仿真...")
    tracking_config = phase3_config.get("tracking", {})
    
    # 初始状态（添加小扰动）
    np.random.seed(42)
    pos_noise = tracking_config.get("position_noise", 20.0)
    vel_noise = tracking_config.get("velocity_noise", 1.0)
    r0_perturbed = r_ref_fine[0] + np.random.randn(3) * pos_noise
    v0_perturbed = v_ref_fine[0] + np.random.randn(3) * vel_noise
    m0_sim = m_ref_fine[0]
    
    # 运行闭环仿真（使用插值后的密集轨迹）
    r_sim, v_sim, m_sim = run_closed_loop(
        r0_perturbed, v0_perturbed, m0_sim,
        r_ref_fine, v_ref_fine, m_ref_fine, u_ref_fine, t_control
    )
    
    # 计算终端误差
    rf = r_ref_fine[-1]
    vf = v_ref_fine[-1]
    pos_error = np.linalg.norm(r_sim[-1] - rf)
    vel_error = np.linalg.norm(v_sim[-1] - vf)
    
    print(f"  仿真步数: {len(t_control)}")
    print(f"  终端位置误差: {pos_error:.4f} m")
    print(f"  终端速度误差: {vel_error:.4f} m/s")
    
    # 3. 蒙特卡洛验证
    print("\n[3/3] 蒙特卡洛验证...")
    mc_config = phase3_config.get("monte_carlo", {})
    
    if mc_config.get("enabled", True):
        n_sims = mc_config.get("n_simulations", 20)
        pos_std = mc_config.get("position_noise", 20.0)  # 位置扰动20m
        vel_std = mc_config.get("velocity_noise", 1.0)   # 速度扰动1m/s
        
        print(f"  仿真次数: {n_sims}")
        print(f"  位置扰动: {pos_std} m")
        print(f"  速度扰动: {vel_std} m/s")
        
        np.random.seed(42)
        success_count = 0
        pos_errors = []
        vel_errors = []
        
        for i in range(n_sims):
            # 添加扰动
            r0_mc = r_ref_fine[0] + np.random.randn(3) * pos_std
            v0_mc = v_ref_fine[0] + np.random.randn(3) * vel_std
            m0_mc = m_ref_fine[0]
            
            # 闭环仿真
            r_mc, v_mc, m_mc = run_closed_loop(
                r0_mc, v0_mc, m0_mc,
                r_ref_fine, v_ref_fine, m_ref_fine, u_ref_fine, t_control
            )
            
            pos_err = np.linalg.norm(r_mc[-1] - rf)
            vel_err = np.linalg.norm(v_mc[-1] - vf)
            pos_errors.append(pos_err)
            vel_errors.append(vel_err)
            
            if pos_err < 50 and vel_err < 5:
                success_count += 1
        
        mc_results = {
            'success_rate': success_count / n_sims,
            'mean_pos_error': np.mean(pos_errors),
            'mean_vel_error': np.mean(vel_errors),
        }
        
        print(f"  成功率: {mc_results['success_rate']*100:.1f}%")
        print(f"  平均位置误差: {mc_results['mean_pos_error']:.4f} m")
        print(f"  平均速度误差: {mc_results['mean_vel_error']:.4f} m/s")
    else:
        mc_results = {'success_rate': 1.0, 'mean_pos_error': 0, 'mean_vel_error': 0}
        print("  蒙特卡洛验证已禁用")
    
    return {
        'tracking_result': {
            't': t_control,
            'r': r_sim,
            'v': v_sim,
            'm': m_sim,
            'pos_error': pos_error,
            'vel_error': vel_error,
        },
        'monte_carlo_result': mc_results,
    }


# ==================== 主函数 ====================

class SimpleAsteroid:
    """简化的点质量引力模型（绕过DNN）"""
    
    def __init__(self, config):
        self.mu = config["phase2"]["asteroid"]["mu"]
        self.omega = np.array(config["phase2"]["asteroid"]["omega"])
    
    def compute_gravity(self, position):
        """计算引力加速度（点质量模型）"""
        r = np.linalg.norm(position)
        if r < 1e-10:
            return np.zeros(3)
        return -self.mu * position / (r**3)


def main():
    parser = argparse.ArgumentParser(description="小行星附着最优轨迹设计")
    parser.add_argument("--skip-training", action="store_true", help="跳过DNN训练")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="只运行指定阶段")
    parser.add_argument("--method", type=str, default="homotopy", 
                       choices=["direct", "cvxpy", "pseudospectral", "shooting", "homotopy"],
                       help="轨迹优化方法")
    parser.add_argument("--simple", action="store_true", help="使用点质量引力模型（绕过DNN）")
    args = parser.parse_args()
    
    print("="*70)
    print("小行星附着最优轨迹设计 - 完整流程")
    print("="*70)
    
    start_time = time.time()
    
    # 加载配置
    config = load_config()
    create_directories(config)
    
    # 创建航天器模型
    spacecraft = Spacecraft(config)
    
    # 使用简化模型或DNN模型
    if args.simple:
        print("\n[简化模式] 使用点质量引力模型")
        asteroid = SimpleAsteroid(config)
    else:
        # 第一阶段
        if args.phase is None or args.phase == 1:
            dnn_model, ply_model = run_phase1(config, skip_training=args.skip_training)
            asteroid = Asteroid(dnn_model, config)
        else:
            # 只运行后续阶段时，加载已有模型
            _, _, GravityAndGradientDNN, _ = _import_gravity_learning()
            dnn_model = GravityAndGradientDNN.load_model(config['phase1']['output']['model_file'])
            asteroid = Asteroid(dnn_model, config)
    
    # 第二阶段
    if args.phase is None or args.phase == 2:
        trajectory_result = run_phase2(config, asteroid, spacecraft, method=args.method)
    else:
        trajectory_result = None
    
    # 第三阶段
    if args.phase is None or args.phase == 3:
        if trajectory_result is not None:
            control_result = run_phase3(config, asteroid, spacecraft, trajectory_result)
        elif args.phase == 3:
            print("错误: 第三阶段需要先运行第二阶段")
            return
    
    # 汇总
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    print(f"总耗时: {total_time:.2f} s")
    
    if trajectory_result and trajectory_result.get('success'):
        print(f"\n轨迹优化结果:")
        print(f"  方法: {args.method}")
        print(f"  燃料消耗: {trajectory_result['fuel_consumption']:.2f} kg")
        print(f"  位置误差: {trajectory_result['pos_error']:.4f} m")
        print(f"  速度误差: {trajectory_result['vel_error']:.4f} m/s")


if __name__ == "__main__":
    main()
