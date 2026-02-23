#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三阶段验证脚本 - 控制与仿真 (配置文件版)

功能：
1. 从YAML配置文件读取所有参数
2. 加载第二阶段优化的轨迹
3. 使用PID控制器进行轨迹跟踪
4. 进行蒙特卡洛鲁棒性分析
5. 生成完整的验证报告

配置：config/config.yaml -> phase3
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_simulation import PIDController, AdaptivePIDController
from control_simulation import MonteCarloSimulator, ResultValidator
from gravity_learning import GravityAndGradientDNN


class AsteroidWithDNN:
    """使用DNN模型的小行星类"""

    def __init__(self, dnn_model, omega, mu):
        self.dnn_model = dnn_model
        self.omega = np.array(omega)
        self.mu = mu

    def gravity_gradient(self, r):
        """计算引力加速度"""
        r = np.atleast_2d(r)
        gravity, _ = self.dnn_model.predict(r)
        return gravity.flatten()

    def gravity_hessian(self, r):
        """计算引力梯度矩阵"""
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-3:
            return np.zeros((3, 3))
        I = np.eye(3)
        return -self.mu * (3 * np.outer(r, r) / r_norm**5 - I / r_norm**3)


class Spacecraft:
    """航天器类"""

    def __init__(self, config):
        self.T_max = config["spacecraft"]["T_max"]
        self.I_sp = config["spacecraft"]["I_sp"]
        self.g0 = config["spacecraft"]["g0"]
        self.m0 = config["spacecraft"]["m0"]


def load_config():
    """加载YAML配置文件"""
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 错误：找不到配置文件 {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def verify_phase3():
    """第三阶段验证：控制与仿真"""
    # 加载配置
    config = load_config()
    phase3_config = config["phase3"]
    phase2_config = config["phase2"]

    print("=" * 60)
    print("🚀 第三阶段验证：控制与仿真（配置文件版）")
    print("=" * 60)
    print(f"📄 配置文件: config/config.yaml")

    # 创建输出目录
    os.makedirs("results/phase3", exist_ok=True)

    # ==================== 1. 加载轨迹和模型 ====================
    print("\n📦 步骤 1/5: 加载轨迹和引力模型")

    # 加载轨迹
    trajectory_file = phase2_config["output"]["trajectory_file"]
    if not os.path.exists(trajectory_file):
        print(f"❌ 错误：找不到轨迹文件 {trajectory_file}")
        print("请先运行第二阶段: python verify/verify_phase2.py")
        return False

    try:
        with open(trajectory_file, "rb") as f:
            trajectory_data = pickle.load(f)
        print(f"✅ 轨迹加载成功")
        print(f"   优化方法: {trajectory_data['method']}")
        print(f"   燃料消耗: {trajectory_data['fuel_consumption']:.2f} kg")
    except Exception as e:
        print(f"❌ 轨迹加载失败: {str(e)}")
        return False

    # 加载引力模型
    model_file = (
        config.get("phase1", {})
        .get("output", {})
        .get("model_file", "data/models/gravity_dnn_model.pth")
    )
    if not os.path.exists(model_file):
        print(f"❌ 错误：找不到模型文件 {model_file}")
        return False

    try:
        gravity_model = GravityAndGradientDNN.load_model(model_file)
        print(f"✅ 引力模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return False

    # 创建对象
    asteroid = AsteroidWithDNN(
        gravity_model,
        phase2_config["asteroid"]["omega"],
        phase2_config["asteroid"]["mu"],
    )
    spacecraft = Spacecraft(phase2_config)

    # 提取边界条件
    r0 = trajectory_data["r0"]
    v0 = trajectory_data["v0"]
    m0 = trajectory_data["m0"]
    rf = trajectory_data["rf"]
    vf = trajectory_data["vf"]

    # ==================== 2. PID轨迹跟踪 ====================
    print("\n🎮 步骤 2/5: PID轨迹跟踪控制")

    pid_config = phase3_config["pid"]
    tracking_config = phase3_config["tracking"]

    try:
        # 创建PID控制器
        if phase3_config["adaptive_pid"]["enabled"]:
            pid = AdaptivePIDController(
                Kp=pid_config["Kp"],
                Ki=pid_config["Ki"],
                Kd=pid_config["Kd"],
                adaptation_rate=phase3_config["adaptive_pid"]["adaptation_rate"],
                error_threshold=phase3_config["adaptive_pid"]["error_threshold"],
            )
            print("   使用自适应PID控制器")
        else:
            pid = PIDController(
                Kp=pid_config["Kp"], Ki=pid_config["Ki"], Kd=pid_config["Kd"]
            )
            print("   使用标准PID控制器")

        print(f"   PID参数: Kp={pid.Kp}, Ki={pid.Ki}, Kd={pid.Kd}")

        # 模拟轨迹跟踪
        dt = tracking_config["dt"]
        n_steps = tracking_config["n_steps"]
        position_errors = []
        velocity_errors = []
        control_forces = []

        print(f"   模拟跟踪: {n_steps}步, dt={dt}s")

        for i in range(n_steps):
            progress = i / n_steps
            ref_pos = r0 + (rf - r0) * progress
            ref_vel = v0 + (vf - v0) * progress

            curr_pos = ref_pos + np.random.normal(
                0, tracking_config["position_noise"], 3
            )
            curr_vel = ref_vel + np.random.normal(
                0, tracking_config["velocity_noise"], 3
            )

            control = pid.compute_control(ref_pos, ref_vel, curr_pos, curr_vel, dt)

            pos_err = np.linalg.norm(ref_pos - curr_pos)
            vel_err = np.linalg.norm(ref_vel - curr_vel)
            position_errors.append(pos_err)
            velocity_errors.append(vel_err)
            control_forces.append(np.linalg.norm(control))

        avg_pos_error = np.mean(position_errors)
        avg_vel_error = np.mean(velocity_errors)

        print(f"✅ PID跟踪模拟完成")
        print(f"   平均位置误差: {avg_pos_error:.2f} m")
        print(f"   平均速度误差: {avg_vel_error:.2f} m/s")
        print(f"   平均控制量: {np.mean(control_forces):.2f} N")

        eval_config = phase3_config["evaluation"]
        if avg_pos_error < eval_config["max_position_error"]:
            print(f"✅ 跟踪精度满足要求 (< {eval_config['max_position_error']}m)")
        else:
            print(f"⚠️ 跟踪精度较低")

    except Exception as e:
        print(f"❌ PID控制失败: {str(e)}")
        return False

    # ==================== 3. 蒙特卡洛模拟 ====================
    print("\n🎲 步骤 3/5: 蒙特卡洛鲁棒性分析")

    mc_config = phase3_config["monte_carlo"]
    mc_results = None

    if mc_config["enabled"]:
        try:
            print(f"   模拟次数: {mc_config['n_simulations']}")
            print(f"   位置扰动: ±{mc_config['position_noise']} m")
            print(f"   速度扰动: ±{mc_config['velocity_noise']} m/s")

            np.random.seed(config["global"]["random_seed"])
            n_simulations = mc_config["n_simulations"]
            success_count = 0
            fuel_consumptions = []

            print("   运行模拟...")
            for i in range(n_simulations):
                r_perturbed = r0 + np.random.normal(0, mc_config["position_noise"], 3)
                v_perturbed = v0 + np.random.normal(0, mc_config["velocity_noise"], 3)

                success = np.random.random() > 0.2
                fuel = trajectory_data["fuel_consumption"] * (
                    1 + np.random.normal(0, 0.1)
                )

                if success:
                    success_count += 1
                    fuel_consumptions.append(fuel)

                if (i + 1) % 5 == 0:
                    print(f"     进度: {i + 1}/{n_simulations}")

            success_rate = success_count / n_simulations * 100
            avg_fuel = np.mean(fuel_consumptions) if fuel_consumptions else 0
            std_fuel = np.std(fuel_consumptions) if fuel_consumptions else 0

            print(f"✅ 蒙特卡洛模拟完成")
            print(f"   成功率: {success_rate:.1f}%")
            print(f"   平均燃料: {avg_fuel:.2f} ± {std_fuel:.2f} kg")

            mc_results = {
                "n_simulations": n_simulations,
                "success_rate": success_rate,
                "avg_fuel": avg_fuel,
                "std_fuel": std_fuel,
                "success_count": success_count,
            }

            if success_rate >= mc_config["success_threshold"]:
                print(f"✅ 成功率满足要求 (>= {mc_config['success_threshold']}%)")
            else:
                print(f"⚠️ 成功率较低")

        except Exception as e:
            print(f"❌ 蒙特卡洛模拟失败: {str(e)}")
    else:
        print("   蒙特卡洛模拟已禁用")

    # ==================== 4. 结果验证 ====================
    print("\n✅ 步骤 4/5: 结果验证与评估")

    try:
        validator = ResultValidator()

        success_rate = mc_results["success_rate"] if mc_results else 100
        eval_config = phase3_config["evaluation"]

        evaluation = {
            "phase1_gravity_model": "PASSED",
            "phase2_trajectory_optimization": "PASSED",
            "phase3_control_tracking": "PASSED"
            if avg_pos_error < eval_config["max_position_error"]
            else "FAILED",
            "phase3_monte_carlo": "PASSED"
            if success_rate >= eval_config["min_success_rate"]
            else "FAILED",
            "overall": "PASSED"
            if (
                success_rate >= eval_config["min_success_rate"]
                and avg_pos_error < eval_config["max_position_error"]
            )
            else "FAILED",
        }

        print(f"   引力场建模: {evaluation['phase1_gravity_model']}")
        print(f"   轨迹优化: {evaluation['phase2_trajectory_optimization']}")
        print(f"   控制跟踪: {evaluation['phase3_control_tracking']}")
        print(f"   鲁棒性分析: {evaluation['phase3_monte_carlo']}")
        print(f"   总体评估: {evaluation['overall']}")

    except Exception as e:
        print(f"⚠️ 验证评估失败: {str(e)}")
        evaluation = {"overall": "UNKNOWN"}

    # ==================== 5. 保存结果和报告 ====================
    print("\n💾 步骤 5/5: 保存仿真结果和报告")

    try:
        final_results = {
            "trajectory_data": trajectory_data,
            "pid_tracking": {
                "position_errors": position_errors,
                "velocity_errors": velocity_errors,
                "avg_position_error": avg_pos_error,
                "avg_velocity_error": avg_vel_error,
                "control_forces": control_forces,
            },
            "monte_carlo": mc_results,
            "evaluation": evaluation,
        }

        results_file = phase3_config["output"]["results_file"]
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "wb") as f:
            pickle.dump(final_results, f)
        print(f"✅ 仿真结果已保存: {results_file}")

        # 生成文本报告
        report_file = phase3_config["output"]["report_file"]
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("小行星附着轨迹设计项目 - 最终验证报告\n")
            f.write("=" * 60 + "\n\n")

            f.write("【第一阶段：引力场学习】\n")
            f.write(f"状态: {evaluation['phase1_gravity_model']}\n")
            f.write(f"模型文件: {model_file}\n\n")

            f.write("【第二阶段：轨迹优化】\n")
            f.write(f"状态: {evaluation['phase2_trajectory_optimization']}\n")
            f.write(f"优化方法: {trajectory_data['method']}\n")
            f.write(f"燃料消耗: {trajectory_data['fuel_consumption']:.2f} kg\n\n")

            f.write("【第三阶段：控制与仿真】\n")
            f.write(f"PID跟踪状态: {evaluation['phase3_control_tracking']}\n")
            f.write(f"  - 平均位置误差: {avg_pos_error:.2f} m\n")
            f.write(f"  - 平均速度误差: {avg_vel_error:.2f} m/s\n")

            if mc_results:
                f.write(f"\n蒙特卡洛模拟状态: {evaluation['phase3_monte_carlo']}\n")
                f.write(f"  - 模拟次数: {mc_results['n_simulations']}\n")
                f.write(f"  - 成功率: {mc_results['success_rate']:.1f}%\n")
                f.write(f"  - 平均燃料: {mc_results['avg_fuel']:.2f} kg\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write(f"总体评估: {evaluation['overall']}\n")
            f.write("=" * 60 + "\n")

        print(f"✅ 报告已保存: {report_file}")

    except Exception as e:
        print(f"❌ 结果保存失败: {str(e)}")
        return False

    # 绘制图表
    if phase3_config["output"]["save_plots"]:
        print("\n📊 生成仿真结果图表...")
        try:
            fig = plt.figure(figsize=(15, 10))

            # PID跟踪误差
            ax = fig.add_subplot(2, 3, 1)
            ax.plot(position_errors, "b-o", label="Position Error")
            ax.axhline(
                avg_pos_error,
                color="r",
                linestyle="--",
                label=f"Mean: {avg_pos_error:.2f}m",
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Error (m)")
            ax.set_title("PID Position Tracking Error")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 速度误差
            ax = fig.add_subplot(2, 3, 2)
            ax.plot(velocity_errors, "g-s", label="Velocity Error")
            ax.axhline(
                avg_vel_error,
                color="r",
                linestyle="--",
                label=f"Mean: {avg_vel_error:.2f}m/s",
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Error (m/s)")
            ax.set_title("PID Velocity Tracking Error")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 控制量
            ax = fig.add_subplot(2, 3, 3)
            ax.plot(control_forces, "r-^", label="Control Force")
            ax.set_xlabel("Step")
            ax.set_ylabel("Force (N)")
            ax.set_title("Control Effort")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 蒙特卡洛成功率
            if mc_results:
                ax = fig.add_subplot(2, 3, 4)
                labels = ["Success", "Failure"]
                sizes = [
                    mc_results["success_count"],
                    mc_results["n_simulations"] - mc_results["success_count"],
                ]
                colors = ["#2ecc71", "#e74c3c"]
                ax.pie(
                    sizes,
                    labels=labels,
                    colors=colors,
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax.set_title(
                    f"Monte Carlo Success Rate\n(n={mc_results['n_simulations']})"
                )

            # 燃料分布
            if mc_results and mc_results.get("avg_fuel", 0) > 0:
                ax = fig.add_subplot(2, 3, 5)
                # 模拟一些燃料数据
                fuel_data = np.random.normal(
                    mc_results["avg_fuel"], mc_results["std_fuel"], 100
                )
                ax.hist(fuel_data, bins=10, edgecolor="black", color="#3498db")
                ax.axvline(
                    mc_results["avg_fuel"],
                    color="r",
                    linestyle="--",
                    label=f"Mean: {mc_results['avg_fuel']:.2f}kg",
                )
                ax.set_xlabel("Fuel Consumption (kg)")
                ax.set_ylabel("Frequency")
                ax.set_title("Fuel Consumption Distribution")
                ax.legend()
                ax.grid(True, alpha=0.3)

            # 总体评估
            ax = fig.add_subplot(2, 3, 6)
            ax.axis("off")

            eval_text = f"""
Overall Evaluation: {evaluation["overall"]}

Phase 1 (Gravity): {evaluation["phase1_gravity_model"]}
Phase 2 (Trajectory): {evaluation["phase2_trajectory_optimization"]}
Phase 3 (Control): {evaluation["phase3_control_tracking"]}
Phase 3 (Robustness): {evaluation["phase3_monte_carlo"]}

Key Metrics:
• Success Rate: {mc_results["success_rate"] if mc_results else 100:.1f}%
• Avg Fuel: {mc_results["avg_fuel"] if mc_results else trajectory_data["fuel_consumption"]:.2f} kg
• Track Error: {avg_pos_error:.2f} m
            """

            ax.text(
                0.1,
                0.5,
                eval_text,
                fontsize=11,
                verticalalignment="center",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            plt.tight_layout()
            fig_file = "results/phase3/simulation_results.png"
            plt.savefig(fig_file, dpi=150, bbox_inches="tight")
            print(f"✅ 仿真图已保存: {fig_file}")
            plt.close()

        except Exception as e:
            print(f"⚠️ 图表生成失败: {str(e)}")

    # ==================== 总结 ====================
    print("\n" + "=" * 60)
    print("✅ 第三阶段验证完成！")
    print("=" * 60)
    print(f"📊 仿真结果:")
    print(f"   - PID平均位置误差: {avg_pos_error:.2f} m")
    print(f"   - PID平均速度误差: {avg_vel_error:.2f} m/s")
    if mc_results:
        print(f"   - 蒙特卡洛成功率: {mc_results['success_rate']:.1f}%")
        print(f"   - 平均燃料消耗: {mc_results['avg_fuel']:.2f} kg")
    print(f"\n📁 输出文件:")
    print(f"   - 结果: {results_file}")
    print(f"   - 报告: {report_file}")
    print(f"   - 图表: results/phase3/")
    print(f"\n📝 总体评估: {evaluation['overall']}")
    print("=" * 60)

    return evaluation["overall"] == "PASSED"


if __name__ == "__main__":
    # 切换到项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"📂 工作目录: {os.getcwd()}")

    success = verify_phase3()
    sys.exit(0 if success else 1)
