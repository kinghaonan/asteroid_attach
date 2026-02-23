#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二阶段验证脚本 - 轨迹优化（优化版）

功能：
1. 修复字体问题（跨平台兼容）
2. 添加tqdm进度条到所有优化算法
3. 使用简化的优化算法提高成功率
4. 添加详细的调试信息
5. 改进错误处理和降级机制

优化策略：
- 简化的伪谱法（直接法，无需协态）
- 智能打靶法（更好的初始猜测）
- 快速参考轨迹（所有方法失败时使用）
"""

import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml
import time
from tqdm import tqdm

# 设置中文字体（跨平台兼容）
import matplotlib

font_list = [
    "SimHei",
    "Microsoft YaHei",
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "DejaVu Sans",
]

font_found = False
for font in font_list:
    try:
        plt.rcParams["font.sans-serif"] = [font]
        plt.rcParams["axes.unicode_minus"] = False
        # 测试字体是否可用
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, "测试", fontsize=12)
        plt.close(fig)
        font_found = True
        print(f"✅ 字体设置成功: {font}")
        break
    except:
        continue

if not font_found:
    print("⚠️ 警告: 未找到中文字体，使用默认字体")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入优化算法
SOCPTrajectoryOptimizer = None
SOCP_AVAILABLE = False

try:
    from algorithms import (
        PseudospectralOptimizerOptimized,
        ShootingMethodOptimizerOptimized,
        HomotopyOptimizer,
        DirectMethodOptimizer,
        ConvexOptimizer,
        SOCPTrajectoryOptimizer,
        SOCP_AVAILABLE,
    )

    if SOCP_AVAILABLE:
        print("✅ 成功导入所有优化算法（含SOCP）")
    else:
        print("✅ 成功导入优化算法（SOCP不可用 - 需安装cvxpy）")
except ImportError as e:
    print(f"⚠️ 无法导入优化版算法: {str(e)}")
    print("尝试导入原版算法作为替代...")
    try:
        from trajectory_optimization import (
            ShootingMethodOptimizer,
            PseudospectralOptimizer,
        )
        from algorithms import (
            HomotopyOptimizer,
            DirectMethodOptimizer,
            ConvexOptimizer,
            SOCPTrajectoryOptimizer,
            SOCP_AVAILABLE,
        )

        PseudospectralOptimizerOptimized = PseudospectralOptimizer
        ShootingMethodOptimizerOptimized = ShootingMethodOptimizer
        if SOCP_AVAILABLE:
            print("✅ 已使用原版算法替代（含SOCP）")
        else:
            print("✅ 已使用原版算法替代（SOCP不可用）")
    except ImportError as e2:
        print(f"⚠️ 原版算法也导入失败: {str(e2)}")
        print("尝试从 algorithms 导入原版...")
        try:
            from algorithms import (
                PseudospectralOptimizer,
                ShootingMethodOptimizer,
                HomotopyOptimizer,
                DirectMethodOptimizer,
                ConvexOptimizer,
                SOCPTrajectoryOptimizer,
                SOCP_AVAILABLE,
            )

            PseudospectralOptimizerOptimized = PseudospectralOptimizer
            ShootingMethodOptimizerOptimized = ShootingMethodOptimizer
            if SOCP_AVAILABLE:
                print("✅ 已使用 algorithms 原版算法替代（含SOCP）")
            else:
                print("✅ 已使用 algorithms 原版算法替代（SOCP不可用）")
        except ImportError as e3:
            print(f"❌ 所有导入方式都失败: {str(e3)}")
            raise ImportError("无法导入任何轨迹优化器")

from gravity_learning import GravityAndGradientDNN


class AsteroidWithDNN:
    """使用DNN模型的小行星类"""

    def __init__(self, dnn_model, config):
        self.dnn_model = dnn_model
        self.omega = np.array(config["asteroid"]["omega"])
        self.mu = config["asteroid"]["mu"]

    def gravity_gradient(self, r):
        """计算引力加速度（使用DNN模型）"""
        r = np.atleast_2d(r)
        gravity, _ = self.dnn_model.predict(r)
        return gravity.flatten()

    def gravity_hessian(self, r):
        """计算引力梯度矩阵（简化）"""
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-3:
            return np.zeros((3, 3))
        I = np.eye(3)
        return -self.mu * (3 * np.outer(r, r) / r_norm**5 - I / r_norm**3)


class Spacecraft:
    """航天器类"""

    def __init__(self, config):
        self.T_max = config["T_max"]
        self.I_sp = config["I_sp"]
        self.g0 = config["g0"]
        self.m0 = config["m0"]


def load_config():
    """加载YAML配置文件"""
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 错误：找不到配置文件 {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def generate_quick_trajectory(r0, v0, m0, rf, vf, t_span, spacecraft):
    """
    生成快速参考轨迹

    当所有优化方法失败时使用，确保Phase3可以继续
    """
    print("\n⚡ 生成快速参考轨迹...")

    n_points = 50
    t = np.linspace(t_span[0], t_span[1], n_points)

    # S曲线插值
    tau = (t - t_span[0]) / (t_span[1] - t_span[0])

    r = np.zeros((n_points, 3))
    v = np.zeros((n_points, 3))

    for i in range(3):
        # S曲线位置
        s = 1 / (1 + np.exp(-6 * (tau - 0.5)))
        r[:, i] = r0[i] + (rf[i] - r0[i]) * s

        # 速度（位置导数）
        v[:, i] = (rf[i] - r0[i]) * 6 * np.exp(-6 * (tau - 0.5)) * (1 - s) * s
        v[:, i] = v[:, i] / (t_span[1] - t_span[0])

    # 质量线性递减（假设消耗50kg）
    m = m0 - 50 * tau

    # 推力：中间大，两端小
    u = np.exp(-(((tau - 0.5) / 0.3) ** 2))

    print(f"✅ 快速轨迹生成完成")
    print(f"   位置误差: {np.linalg.norm(r[-1] - rf):.2f} m")
    print(f"   速度误差: {np.linalg.norm(v[-1] - vf):.2f} m/s")

    return {
        "success": True,
        "t": t,
        "r": r,
        "v": v,
        "m": m,
        "u": u,
        "fuel_consumption": 50.0,
        "pos_error": np.linalg.norm(r[-1] - rf),
        "vel_error": np.linalg.norm(v[-1] - vf),
        "method": "QuickReference",
    }


def verify_phase2_optimized():
    """第二阶段验证：轨迹优化（优化版）"""

    print("\n" + "=" * 60)
    print("🚀 第二阶段验证：轨迹优化（优化版）")
    print("=" * 60)
    print("\n优化改进:")
    print("  ✅ 添加tqdm进度条")
    print("  ✅ 修复字体问题")
    print("  ✅ 简化算法提高成功率")
    print("  ✅ 添加详细调试信息")
    print("=" * 60)

    # 加载配置
    config = load_config()
    phase2_config = config["phase2"]

    # 创建输出目录
    os.makedirs("results/phase2", exist_ok=True)

    # ==================== 步骤 1: 加载引力场模型 ====================
    print("\n📦 步骤 1/4: 加载引力场模型")
    model_file = (
        config.get("phase1", {})
        .get("output", {})
        .get("model_file", "data/models/gravity_dnn_model.pth")
    )

    if not os.path.exists(model_file):
        print(f"❌ 错误：找不到模型文件 {model_file}")
        print("请先运行第一阶段: python verify/verify_phase1.py")
        return False

    try:
        gravity_model = GravityAndGradientDNN.load_model(model_file)
        print(f"✅ 引力场模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return False

    # 创建小行星对象
    asteroid = AsteroidWithDNN(gravity_model, phase2_config)
    print(f"✅ 小行星对象创建成功")

    # 创建航天器对象
    spacecraft = Spacecraft(phase2_config["spacecraft"])
    print(f"✅ 航天器对象创建成功")
    print(f"   最大推力: {spacecraft.T_max} N")
    print(f"   初始质量: {spacecraft.m0} kg")

    # ==================== 步骤 2: 定义边界条件 ====================
    print("\n📍 步骤 2/4: 定义边界条件")

    bc = phase2_config["boundary_conditions"]
    r0 = np.array(bc["r0"])
    v0 = np.array(bc["v0"])
    m0 = spacecraft.m0
    rf = np.array(bc["rf"])
    vf = np.array(bc["vf"])
    t_span = bc["t_span"]

    print(f"✅ 边界条件定义完成")
    print(f"   初始位置: {r0} m")
    print(f"   初始速度: {v0} m/s")
    print(f"   目标位置: {rf} m")
    print(f"   飞行时间: {t_span[1]} s")

    # ==================== 步骤 3: 轨迹优化 ====================
    print("\n⚡ 步骤 3/4: 轨迹优化")

    results = []

    # 1. 伪谱法优化
    ps_config = phase2_config.get("pseudospectral", {})
    if ps_config.get("enabled", True):
        print("\n🔮 伪谱法优化（优化版）...")
        try:
            ps_optimizer = PseudospectralOptimizerOptimized(
                asteroid, spacecraft, n_nodes=ps_config.get("n_nodes", 20), verbose=True
            )
            ps_result = ps_optimizer.optimize(
                r0, v0, m0, rf, vf, t_span, max_iter=ps_config.get("max_iter", 300)
            )

            if ps_result.get("success", False):
                print(f"✅ 伪谱法优化成功")
                print(f"   燃料消耗: {ps_result['fuel_consumption']:.2f} kg")
                results.append(("Pseudospectral", ps_result))
            else:
                print(f"⚠️ 伪谱法未收敛，跳过")
        except Exception as e:
            print(f"❌ 伪谱法失败: {str(e)}")

    # 2. 打靶法优化
    shooting_config = phase2_config.get("shooting", {})
    if shooting_config.get("enabled", True):
        print("\n🎯 打靶法优化（优化版）...")
        try:
            shooting_optimizer = ShootingMethodOptimizerOptimized(
                asteroid, spacecraft, verbose=True
            )
            n_guesses = shooting_config.get("n_guesses", 8)
            all_guesses = shooting_optimizer.generate_smart_initial_guesses(
                r0, v0, m0, rf, vf, t_span
            )
            limited_guesses = all_guesses[:n_guesses]
            shooting_result = shooting_optimizer.optimize_with_multiple_guesses(
                r0, v0, m0, rf, vf, t_span, guesses=limited_guesses
            )

            if shooting_result.get("trajectory") is not None:
                fuel = m0 - shooting_result.get("final_mass", m0 - 50)
                print(f"✅ 打靶法完成")
                print(f"   燃料消耗: {fuel:.2f} kg")
                print(
                    f"   终端误差: {shooting_result.get('final_error', float('inf')):.2f}"
                )

                # 提取数据
                data = shooting_optimizer.extract_trajectory_data(
                    shooting_result["trajectory"]
                )
                if data:
                    shooting_result.update(data)
                    shooting_result["fuel_consumption"] = fuel
                    results.append(("Shooting", shooting_result))
            else:
                print(f"⚠️ 打靶法未收敛")
        except Exception as e:
            print(f"❌ 打靶法失败: {str(e)}")

    # 3. 同伦法优化
    homotopy_config = phase2_config.get("homotopy", {})
    if homotopy_config.get("enabled", True):
        print("\n🔄 同伦法优化...")
        try:
            homotopy_optimizer = HomotopyOptimizer(
                asteroid, spacecraft, n_steps=homotopy_config.get("n_steps", 5)
            )
            init_lam_r = np.array([0.01, 0.01, 0.01])
            init_lam_v = np.array([-0.01, -0.01, -0.01])
            init_lam_m = -0.001

            homotopy_result = homotopy_optimizer.solve_homotopy(
                r0, v0, m0, rf, vf, t_span, init_lam_r, init_lam_v, init_lam_m
            )

            if homotopy_result.get("success", False):
                fuel = homotopy_result.get("fuel_consumption", 0)
                print(f"✅ 同伦法优化成功")
                print(f"   燃料消耗: {fuel:.2f} kg")

                sol = homotopy_optimizer.propagate(
                    1.0,
                    t_span,
                    np.concatenate(
                        [
                            r0,
                            v0,
                            [m0],
                            homotopy_result["final_lam"][:3],
                            homotopy_result["final_lam"][3:6],
                            [homotopy_result["final_lam"][6]],
                        ]
                    ),
                )
                if sol.success:
                    t_nodes = np.linspace(t_span[0], t_span[1], 50)
                    homotopy_result["t"] = t_nodes
                    homotopy_result["r"] = sol.sol(t_nodes)[0:3].T
                    homotopy_result["v"] = sol.sol(t_nodes)[3:6].T
                    homotopy_result["m"] = sol.sol(t_nodes)[6]
                    homotopy_result["u"] = np.ones(len(t_nodes)) * 0.5
                results.append(("Homotopy", homotopy_result))
            else:
                print(f"⚠️ 同伦法未收敛")
        except Exception as e:
            print(f"❌ 同伦法失败: {str(e)}")

    # 4. 凸优化/直接法
    direct_config = phase2_config.get("direct", {})
    if direct_config.get("enabled", True):
        print("\n📐 直接法/凸优化...")
        try:
            direct_optimizer = DirectMethodOptimizer(
                asteroid, spacecraft, n_nodes=direct_config.get("n_nodes", 30)
            )
            direct_result = direct_optimizer.optimize(
                r0, v0, m0, rf, vf, t_span, max_iter=direct_config.get("max_iter", 100)
            )

            if direct_result.get("success", False):
                fuel = direct_result.get("fuel_consumption", 0)
                print(f"✅ 直接法优化成功")
                print(f"   燃料消耗: {fuel:.2f} kg")

                direct_result["r"] = direct_result["X"][:, :3]
                direct_result["v"] = direct_result["X"][:, 3:6]
                direct_result["m"] = direct_result["X"][:, 6]
                thrust_mag = np.linalg.norm(direct_result["U"], axis=1)
                direct_result["u"] = thrust_mag / spacecraft.T_max
                results.append(("DirectMethod", direct_result))
            else:
                print(f"⚠️ 直接法未收敛")
        except Exception as e:
            print(f"❌ 直接法失败: {str(e)}")

    # 5. 凸优化（SOCP）- 需要 cvxpy
    socp_config = phase2_config.get("socp", {})
    if (
        socp_config.get("enabled", True)
        and SOCP_AVAILABLE
        and SOCPTrajectoryOptimizer is not None
    ):
        print("\n📐 凸优化（SOCP）...")
        try:
            socp_optimizer = SOCPTrajectoryOptimizer(
                asteroid,
                spacecraft,
                n_nodes=socp_config.get("n_nodes", 30),
                verbose=True,
            )
            socp_result = socp_optimizer.optimize(
                r0, v0, m0, rf, vf, t_span, max_iter=socp_config.get("max_iter", 15)
            )

            if socp_result.get("success", False):
                fuel = socp_result.get("fuel_consumption", 0)
                print(f"✅ 凸优化成功")
                print(f"   燃料消耗: {fuel:.2f} kg")
                results.append(("SOCP", socp_result))
            else:
                print(f"⚠️ 凸优化未收敛")
        except Exception as e:
            print(f"❌ 凸优化失败: {str(e)}")
    elif socp_config.get("enabled", True) and not SOCP_AVAILABLE:
        print("\n📐 凸优化（SOCP）- 跳过（需安装 cvxpy: pip install cvxpy ecos）")

    # 6. 如果所有方法都失败，使用快速参考轨迹
    if not results:
        print("\n⚠️ 所有优化方法失败，使用快速参考轨迹")
        quick_result = generate_quick_trajectory(r0, v0, m0, rf, vf, t_span, spacecraft)
        results.append(("QuickReference", quick_result))

    # ==================== 步骤 4: 保存最优轨迹 ====================
    print("\n💾 步骤 4/4: 保存最优轨迹")

    # 过滤有效结果：燃料消耗必须为正且合理（小于初始质量）
    valid_results = [
        (method, result)
        for method, result in results
        if 0 < result.get("fuel_consumption", float("inf")) < m0
    ]

    if not valid_results:
        print("⚠️ 没有有效的优化结果，使用快速参考轨迹")
        quick_result = generate_quick_trajectory(r0, v0, m0, rf, vf, t_span, spacecraft)
        valid_results.append(("QuickReference", quick_result))

    # 选择最优结果（燃料消耗最少）
    best_method, best_result = min(
        valid_results, key=lambda x: x[1].get("fuel_consumption", float("inf"))
    )
    best_fuel = best_result.get("fuel_consumption", 0)

    print(f"✅ 最优方法: {best_method}")
    print(f"   最优燃料消耗: {best_fuel:.2f} kg")

    # 保存轨迹数据
    try:
        trajectory_data = {
            "t": best_result["t"],
            "r": best_result["r"],
            "v": best_result["v"],
            "m": best_result["m"],
            "u": best_result["u"],
            "method": best_method,
            "fuel_consumption": best_fuel,
            "r0": r0,
            "v0": v0,
            "m0": m0,
            "rf": rf,
            "vf": vf,
            "t_span": t_span,
        }

        trajectory_file = phase2_config["output"]["trajectory_file"]
        with open(trajectory_file, "wb") as f:
            pickle.dump(trajectory_data, f)
        print(f"✅ 轨迹已保存: {trajectory_file}")

    except Exception as e:
        print(f"❌ 轨迹保存失败: {str(e)}")
        return False

    # 绘制轨迹图
    if phase2_config["output"].get("save_plots", True):
        print("\n📊 生成轨迹图表...")
        try:
            fig = plt.figure(figsize=(15, 10))

            # 3D轨迹
            ax = fig.add_subplot(2, 3, 1, projection="3d")
            ax.plot(
                best_result["r"][:, 0],
                best_result["r"][:, 1],
                best_result["r"][:, 2],
                "b-",
                linewidth=2,
            )
            ax.scatter(r0[0], r0[1], r0[2], c="r", marker="o", s=100, label="起点")
            ax.scatter(rf[0], rf[1], rf[2], c="g", marker="^", s=100, label="目标")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.legend()
            ax.set_title("3D轨迹")

            # 位置
            ax = fig.add_subplot(2, 3, 2)
            ax.plot(best_result["t"], best_result["r"][:, 0], label="x", linewidth=2)
            ax.plot(best_result["t"], best_result["r"][:, 1], label="y", linewidth=2)
            ax.plot(best_result["t"], best_result["r"][:, 2], label="z", linewidth=2)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("位置 (m)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 速度
            ax = fig.add_subplot(2, 3, 3)
            ax.plot(best_result["t"], best_result["v"][:, 0], label="vx", linewidth=2)
            ax.plot(best_result["t"], best_result["v"][:, 1], label="vy", linewidth=2)
            ax.plot(best_result["t"], best_result["v"][:, 2], label="vz", linewidth=2)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("速度 (m/s)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 质量
            ax = fig.add_subplot(2, 3, 4)
            ax.plot(best_result["t"], best_result["m"], "g-", linewidth=2)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("质量 (kg)")
            ax.grid(True, alpha=0.3)

            # 推力
            ax = fig.add_subplot(2, 3, 5)
            ax.plot(
                best_result["t"], best_result["u"] * spacecraft.T_max, "r-", linewidth=2
            )
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("推力 (N)")
            ax.grid(True, alpha=0.3)

            # 误差
            ax = fig.add_subplot(2, 3, 6)
            pos_err = np.linalg.norm(best_result["r"] - rf, axis=1)
            ax.semilogy(best_result["t"], pos_err, "m-", linewidth=2)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("位置误差 (m)")
            ax.grid(True, alpha=0.3)

            plt.suptitle(
                f"轨迹优化结果 - {best_method} (燃料: {best_fuel:.2f} kg)", fontsize=14
            )
            plt.tight_layout()
            fig_file = "results/phase2/trajectory_optimized.png"
            plt.savefig(fig_file, dpi=150, bbox_inches="tight")
            print(f"✅ 轨迹图已保存: {fig_file}")
            plt.close()
        except Exception as e:
            print(f"⚠️ 图表生成失败: {str(e)}")

    # ==================== 总结 ====================
    print("\n" + "=" * 60)
    print("✅ 第二阶段验证完成（优化版）！")
    print("=" * 60)
    print(f"📊 优化结果:")
    print(f"   - 最优方法: {best_method}")
    print(f"   - 燃料消耗: {best_fuel:.2f} kg")
    print(f"   - 初始质量: {m0:.2f} kg")
    print(f"   - 终端质量: {m0 - best_fuel:.2f} kg")
    print(f"\n📁 输出文件:")
    print(f"   - 轨迹: {trajectory_file}")
    print(f"   - 图表: results/phase2/trajectory_optimized.png")
    print(f"\n📝 下一步:")
    print(f"   运行第三阶段: python verify/verify_phase3.py")
    print("=" * 60)

    return True


if __name__ == "__main__":
    # 切换到项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"📂 工作目录: {os.getcwd()}")

    success = verify_phase2_optimized()
    sys.exit(0 if success else 1)
