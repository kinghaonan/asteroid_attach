#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一阶段验证脚本 - 引力场学习 (配置文件版，含引力梯度)

功能：
1. 从YAML配置文件读取所有参数
2. 优先从pkl文件加载样本（支持复用已计算的样本）
3. 如果pkl不存在，则计算新样本并保存
4. 加载PLY小行星模型
5. 训练DNN模型（同时学习引力和引力梯度）
6. 验证模型精度
7. 保存模型供第二阶段使用

配置：config/config.yaml -> phase1

注意：此版本优先使用pkl格式的样本文件
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle

# 设置字体（使用系统默认字体避免编码问题）
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gravity_learning import PLYAsteroidModel, PolyhedralGravitySampler
from gravity_learning import GravityAndGradientDNN, GravityGradientTrainer


def load_config():
    """加载YAML配置文件"""
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 错误：找不到配置文件 {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def load_samples_from_pkl(pkl_file):
    """
    从pkl文件加载样本数据

    pkl文件格式（来自ply模型训练改.py）：
    {
        "positions": points,  # (N, 3)
        "gravity": gravity,   # (N, 3)
        "gradient": gradient, # (N, 9)
        "metadata": {...}
    }
    """
    print(f"\n📂 正在从pkl文件加载样本: {pkl_file}")

    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"pkl文件不存在: {pkl_file}")

    with open(pkl_file, "rb") as f:
        sample_data = pickle.load(f)

    points = sample_data["positions"]
    gravity = sample_data["gravity"]
    gradient = sample_data.get("gradient", None)

    print(f"✅ pkl样本加载成功: {len(points)} 个点")
    print(f"   包含引力: {gravity.shape}", end="")
    if gradient is not None:
        print(f", 梯度: {gradient.shape}")
    else:
        print(" (无梯度数据)")

    # 显示元数据
    if "metadata" in sample_data:
        meta = sample_data["metadata"]
        print(f"   元数据:")
        for key, value in meta.items():
            if key != "sampling_time":  # 跳过时间戳
                print(f"     {key}: {value}")

    return points, gravity, gradient


def save_samples_to_npz(points, gravity, gradient, npz_file):
    """将样本保存为npz格式（用于快速加载）"""
    os.makedirs(os.path.dirname(npz_file), exist_ok=True)
    np.savez(npz_file, points=points, gravity=gravity, gradient=gradient)
    print(f"✅ 样本已缓存为npz: {npz_file}")


def load_or_generate_samples(sampler, config):
    """
    加载或生成引力样本（优先使用pkl文件）

    优先级：
    1. 如果pkl文件存在，直接从pkl加载
    2. 如果npz缓存存在，从npz加载
    3. 否则计算新样本
    """
    sample_config = config["phase1"]["sampling"]
    pkl_file = sample_config.get("pkl_file", "data/samples/samples_with_gradient.pkl")
    npz_file = sample_config.get("npz_file", "data/samples/gravity_samples.npz")
    reuse_existing = sample_config["reuse_existing"]

    # 1. 优先尝试从pkl文件加载
    if reuse_existing and os.path.exists(pkl_file):
        try:
            points, gravity, gradient = load_samples_from_pkl(pkl_file)

            # 如果pkl加载成功且有梯度数据，同时保存为npz缓存（加速下次加载）
            if gradient is not None and not os.path.exists(npz_file):
                save_samples_to_npz(points, gravity, gradient, npz_file)

            return points, gravity, gradient, True, "pkl"
        except Exception as e:
            print(f"⚠️ 从pkl加载失败: {str(e)}")
            print("   尝试其他方式...")

    # 2. 尝试从npz缓存加载
    if reuse_existing and os.path.exists(npz_file):
        print(f"\n📂 从npz缓存加载样本...")
        try:
            data = np.load(npz_file)
            points = data["points"]
            gravity = data["gravity"]
            gradient = data["gradient"]
            print(f"✅ npz缓存加载成功: {points.shape[0]} 个点")
            return points, gravity, gradient, True, "npz"
        except Exception as e:
            print(f"⚠️ 从npz加载失败: {str(e)}")
            print("   将重新计算样本...")

    # 3. 计算新样本
    print(f"\n📊 生成新的引力样本（同时计算引力和引力梯度）...")
    print(f"   样本数量: {sample_config['num_samples']}")
    print(
        f"   采样范围: {sample_config['min_r_ratio']} - {sample_config['max_r_ratio']} 倍半径"
    )

    points = sampler.generate_sampling_points(
        num_samples=sample_config["num_samples"],
        min_r_ratio=sample_config["min_r_ratio"],
        max_r_ratio=sample_config["max_r_ratio"],
    )

    # 计算引力
    print("   计算引力加速度...")
    gravity = sampler.calculate_polyhedral_gravity(points)

    # 计算引力梯度
    print("   计算引力梯度...")
    gradient = sampler.calculate_polyhedral_gravity_gradient(points)

    # 保存为npz缓存
    save_samples_to_npz(points, gravity, gradient, npz_file)

    return points, gravity, gradient, False, "computed"


def verify_phase1():
    """第一阶段验证：引力场学习（含引力梯度）"""
    # 加载配置
    config = load_config()
    phase1_config = config["phase1"]

    print("=" * 60)
    print("第一阶段验证：引力场学习（含引力梯度）")
    print("=" * 60)
    print(f"配置文件: config/config.yaml")

    # 创建输出目录
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/samples", exist_ok=True)
    os.makedirs("results/phase1", exist_ok=True)

    # ==================== 1. 加载PLY模型（可选，用于验证）====================
    print("\n步骤 1/6: 加载PLY小行星模型")
    ply_file = phase1_config["ply_file"]

    if not os.path.exists(ply_file):
        print(f"⚠️ 警告：找不到PLY文件 {ply_file}")
        print("   将尝试直接从样本文件加载数据")
        sampler = None
        radius = 0
    else:
        try:
            asteroid_model = PLYAsteroidModel(ply_file)
            asteroid_model.scale_to_real_size(
                target_diameter_m=phase1_config["target_diameter_m"]
            )
            radius = asteroid_model.get_asteroid_radius()
            sampler = PolyhedralGravitySampler(
                asteroid_model, asteroid_density=phase1_config["asteroid_density"]
            )
            print(f"✅ PLY模型加载成功")
            print(f"   小行星半径: {radius:.2f} m")
        except Exception as e:
            print(f"⚠️ PLY模型加载失败: {str(e)}")
            sampler = None
            radius = 0

    # ==================== 2. 加载样本（优先从pkl）====================
    print("\n步骤 2/6: 准备引力样本数据（优先从pkl加载）")

    try:
        points, gravity, gradient, loaded_from_file, source = load_or_generate_samples(
            sampler, config
        )

        if loaded_from_file:
            print(f"✅ 从{source.upper()}加载样本: {points.shape}")
        else:
            print(f"✅ 新生成样本: {points.shape}")

        print(f"   引力范围: [{np.min(gravity):.6e}, {np.max(gravity):.6e}] m/s²")

        if gradient is not None:
            print(f"   梯度范围: [{np.min(gradient):.6e}, {np.max(gradient):.6e}] 1/s²")
        else:
            print("   ⚠️ 警告：样本中没有梯度数据！")
            return False

    except Exception as e:
        print(f"❌ 样本准备失败: {str(e)}")
        return False

    # ==================== 3. 训练DNN模型（联合训练）====================
    print("\n步骤 3/6: 训练DNN模型（同时学习引力和引力梯度）")

    print("准备训练数据...")
    model_config = phase1_config["model"]
    train_config = phase1_config["training"]

    # 初始化模型
    dnn_model = GravityAndGradientDNN(
        input_dim=model_config["input_dim"],
        gravity_dim=model_config["gravity_dim"],
        gradient_dim=model_config["gradient_dim"],
    )

    trainer = GravityGradientTrainer(
        dnn_model,
        gravity_weight=train_config["gravity_weight"],
        gradient_weight=train_config["gradient_weight"],
    )

    print(f"模型架构:")
    print(f"  输入维度: {model_config['input_dim']}")
    print(f"  引力输出: {model_config['gravity_dim']}")
    print(f"  梯度输出: {model_config['gradient_dim']}")
    print(f"训练参数:")
    print(f"  Epochs: {train_config['epochs']}")
    print(f"  Batch size: {train_config['batch_size']}")
    print(f"  Learning rate: {train_config['learning_rate']}")
    print(f"  引力损失权重: {train_config['gravity_weight']}")
    print(f"  梯度损失权重: {train_config['gradient_weight']}")

    try:
        train_loader, val_loader = trainer.prepare_data(
            points, gravity, gradient, batch_size=train_config["batch_size"]
        )
        print(f"✅ 数据准备完成（同时包含引力和梯度）")
    except Exception as e:
        print(f"❌ 数据准备失败: {str(e)}")
        return False

    # 训练模型
    print("开始联合训练...")
    try:
        trainer.train(
            train_loader,
            val_loader,
            epochs=train_config["epochs"],
            print_freq=train_config["print_freq"],
        )
        print(f"✅ 模型训练完成（同时学习引力和梯度）")
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        return False

    # ==================== 4. 验证模型精度（引力和梯度）====================
    print("\n步骤 4/6: 验证模型精度（引力+梯度）")
    try:
        test_size = min(100, len(points))
        pred_gravity, pred_gradient = dnn_model.predict(points[:test_size])

        # 计算引力误差
        gravity_error = np.abs(pred_gravity - gravity[:test_size])
        gravity_error_pct = (
            np.mean(gravity_error / (np.abs(gravity[:test_size]) + 1e-10)) * 100
        )

        # 计算梯度误差
        gradient_error = np.abs(pred_gradient - gradient[:test_size])
        gradient_error_pct = (
            np.mean(gradient_error / (np.abs(gradient[:test_size]) + 1e-10)) * 100
        )

        print(f"平均引力预测误差: {gravity_error_pct:.2f}%")
        print(f"平均梯度预测误差: {gradient_error_pct:.2f}%")

        if gravity_error_pct < 10 and gradient_error_pct < 10:
            print(f"✅ 模型精度满足要求 (引力<10%, 梯度<10%)")
        else:
            print(f"⚠️ 模型精度较低，建议增加训练样本或epoch")
    except Exception as e:
        print(f"❌ 精度验证失败: {str(e)}")
        return False

    # ==================== 5. 保存模型 ====================
    print("\n步骤 5/6: 保存训练好的模型")
    model_file = phase1_config["output"]["model_file"]
    try:
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        dnn_model.save_model(model_file)
        print(f"✅ 模型已保存: {model_file}")
        print(f"模型同时包含引力和梯度预测能力")
    except Exception as e:
        print(f"❌ 模型保存失败: {str(e)}")
        return False

    # ==================== 6. 绘制验证图表 ====================
    if phase1_config["output"]["save_plots"]:
        print("\n步骤 6/6: 生成验证图表...")
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # 引力大小分布
            ax = axes[0, 0]
            gravity_mag = np.linalg.norm(gravity, axis=1)
            ax.hist(gravity_mag, bins=30, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Gravity Magnitude (m/s²)")
            ax.set_ylabel("Frequency")
            ax.set_title("Gravity Magnitude Distribution")
            ax.grid(True, alpha=0.3)

            # 引力梯度大小分布
            ax = axes[0, 1]
            gradient_mag = np.linalg.norm(gradient, axis=1)
            ax.hist(gradient_mag, bins=30, edgecolor="black", alpha=0.7, color="orange")
            ax.set_xlabel("Gradient Magnitude (1/s²)")
            ax.set_ylabel("Frequency")
            ax.set_title("Gravity Gradient Magnitude Distribution")
            ax.grid(True, alpha=0.3)

            # 样本点空间分布
            ax = axes[0, 2]
            scatter = ax.scatter(
                points[:, 0], points[:, 1], c=gravity_mag, cmap="viridis", s=1
            )
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title("Sample Points Distribution (colored by gravity)")
            plt.colorbar(scatter, ax=ax, label="Gravity (m/s²)")

            # 引力预测vs真实
            ax = axes[1, 0]
            true_grav_mag = np.linalg.norm(gravity[:test_size], axis=1)
            pred_grav_mag = np.linalg.norm(pred_gravity, axis=1)
            ax.scatter(true_grav_mag, pred_grav_mag, alpha=0.5, s=20)
            ax.plot(
                [true_grav_mag.min(), true_grav_mag.max()],
                [true_grav_mag.min(), true_grav_mag.max()],
                "r--",
                lw=2,
            )
            ax.set_xlabel("True Gravity (m/s²)")
            ax.set_ylabel("Predicted Gravity (m/s²)")
            ax.set_title(f"Gravity Prediction (error: {gravity_error_pct:.2f}%)")
            ax.grid(True, alpha=0.3)

            # 梯度预测vs真实
            ax = axes[1, 1]
            true_grad_mag = np.linalg.norm(gradient[:test_size], axis=1)
            pred_grad_mag = np.linalg.norm(pred_gradient, axis=1)
            ax.scatter(true_grad_mag, pred_grad_mag, alpha=0.5, s=20, color="orange")
            ax.plot(
                [true_grad_mag.min(), true_grad_mag.max()],
                [true_grad_mag.min(), true_grad_mag.max()],
                "r--",
                lw=2,
            )
            ax.set_xlabel("True Gradient (1/s²)")
            ax.set_ylabel("Predicted Gradient (1/s²)")
            ax.set_title(f"Gradient Prediction (error: {gradient_error_pct:.2f}%)")
            ax.grid(True, alpha=0.3)

            # 训练损失曲线
            ax = axes[1, 2]
            if trainer.history["train_total_loss"]:
                ax.plot(
                    trainer.history["train_total_loss"], label="Total Loss", linewidth=2
                )
                ax.plot(
                    trainer.history["train_gravity_loss"],
                    label="Gravity Loss",
                    alpha=0.7,
                )
                ax.plot(
                    trainer.history["train_gradient_loss"],
                    label="Gradient Loss",
                    alpha=0.7,
                )
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Training History")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale("log")

            plt.tight_layout()
            fig_file = "results/phase1/verification_results.png"
            plt.savefig(fig_file, dpi=150, bbox_inches="tight")
            print(f"✅ 图表已保存: {fig_file}")
            plt.close()
        except Exception as e:
            print(f"⚠️ 图表生成失败: {str(e)}")

    # ==================== 总结 ====================
    print("\n" + "=" * 60)
    print("✅ 第一阶段验证完成！")
    print("=" * 60)
    print(f"统计信息:")
    print(f"  样本数量: {len(points)}")
    if radius > 0:
        print(f"  小行星半径: {radius:.2f} m")
    print(f"  引力预测误差: {gravity_error_pct:.2f}%")
    print(f"  梯度预测误差: {gradient_error_pct:.2f}%")
    print(f"\n输出文件:")
    print(f"  模型: {model_file}")
    print(f"  结果: results/phase1/")
    print(f"\n模型能力:")
    print(f"  ✅ 引力场预测")
    print(f"  ✅ 引力梯度预测")
    print(f"\n下一步:")
    print(f"  运行第二阶段: python verify/verify_phase2.py")
    print("=" * 60)

    return True


if __name__ == "__main__":
    # 切换到项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"工作目录: {os.getcwd()}")

    success = verify_phase1()
    sys.exit(0 if success else 1)
