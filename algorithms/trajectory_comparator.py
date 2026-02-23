#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 结果对比分析工具

提供多种轨迹优化算法结果的对比分析功能：
1. 多方法结果可视化对比
2. 收敛性分析
3. 计算效率评估
4. 轨迹质量指标（燃料消耗、终端误差、平滑度等）
5. 生成综合评估报告

支持对比的方法：
- 打靶法 (Shooting Method)
- 多重打靶法 (Multiple Shooting)
- 伪谱法 (Pseudospectral)
- 同伦法 (Homotopy)
- 自适应同伦法 (Adaptive Homotopy)
- 混合优化法 (Hybrid Optimization)
- 直接法 (Direct Method)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import time
import pickle
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TrajectoryMetrics:
    """轨迹质量指标数据类"""

    method_name: str
    success: bool
    fuel_consumption: float
    pos_error: float
    vel_error: float
    computation_time: float
    iterations: int
    final_mass: float

    # 可选指标
    smoothness: Optional[float] = None
    max_thrust: Optional[float] = None
    avg_thrust: Optional[float] = None
    thrust_switch_count: Optional[int] = None


class TrajectoryComparator:
    """
    轨迹优化结果对比分析器

    Attributes:
        results: 存储各方法的结果字典
        metrics: 各方法的指标数据
        reference_result: 参考结果（用于计算相对误差）
    """

    def __init__(self, save_dir: Optional[str] = None):
        """
        初始化对比分析器

        Parameters:
            save_dir: 结果保存目录
        """
        self.results: Dict[str, Dict] = {}
        self.metrics: Dict[str, TrajectoryMetrics] = {}
        self.reference_result: Optional[str] = None
        self.save_dir = Path(save_dir) if save_dir else Path("results/comparison")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def add_result(
        self, method_name: str, result: Dict, computation_time: Optional[float] = None
    ):
        """
        添加优化结果

        Parameters:
            method_name: 方法名称
            result: 优化结果字典
            computation_time: 计算时间（如果结果中未包含）
        """
        self.results[method_name] = result

        # 提取指标
        metrics = TrajectoryMetrics(
            method_name=method_name,
            success=result.get("success", False),
            fuel_consumption=result.get("fuel_consumption", 0.0),
            pos_error=result.get("pos_error", float("inf")),
            vel_error=result.get("vel_error", 0.0),
            computation_time=result.get("time", computation_time or 0.0),
            iterations=result.get("iterations", result.get("total_steps", 0)),
            final_mass=result.get("final_mass", 0.0),
        )

        # 计算额外指标
        if "u" in result and len(result["u"]) > 0:
            metrics.max_thrust = np.max(result["u"])
            metrics.avg_thrust = np.mean(result["u"])
            # 计算开关次数
            u_diff = np.diff(result["u"])
            metrics.thrust_switch_count = np.sum(np.abs(u_diff) > 0.1)

        # 计算平滑度
        if "r" in result and len(result["r"]) > 1:
            r = result["r"]
            if len(r.shape) == 2:
                # 计算曲率变化
                curvature = self._compute_curvature(r)
                metrics.smoothness = np.mean(curvature)

        self.metrics[method_name] = metrics

    def _compute_curvature(self, trajectory: np.ndarray) -> np.ndarray:
        """计算轨迹曲率"""
        n = len(trajectory)
        if n < 3:
            return np.array([0.0])

        curvature = []
        for i in range(1, n - 1):
            v1 = trajectory[i] - trajectory[i - 1]
            v2 = trajectory[i + 1] - trajectory[i]

            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 1e-10 and v2_norm > 1e-10:
                angle = np.arccos(np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1, 1))
                curvature.append(angle)
            else:
                curvature.append(0.0)

        return np.array(curvature)

    def set_reference(self, method_name: str):
        """设置参考方法"""
        if method_name in self.results:
            self.reference_result = method_name
        else:
            raise ValueError(f"未知方法: {method_name}")

    def compare_all(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        对比所有方法

        Returns:
            DataFrame: 对比结果表
        """
        if not self.metrics:
            print("没有结果可供对比")
            return pd.DataFrame()

        data = []
        for method, metrics in self.metrics.items():
            data.append(asdict(metrics))

        df = pd.DataFrame(data)

        # 按燃料消耗排序
        df = df.sort_values("fuel_consumption")

        # 显示结果
        print("\n" + "=" * 80)
        print("📊 轨迹优化方法对比")
        print("=" * 80)
        print(df.to_string(index=False))

        # 保存到文件
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\n✅ 对比结果已保存: {save_path}")

        return df

    def plot_comparison_dashboard(
        self,
        methods: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ):
        """
        绘制综合对比仪表板

        Parameters:
            methods: 要对比的方法列表（None表示全部）
            save_path: 保存路径
        """
        if methods is None:
            methods = list(self.results.keys())

        if len(methods) == 0:
            print("没有方法可供对比")
            return

        # 创建大图
        fig = plt.figure(figsize=(20, 14))

        # 颜色映射
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

        # ===== 1. 3D轨迹对比 =====
        ax = fig.add_subplot(3, 4, 1, projection="3d")
        for i, method in enumerate(methods):
            result = self.results[method]
            if "r" in result and len(result["r"]) > 0:
                r = result["r"]
                ax.plot(
                    r[:, 0],
                    r[:, 1],
                    r[:, 2],
                    color=colors[i],
                    linewidth=2,
                    label=method,
                )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(fontsize=8)
        ax.set_title("3D轨迹对比", fontsize=12, fontweight="bold")

        # ===== 2. 燃料消耗对比（柱状图）=====
        ax = fig.add_subplot(3, 4, 2)
        fuels = [self.metrics[m].fuel_consumption for m in methods]
        bars = ax.bar(range(len(methods)), fuels, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("燃料消耗 (kg)")
        ax.set_title("燃料消耗对比", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # 标注数值
        for i, (bar, fuel) in enumerate(zip(bars, fuels)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{fuel:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # ===== 3. 位置误差对比（对数柱状图）=====
        ax = fig.add_subplot(3, 4, 3)
        pos_errors = [max(self.metrics[m].pos_error, 1e-3) for m in methods]
        bars = ax.bar(range(len(methods)), pos_errors, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("位置误差 (m)")
        ax.set_yscale("log")
        ax.set_title("位置误差对比 (对数)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # ===== 4. 计算时间对比 =====
        ax = fig.add_subplot(3, 4, 4)
        times = [self.metrics[m].computation_time for m in methods]
        bars = ax.bar(range(len(methods)), times, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("计算时间 (s)")
        ax.set_title("计算效率对比", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # ===== 5-7. 各方法X/Y/Z位置对比 =====
        coords = ["X", "Y", "Z"]
        for idx, coord in enumerate(coords):
            ax = fig.add_subplot(3, 4, 5 + idx)
            for i, method in enumerate(methods):
                result = self.results[method]
                if "r" in result and "t" in result:
                    r = result["r"]
                    t = result["t"]
                    ax.plot(t, r[:, idx], color=colors[i], linewidth=2, label=method)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel(f"{coord} (m)")
            ax.set_title(f"{coord}位置对比", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # ===== 8. 推力历史对比 =====
        ax = fig.add_subplot(3, 4, 8)
        for i, method in enumerate(methods):
            result = self.results[method]
            if "u" in result and "t" in result:
                u = result["u"]
                t = result["t"]
                ax.plot(t, u, color=colors[i], linewidth=2, label=method)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("推力比")
        ax.set_title("推力历史对比", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ===== 9. 速度对比 =====
        ax = fig.add_subplot(3, 4, 9)
        for i, method in enumerate(methods):
            result = self.results[method]
            if "v" in result and "t" in result:
                v = result["v"]
                t = result["t"]
                v_mag = np.linalg.norm(v, axis=1)
                ax.plot(t, v_mag, color=colors[i], linewidth=2, label=method)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("速度 (m/s)")
        ax.set_title("速度大小对比", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ===== 10. 质量变化对比 =====
        ax = fig.add_subplot(3, 4, 10)
        for i, method in enumerate(methods):
            result = self.results[method]
            if "m" in result and "t" in result:
                m = result["m"]
                t = result["t"]
                ax.plot(t, m, color=colors[i], linewidth=2, label=method)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("质量 (kg)")
        ax.set_title("质量变化对比", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ===== 11. 收敛迭代次数对比 =====
        ax = fig.add_subplot(3, 4, 11)
        iterations = [self.metrics[m].iterations for m in methods]
        bars = ax.bar(range(len(methods)), iterations, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("迭代次数")
        ax.set_title("收敛迭代次数", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # ===== 12. 综合评分雷达图 =====
        ax = fig.add_subplot(3, 4, 12, projection="polar")
        self._plot_radar_chart(ax, methods, colors)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 对比仪表板已保存: {save_path}")

        plt.show()

    def _plot_radar_chart(self, ax, methods: List[str], colors):
        """绘制雷达图"""
        # 指标：燃料效率、精度、速度、成功率、平滑度
        categories = ["燃料效率", "精度", "速度", "成功率", "平滑度"]
        N = len(categories)

        # 计算角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        for i, method in enumerate(methods[:4]):  # 最多显示4个方法
            metrics = self.metrics[method]

            # 归一化评分 (0-1)
            # 燃料效率：燃料消耗越少越好
            all_fuels = [self.metrics[m].fuel_consumption for m in methods]
            fuel_score = 1 - (metrics.fuel_consumption - min(all_fuels)) / (
                max(all_fuels) - min(all_fuels) + 1e-10
            )

            # 精度：误差越小越好
            all_errors = [self.metrics[m].pos_error for m in methods]
            accuracy_score = 1 - (metrics.pos_error - min(all_errors)) / (
                max(all_errors) - min(all_errors) + 1e-10
            )

            # 速度：时间越短越好
            all_times = [self.metrics[m].computation_time for m in methods]
            speed_score = 1 - (metrics.computation_time - min(all_times)) / (
                max(all_times) - min(all_times) + 1e-10
            )

            # 成功率
            success_score = 1.0 if metrics.success else 0.0

            # 平滑度
            smoothness_score = (
                0.8 if metrics.smoothness is None else max(0, 1 - metrics.smoothness)
            )

            values = [
                fuel_score,
                accuracy_score,
                speed_score,
                success_score,
                smoothness_score,
            ]
            values += values[:1]

            ax.plot(angles, values, "o-", linewidth=2, color=colors[i], label=method)
            ax.fill(angles, values, alpha=0.15, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)
        ax.set_title("综合评分雷达图", fontsize=12, fontweight="bold", pad=20)

    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        生成综合分析报告

        Returns:
            str: 报告文本
        """
        lines = []
        lines.append("=" * 80)
        lines.append("小行星附着轨迹优化 - 方法对比分析报告")
        lines.append("=" * 80)
        lines.append("")

        # 1. 概览
        lines.append("📊 概览")
        lines.append("-" * 80)
        lines.append(f"对比方法数: {len(self.results)}")
        lines.append(
            f"成功方法数: {sum(1 for m in self.metrics.values() if m.success)}"
        )
        lines.append("")

        # 2. 详细对比
        lines.append("📈 详细指标对比")
        lines.append("-" * 80)

        # 创建表格
        df = self.compare_all()
        lines.append(df.to_string(index=False))
        lines.append("")

        # 3. 最佳方法分析
        lines.append("🏆 最佳方法分析")
        lines.append("-" * 80)

        # 最省燃料
        best_fuel = min(self.metrics.items(), key=lambda x: x[1].fuel_consumption)
        lines.append(
            f"最省燃料: {best_fuel[0]} ({best_fuel[1].fuel_consumption:.2f} kg)"
        )

        # 最高精度
        best_accuracy = min(self.metrics.items(), key=lambda x: x[1].pos_error)
        lines.append(
            f"最高精度: {best_accuracy[0]} (误差 {best_accuracy[1].pos_error:.2f} m)"
        )

        # 最快计算
        best_speed = min(self.metrics.items(), key=lambda x: x[1].computation_time)
        lines.append(
            f"最快计算: {best_speed[0]} ({best_speed[1].computation_time:.2f} s)"
        )

        # 最高成功率
        successful = [m for m in self.metrics.values() if m.success]
        if successful:
            lines.append(f"成功方法: {', '.join(m.method_name for m in successful)}")
        lines.append("")

        # 4. 建议
        lines.append("💡 推荐建议")
        lines.append("-" * 80)

        if best_fuel[0] == best_accuracy[0]:
            lines.append(f"推荐使用 {best_fuel[0]}：兼顾燃料效率和精度")
        else:
            lines.append(f"追求燃料效率: 使用 {best_fuel[0]}")
            lines.append(f"追求高精度: 使用 {best_accuracy[0]}")

        if best_speed[1].computation_time < 10:
            lines.append(f"追求快速计算: 使用 {best_speed[0]}")

        lines.append("")
        lines.append("=" * 80)

        report = "\n".join(lines)
        print(report)

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n✅ 报告已保存: {save_path}")

        return report

    def plot_convergence_analysis(self, save_path: Optional[str] = None):
        """
        绘制收敛性分析图
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        methods = list(self.results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

        # 1. 燃料消耗 vs 计算时间
        ax = axes[0, 0]
        for i, method in enumerate(methods):
            m = self.metrics[method]
            marker = "o" if m.success else "x"
            ax.scatter(
                m.computation_time,
                m.fuel_consumption,
                c=[colors[i]],
                s=100,
                marker=marker,
                label=method,
            )
        ax.set_xlabel("计算时间 (s)")
        ax.set_ylabel("燃料消耗 (kg)")
        ax.set_title("效率-性能权衡")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. 位置误差 vs 燃料消耗
        ax = axes[0, 1]
        for i, method in enumerate(methods):
            m = self.metrics[method]
            marker = "o" if m.success else "x"
            ax.scatter(
                m.fuel_consumption,
                m.pos_error,
                c=[colors[i]],
                s=100,
                marker=marker,
                label=method,
            )
        ax.set_xlabel("燃料消耗 (kg)")
        ax.set_ylabel("位置误差 (m)")
        ax.set_yscale("log")
        ax.set_title("燃料-精度权衡")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. 成功率饼图
        ax = axes[1, 0]
        success_count = sum(1 for m in self.metrics.values() if m.success)
        fail_count = len(self.metrics) - success_count
        ax.pie(
            [success_count, fail_count],
            labels=["成功", "失败"],
            autopct="%1.1f%%",
            colors=["#2ecc71", "#e74c3c"],
        )
        ax.set_title("成功率统计")

        # 4. 迭代次数分布
        ax = axes[1, 1]
        iter_data = [self.metrics[m].iterations for m in methods]
        ax.bar(range(len(methods)), iter_data, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("迭代次数")
        ax.set_title("收敛迭代次数分布")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 收敛分析图已保存: {save_path}")

        plt.show()

    def save_comparison(self, filepath: str):
        """保存对比结果到文件"""
        data = {
            "results": self.results,
            "metrics": {k: asdict(v) for k, v in self.metrics.items()},
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"✅ 对比结果已保存: {filepath}")

    def load_comparison(self, filepath: str):
        """从文件加载对比结果"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.results = data["results"]
        for k, v in data["metrics"].items():
            self.metrics[k] = TrajectoryMetrics(**v)

        print(f"✅ 对比结果已加载: {filepath}")


# 示例使用
if __name__ == "__main__":
    print("=" * 80)
    print("轨迹优化结果对比分析工具示例")
    print("=" * 80)

    # 创建对比分析器
    comparator = TrajectoryComparator(save_dir="results/comparison")

    # 模拟一些结果数据（实际使用时替换为真实优化结果）
    np.random.seed(42)

    for method in ["伪谱法", "打靶法", "同伦法", "混合优化"]:
        # 模拟结果
        result = {
            "success": True,
            "fuel_consumption": np.random.uniform(40, 60),
            "pos_error": np.random.uniform(50, 500),
            "vel_error": np.random.uniform(0.1, 2.0),
            "time": np.random.uniform(5, 60),
            "iterations": np.random.randint(20, 200),
            "final_mass": 1000 - np.random.uniform(40, 60),
            "t": np.linspace(0, 770, 100),
            "r": np.random.randn(100, 3).cumsum(axis=0) * 10
            + np.array([5000, 3000, 4000]),
            "v": np.random.randn(100, 3) * 5,
            "m": np.linspace(1000, 950, 100),
            "u": np.random.choice([0, 1], 100, p=[0.6, 0.4]).astype(float),
        }

        comparator.add_result(method, result)

    # 生成对比
    comparator.compare_all()

    # 生成报告
    comparator.generate_report()

    # 绘制对比图
    comparator.plot_comparison_dashboard()

    # 绘制收敛分析
    comparator.plot_convergence_analysis()

    print("\n✅ 对比分析完成！")
