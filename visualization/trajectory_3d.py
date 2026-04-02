#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D轨迹可视化模块

提供专业、美观的3D轨迹可视化功能。
支持：
- 独立3D轨迹图
- 时间颜色编码
- 推力方向箭头
- 小行星模型显示
- 多轨迹对比
- 交互式动画
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from typing import Dict, List, Optional, Tuple, Union
import warnings
import os

# Plotly交互式可视化（可选）
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def load_asteroid_mesh(ply_file: str, scale: float = 1.0):
    """
    加载PLY小行星模型
    
    Parameters:
        ply_file: PLY文件路径
        scale: 缩放因子
        
    Returns:
        vertices: 顶点坐标 (N, 3)
        faces: 面索引 (M, 3)
    """
    try:
        from plyfile import PlyData
    except ImportError:
        print("警告: plyfile未安装，请运行 'pip install plyfile'")
        return None, None
    
    if not os.path.exists(ply_file):
        print(f"警告: PLY文件不存在: {ply_file}")
        return None, None
    
    ply_data = PlyData.read(ply_file)
    
    # 提取顶点
    vertex = ply_data['vertex']
    vertices = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
    
    # 提取面
    face = ply_data['face']
    faces = []
    for f in face:
        face_data = f[0]
        if isinstance(face_data, tuple):
            faces.append(face_data[1])
        else:
            faces.append(face_data)
    faces = np.array(faces, dtype=int)
    
    # 缩放
    if scale != 1.0:
        vertices = vertices * scale
    
    return vertices, faces


class TrajectoryVisualizer3D:
    """
    3D轨迹可视化器
    
    提供丰富的3D轨迹可视化功能，包括：
    - 时间颜色编码轨迹
    - 推力方向箭头
    - 起点终点标记
    - 小行星简化模型
    - 误差分析可视化
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 10),
        dark_mode: bool = True,
        dpi: int = 150,
    ):
        """
        初始化可视化器
        
        Parameters:
            figsize: 图形大小
            dark_mode: 是否使用深色主题
            dpi: 图形分辨率
        """
        self.figsize = figsize
        self.dark_mode = dark_mode
        self.dpi = dpi
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置深色主题
        if dark_mode:
            plt.rcParams.update({
                'figure.facecolor': '#1a1a2e',
                'axes.facecolor': '#16213e',
                'axes.edgecolor': '#e94560',
                'axes.labelcolor': '#ffffff',
                'text.color': '#ffffff',
                'xtick.color': '#ffffff',
                'ytick.color': '#ffffff',
                'grid.color': '#0f3460',
                'axes.grid': True,
            })
    
    def plot_trajectory(
        self,
        result: Dict,
        r0: Optional[np.ndarray] = None,
        rf: Optional[np.ndarray] = None,
        title: str = "航天器附着轨迹",
        show_thrust: bool = True,
        show_asteroid: bool = True,
        asteroid_radius: float = 500.0,
        save_path: Optional[str] = None,
        show_colorbar: bool = True,
        elev: float = 25,
        azim: float = 45,
    ) -> plt.Figure:
        """
        绘制3D轨迹图
        
        Parameters:
            result: 优化结果字典，包含 'r', 't', 'U' 等字段
            r0: 初始位置（可选，会从result中提取）
            rf: 目标位置（可选，会从result中提取）
            title: 图标题
            show_thrust: 是否显示推力方向箭头
            show_asteroid: 是否显示小行星简化模型
            asteroid_radius: 小行星半径
            save_path: 保存路径
            show_colorbar: 是否显示颜色条
            elev: 仰角
            azim: 方位角
            
        Returns:
            matplotlib Figure对象
        """
        # 提取数据
        positions = result.get('r', result.get('positions', None))
        if positions is None:
            raise ValueError("结果中未找到位置数据 'r' 或 'positions'")
        
        times = result.get('t', result.get('times', None))
        if times is None:
            times = np.linspace(0, 1, len(positions))
        
        controls = result.get('U', result.get('controls', None))
        
        # 获取起点终点
        if r0 is None:
            r0 = positions[0]
        if rf is None:
            rf = positions[-1]
        
        # 创建图形
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置背景颜色
        if self.dark_mode:
            ax.set_facecolor('#16213e')
            fig.patch.set_facecolor('#1a1a2e')
        
        # 创建时间颜色编码的轨迹
        # 使用线段集合实现渐变色
        points = positions.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # 归一化时间作为颜色
        t_norm = (times[:-1] - times[0]) / (times[-1] - times[0])
        colors = cm.plasma(t_norm)
        
        # 创建3D线段集合
        lc = Line3DCollection(segments, colors=colors, linewidths=2.5, alpha=0.9)
        ax.add_collection3d(lc)
        
        # 添加推力方向箭头
        if show_thrust and controls is not None:
            thrust_mag = np.linalg.norm(controls, axis=1)
            max_thrust = np.max(thrust_mag) if np.max(thrust_mag) > 0 else 1.0
            
            # 每隔N个点显示一个箭头
            n_arrows = min(15, len(positions))
            indices = np.linspace(0, len(positions) - 2, n_arrows, dtype=int)
            
            for i in indices:
                if thrust_mag[i] > 0.05 * max_thrust:
                    # 箭头方向
                    direction = controls[i] / np.linalg.norm(controls[i])
                    arrow_length = 150  # 箭头长度
                    
                    if self.dark_mode:
                        color = '#00ff88'
                    else:
                        color = 'green'
                    
                    ax.quiver(
                        positions[i, 0], positions[i, 1], positions[i, 2],
                        direction[0] * arrow_length,
                        direction[1] * arrow_length,
                        direction[2] * arrow_length,
                        color=color, alpha=0.7, arrow_length_ratio=0.3,
                        linewidth=1.5
                    )
        
        # 绘制起点
        ax.scatter(
            r0[0], r0[1], r0[2],
            c='#ff6b6b', s=200, marker='o',
            label='起点', edgecolors='white', linewidths=2,
            zorder=10
        )
        
        # 绘制目标点
        ax.scatter(
            rf[0], rf[1], rf[2],
            c='#4ecdc4', s=250, marker='*',
            label='目标点', edgecolors='white', linewidths=2,
            zorder=10
        )
        
        # 绘制终点（实际到达位置）
        ax.scatter(
            positions[-1, 0], positions[-1, 1], positions[-1, 2],
            c='#ffe66d', s=150, marker='^',
            label='终点', edgecolors='white', linewidths=2,
            zorder=10
        )
        
        # 绘制小行星简化模型（球体）
        if show_asteroid:
            # 创建球体网格
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            x_sphere = asteroid_radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = asteroid_radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = asteroid_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # 绘制球体表面
            if self.dark_mode:
                ax.plot_surface(
                    x_sphere, y_sphere, z_sphere,
                    color='#8b5cf6', alpha=0.15,
                    linewidth=0, antialiased=True
                )
            else:
                ax.plot_surface(
                    x_sphere, y_sphere, z_sphere,
                    color='gray', alpha=0.2,
                    linewidth=0, antialiased=True
                )
        
        # 设置坐标轴
        ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
        ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)
        
        # 设置视角
        ax.view_init(elev=elev, azim=azim)
        
        # 设置坐标轴范围
        all_points = np.vstack([positions, r0.reshape(1, -1), rf.reshape(1, -1)])
        margin = 0.1 * np.max(np.ptp(all_points, axis=0))
        
        ax.set_xlim([all_points[:, 0].min() - margin, all_points[:, 0].max() + margin])
        ax.set_ylim([all_points[:, 1].min() - margin, all_points[:, 1].max() + margin])
        ax.set_zlim([all_points[:, 2].min() - margin, all_points[:, 2].max() + margin])
        
        # 添加颜色条
        if show_colorbar:
            sm = cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(times[0], times[-1]))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.1)
            cbar.set_label('时间 (s)', fontsize=11)
        
        # 添加标题和图例
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.8)
        
        # 添加统计信息
        if 'fuel_consumption' in result:
            info_text = f"燃料消耗: {result['fuel_consumption']:.2f} kg"
            if 'pos_error' in result:
                info_text += f"\n位置误差: {result['pos_error']:.4f} m"
            if 'vel_error' in result:
                info_text += f"\n速度误差: {result['vel_error']:.4f} m/s"
            
            ax.text2D(
                0.02, 0.98, info_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.8) if self.dark_mode else dict(boxstyle='round', facecolor='white', alpha=0.8),
                color='white' if self.dark_mode else 'black'
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=fig.get_facecolor(), edgecolor='none')
            print(f"图像已保存至: {save_path}")
        
        return fig
    
    def plot_trajectory_animation(
        self,
        result: Dict,
        r0: Optional[np.ndarray] = None,
        rf: Optional[np.ndarray] = None,
        interval: int = 50,
        save_path: Optional[str] = None,
    ):
        """
        绘制轨迹动画
        
        Parameters:
            result: 优化结果
            r0: 初始位置
            rf: 目标位置
            interval: 动画帧间隔(ms)
            save_path: 保存路径(支持.gif)
        """
        from matplotlib.animation import FuncAnimation
        
        positions = result.get('r', result.get('positions', None))
        if positions is None:
            raise ValueError("结果中未找到位置数据")
        
        if r0 is None:
            r0 = positions[0]
        if rf is None:
            rf = positions[-1]
        
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        if self.dark_mode:
            ax.set_facecolor('#16213e')
            fig.patch.set_facecolor('#1a1a2e')
        
        # 初始化轨迹线
        line, = ax.plot([], [], [], 'w-', linewidth=2, alpha=0.8)
        point, = ax.plot([], [], [], 'wo', markersize=10)
        
        # 起点终点
        ax.scatter(r0[0], r0[1], r0[2], c='#ff6b6b', s=200, marker='o', label='起点')
        ax.scatter(rf[0], rf[1], rf[2], c='#4ecdc4', s=250, marker='*', label='目标')
        
        # 设置坐标轴范围
        margin = 0.1 * np.max(np.ptp(positions, axis=0))
        ax.set_xlim([positions[:, 0].min() - margin, positions[:, 0].max() + margin])
        ax.set_ylim([positions[:, 1].min() - margin, positions[:, 1].max() + margin])
        ax.set_zlim([positions[:, 2].min() - margin, positions[:, 2].max() + margin])
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point
        
        def animate(i):
            line.set_data(positions[:i, 0], positions[:i, 1])
            line.set_3d_properties(positions[:i, 2])
            if i > 0:
                point.set_data([positions[i-1, 0]], [positions[i-1, 1]])
                point.set_3d_properties([positions[i-1, 2]])
            return line, point
        
        anim = FuncAnimation(
            fig, animate, init_func=init,
            frames=len(positions), interval=interval, blit=True
        )
        
        if save_path and save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000//interval)
            print(f"动画已保存至: {save_path}")
        
        plt.show()
        return anim
    
    def plot_multi_view(
        self,
        result: Dict,
        r0: Optional[np.ndarray] = None,
        rf: Optional[np.ndarray] = None,
        title: str = "航天器附着轨迹 - 多视角",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制多视角3D轨迹图
        
        Parameters:
            result: 优化结果
            r0: 初始位置
            rf: 目标位置
            title: 图标题
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        positions = result.get('r', result.get('positions', None))
        if positions is None:
            raise ValueError("结果中未找到位置数据")
        
        if r0 is None:
            r0 = positions[0]
        if rf is None:
            rf = positions[-1]
        
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        
        # 四个视角
        views = [
            (30, 45, "视角1: 俯视斜角"),
            (0, 0, "视角2: XY平面"),
            (90, 0, "视角3: XZ平面"),
            (0, 90, "视角4: YZ平面"),
        ]
        
        for idx, (elev, azim, subtitle) in enumerate(views, 1):
            ax = fig.add_subplot(2, 2, idx, projection='3d')
            
            if self.dark_mode:
                ax.set_facecolor('#16213e')
            
            # 绘制轨迹
            ax.plot(
                positions[:, 0], positions[:, 1], positions[:, 2],
                color='#00ff88', linewidth=2, alpha=0.8
            )
            
            # 起点终点
            ax.scatter(r0[0], r0[1], r0[2], c='#ff6b6b', s=150, marker='o', label='起点')
            ax.scatter(rf[0], rf[1], rf[2], c='#4ecdc4', s=200, marker='*', label='目标')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(subtitle, fontsize=11)
            ax.view_init(elev=elev, azim=azim)
            ax.legend(fontsize=8)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=fig.get_facecolor(), edgecolor='none')
            print(f"图像已保存至: {save_path}")
        
        return fig


def plot_trajectory_3d(
    result: Dict,
    r0: Optional[np.ndarray] = None,
    rf: Optional[np.ndarray] = None,
    **kwargs
) -> plt.Figure:
    """
    快速绘制3D轨迹图的便捷函数
    
    Parameters:
        result: 优化结果
        r0: 初始位置
        rf: 目标位置
        **kwargs: 其他参数传递给 TrajectoryVisualizer3D.plot_trajectory
        
    Returns:
        matplotlib Figure对象
    """
    viz = TrajectoryVisualizer3D()
    return viz.plot_trajectory(result, r0, rf, **kwargs)


def plot_trajectory_comparison(
    results: List[Dict],
    labels: List[str],
    r0: Optional[np.ndarray] = None,
    rf: Optional[np.ndarray] = None,
    title: str = "Trajectory Optimization Results",
    save_path: Optional[str] = None,
    ply_file: Optional[str] = None,
    asteroid_radius: float = 700.0,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plot trajectory optimization results in scientific report style (2x3 subplots)
    
    Subplots:
    1. 3D trajectory + asteroid model
    2. Position components vs time
    3. Velocity components vs time
    4. Mass vs time
    5. Thrust vs time
    6. Position error vs time (log scale)
    
    Parameters:
        results: List of optimization results (uses first one)
        labels: Label list
        r0: Initial position
        rf: Target position
        title: Figure title
        save_path: Save path
        ply_file: PLY asteroid model file path
        asteroid_radius: Asteroid radius
        show_plot: Whether to show the plot
        
    Returns:
        matplotlib Figure object
    """
    # Use first result
    if not results:
        raise ValueError("No valid trajectory results")
    
    result = results[0]
    method = labels[0] if labels else 'unknown'
    
    # Extract data
    positions = result.get('r', result.get('positions', None))
    if positions is None:
        raise ValueError("Position data not found in results")
    
    times = result.get('t', np.linspace(0, 1, len(positions)))
    velocities = result.get('v', result.get('velocities', None))
    masses = result.get('m', result.get('masses', None))
    controls = result.get('U', result.get('controls', None))
    
    if r0 is None:
        r0 = positions[0]
    if rf is None:
        rf = positions[-1]
    
    # Calculate fuel consumption
    if masses is not None:
        fuel = masses[0] - masses[-1]
    else:
        fuel = result.get('fuel_consumption', 0)
    
    # Create figure (scientific report style: white background, default matplotlib colors)
    plt.style.use('default')
    fig = plt.figure(figsize=(14, 10), dpi=120)
    fig.patch.set_facecolor('white')
    
    # Main title
    fig.suptitle(f'Trajectory Optimization Results - {method.upper()} (Fuel: {fuel:.2f} kg)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # ==================== Subplot 1: 3D Trajectory ====================
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.set_facecolor('white')
    
    # Determine asteroid center position
    asteroid_center = rf
    
    # Load and plot asteroid polyhedron model
    if ply_file and os.path.exists(ply_file):
        vertices, faces = load_asteroid_mesh(ply_file)
        if vertices is not None and faces is not None:
            model_center = np.mean(vertices, axis=0)
            model_radius = np.max(np.linalg.norm(vertices - model_center, axis=1))
            scale = asteroid_radius / model_radius if model_radius > 0 else 1.0
            vertices = (vertices - model_center) * scale + asteroid_center
            
            mesh = Poly3DCollection(
                [[vertices[i] for i in face] for face in faces],
                alpha=0.3,
                facecolor='#9467BD',
                edgecolor='#7B68EE',
                linewidth=0.1
            )
            ax1.add_collection3d(mesh)
    
    # Plot trajectory
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
             'b-', linewidth=1.5, label='Trajectory')
    
    # Start and target points
    ax1.scatter(r0[0], r0[1], r0[2], c='red', s=80, marker='o', label='Start')
    ax1.scatter(rf[0], rf[1], rf[2], c='green', s=100, marker='^', label='Target')
    
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_title('3D Trajectory', fontsize=11)
    
    # ==================== Subplot 2: Position Components ====================
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(times, positions[:, 0], 'b-', linewidth=1.5, label='x')
    ax2.plot(times, positions[:, 1], color='#FF7F0E', linewidth=1.5, label='y')
    ax2.plot(times, positions[:, 2], 'g-', linewidth=1.5, label='z')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position Components', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([times[0], times[-1]])
    
    # ==================== Subplot 3: Velocity Components ====================
    ax3 = fig.add_subplot(2, 3, 3)
    if velocities is not None:
        ax3.plot(times, velocities[:, 0], 'b-', linewidth=1.5, label='vx')
        ax3.plot(times, velocities[:, 1], color='#FF7F0E', linewidth=1.5, label='vy')
        ax3.plot(times, velocities[:, 2], 'g-', linewidth=1.5, label='vz')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity Components', fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([times[0], times[-1]])
    
    # ==================== Subplot 4: Mass ====================
    ax4 = fig.add_subplot(2, 3, 4)
    if masses is not None:
        ax4.plot(times, masses, 'g-', linewidth=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Mass (kg)')
    ax4.set_title('Mass', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([times[0], times[-1]])
    
    # ==================== Subplot 5: Thrust ====================
    ax5 = fig.add_subplot(2, 3, 5)
    if controls is not None:
        thrust = np.linalg.norm(controls, axis=1)
        ax5.plot(times, thrust, 'r-', linewidth=1.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Thrust (N)')
    ax5.set_title('Thrust', fontsize=11)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([times[0], times[-1]])
    
    # ==================== Subplot 6: Position Error (log scale) ====================
    ax6 = fig.add_subplot(2, 3, 6)
    # Calculate position error
    if positions is not None:
        pos_errors = np.linalg.norm(positions - rf, axis=1)
        ax6.semilogy(times, pos_errors, color='#9467BD', linewidth=1.5)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Position Error (m)')
    ax6.set_title('Position Error', fontsize=11)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([times[0], times[-1]])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


# ==================== Plotly交互式可视化 ====================

class InteractiveTrajectoryVisualizer:
    """
    Plotly交互式3D轨迹可视化器
    
    支持：
    - 鼠标拖拽旋转
    - 滚轮缩放
    - 悬停显示数据
    - 时间滑块动画
    - 导出HTML
    """
    
    def __init__(self, dark_mode: bool = True):
        """
        初始化交互式可视化器
        
        Parameters:
            dark_mode: 是否使用深色主题
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly未安装，请运行: pip install plotly")
        
        self.dark_mode = dark_mode
        self.template = "plotly_dark" if dark_mode else "plotly_white"
    
    def plot_trajectory(
        self,
        result: Dict,
        r0: Optional[np.ndarray] = None,
        rf: Optional[np.ndarray] = None,
        title: str = "小行星附着最优轨迹",
        show_thrust: bool = True,
        show_asteroid: bool = True,
        asteroid_radius: float = 500.0,
        ply_file: Optional[str] = None,
        save_html: Optional[str] = None,
        auto_open: bool = True,
    ) -> go.Figure:
        """
        绘制交互式3D轨迹图
        
        Parameters:
            result: 优化结果字典
            r0: 初始位置
            rf: 目标位置
            title: 图标题
            show_thrust: 是否显示推力方向
            show_asteroid: 是否显示小行星
            asteroid_radius: 小行星半径（当不使用PLY时）
            ply_file: PLY文件路径（使用真实多面体模型）
            save_html: 保存为HTML的路径
            auto_open: 是否自动在浏览器中打开
            
        Returns:
            Plotly Figure对象
        """
        # 提取数据
        positions = result.get('r', result.get('positions', None))
        if positions is None:
            raise ValueError("结果中未找到位置数据")
        
        times = result.get('t', result.get('times', None))
        if times is None:
            times = np.linspace(0, 1, len(positions))
        
        velocities = result.get('v', result.get('velocities', None))
        masses = result.get('m', result.get('masses', None))
        controls = result.get('U', result.get('controls', None))
        
        if r0 is None:
            r0 = positions[0]
        if rf is None:
            rf = positions[-1]
        
        # 创建图形
        fig = go.Figure()
        
        # 轨迹线（带时间颜色）
        # 使用scatter3d的colorscale实现渐变
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines',
            name='轨迹',
            line=dict(
                width=4,
                color=times,
                colorscale='Turbo',
                showscale=True,
                colorbar=dict(title='时间 (s)', thickness=15, len=0.7),
            ),
            hovertemplate='<b>时间:</b> %{line.color:.1f} s<br>' +
                          '<b>位置:</b> (%{x:.1f}, %{y:.1f}, %{z:.1f}) m<extra></extra>',
        ))
        
        # 起点标记
        fig.add_trace(go.Scatter3d(
            x=[r0[0]], y=[r0[1]], z=[r0[2]],
            mode='markers',
            name='起点',
            marker=dict(
                size=6,
                color='#ff6b6b',
                symbol='circle',
                line=dict(color='white', width=1),
            ),
            hovertemplate='<b>起点</b><br>位置: (%{x:.1f}, %{y:.1f}, %{z:.1f}) m<extra></extra>',
        ))
        
        # 目标点标记
        fig.add_trace(go.Scatter3d(
            x=[rf[0]], y=[rf[1]], z=[rf[2]],
            mode='markers',
            name='目标点',
            marker=dict(
                size=8,
                color='#4ecdc4',
                symbol='diamond',
                line=dict(color='white', width=1),
            ),
            hovertemplate='<b>目标点</b><br>位置: (%{x:.1f}, %{y:.1f}, %{z:.1f}) m<extra></extra>',
        ))
        
        # 终点标记（实际到达位置）
        fig.add_trace(go.Scatter3d(
            x=[positions[-1, 0]], y=[positions[-1, 1]], z=[positions[-1, 2]],
            mode='markers',
            name='终点',
            marker=dict(
                size=6,
                color='#ffe66d',
                symbol='square',
                line=dict(color='white', width=1),
            ),
            hovertemplate='<b>终点</b><br>位置: (%{x:.1f}, %{y:.1f}, %{z:.1f}) m<extra></extra>',
        ))
        
        # 推力方向箭头（使用cone）
        if show_thrust and controls is not None:
            thrust_mag = np.linalg.norm(controls, axis=1)
            max_thrust = np.max(thrust_mag) if np.max(thrust_mag) > 0 else 1.0
            
            # 选择几个关键点显示箭头
            n_arrows = min(10, len(positions))
            indices = np.linspace(0, len(positions) - 2, n_arrows, dtype=int)
            
            arrow_data = []
            for i in indices:
                if thrust_mag[i] > 0.05 * max_thrust:
                    direction = controls[i] / np.linalg.norm(controls[i])
                    arrow_data.append({
                        'x': positions[i, 0],
                        'y': positions[i, 1],
                        'z': positions[i, 2],
                        'u': direction[0],
                        'v': direction[1],
                        'w': direction[2],
                        'norm': thrust_mag[i] / max_thrust,
                    })
            
            if arrow_data:
                arrow_df = np.array([[d['x'], d['y'], d['z'], d['u'], d['v'], d['w']] for d in arrow_data])
                
                fig.add_trace(go.Cone(
                    x=arrow_df[:, 0],
                    y=arrow_df[:, 1],
                    z=arrow_df[:, 2],
                    u=arrow_df[:, 3],
                    v=arrow_df[:, 4],
                    w=arrow_df[:, 5],
                    name='推力方向',
                    sizemode='absolute',
                    sizeref=200,
                    colorscale='Greens',
                    showscale=False,
                    hovertemplate='<b>推力方向</b><extra></extra>',
                ))
        
        # 小行星模型
        if show_asteroid:
            if ply_file and os.path.exists(ply_file):
                # 使用真实多面体模型
                vertices, faces = load_asteroid_mesh(ply_file)
                if vertices is not None and faces is not None:
                    # 计算合适的缩放比例，使模型适合显示
                    center = np.mean(vertices, axis=0)
                    max_dist = np.max(np.linalg.norm(vertices - center, axis=1))
                    scale = asteroid_radius / max_dist if max_dist > 0 else 1.0
                    vertices = (vertices - center) * scale
                    
                    fig.add_trace(go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        name='小行星',
                        opacity=0.6,
                        color='#8b5cf6',
                        hoverinfo='skip',
                        lighting=dict(
                            ambient=0.4,
                            diffuse=0.8,
                            fresnel=0.2,
                            specular=0.5,
                            roughness=0.5,
                        ),
                        lightposition=dict(x=100, y=100, z=100),
                    ))
                    print(f"已加载多面体模型: {os.path.basename(ply_file)}")
                    print(f"  顶点数: {len(vertices)}, 面数: {len(faces)}")
            else:
                # 使用简化球体
                u_sphere = np.linspace(0, 2 * np.pi, 30)
                v_sphere = np.linspace(0, np.pi, 20)
                
                x_sphere = asteroid_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
                y_sphere = asteroid_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
                z_sphere = asteroid_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
                
                fig.add_trace(go.Surface(
                    x=x_sphere, y=y_sphere, z=z_sphere,
                    name='小行星',
                    opacity=0.2,
                    colorscale=[[0, '#8b5cf6'], [1, '#8b5cf6']],
                    showscale=False,
                    hoverinfo='skip',
                ))
        
        # 设置布局
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color='white' if self.dark_mode else 'black'),
                x=0.5,
            ),
            scene=dict(
                xaxis=dict(title='X (m)', backgroundcolor='#16213e' if self.dark_mode else '#f5f5f5',
                          gridcolor='#0f3460' if self.dark_mode else '#e0e0e0',
                          color='white' if self.dark_mode else 'black'),
                yaxis=dict(title='Y (m)', backgroundcolor='#16213e' if self.dark_mode else '#f5f5f5',
                          gridcolor='#0f3460' if self.dark_mode else '#e0e0e0',
                          color='white' if self.dark_mode else 'black'),
                zaxis=dict(title='Z (m)', backgroundcolor='#16213e' if self.dark_mode else '#f5f5f5',
                          gridcolor='#0f3460' if self.dark_mode else '#e0e0e0',
                          color='white' if self.dark_mode else 'black'),
                bgcolor='#16213e' if self.dark_mode else '#ffffff',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    up=dict(x=0, y=0, z=1),
                ),
            ),
            paper_bgcolor='#1a1a2e' if self.dark_mode else '#ffffff',
            plot_bgcolor='#1a1a2e' if self.dark_mode else '#ffffff',
            font=dict(color='white' if self.dark_mode else 'black'),
            legend=dict(
                x=0.02, y=0.98,
                bgcolor='rgba(15, 52, 96, 0.8)' if self.dark_mode else 'rgba(255, 255, 255, 0.8)',
                bordercolor='#e94560' if self.dark_mode else '#333333',
                borderwidth=1,
            ),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        
        # 添加统计信息标注
        annotations = []
        info_parts = [f"<b>燃料消耗:</b> {result['fuel_consumption']:.2f} kg"]
        if 'pos_error' in result:
            info_parts.append(f"<b>位置误差:</b> {result['pos_error']:.4f} m")
        if 'vel_error' in result:
            info_parts.append(f"<b>速度误差:</b> {result['vel_error']:.4f} m/s")
        if 'elapsed_time' in result:
            info_parts.append(f"<b>计算时间:</b> {result['elapsed_time']:.2f} s")
        
        fig.add_annotation(
            text="<br>".join(info_parts),
            xref="paper", yref="paper",
            x=0.02, y=0.85,
            showarrow=False,
            font=dict(size=11, color='white' if self.dark_mode else 'black'),
            bgcolor='rgba(15, 52, 96, 0.8)' if self.dark_mode else 'rgba(255, 255, 255, 0.9)',
            bordercolor='#e94560' if self.dark_mode else '#333333',
            borderwidth=1,
            borderpad=5,
        )
        
        # 保存HTML
        if save_html:
            fig.write_html(save_html, include_plotlyjs='cdn')
            print(f"交互式图表已保存至: {save_html}")
            if auto_open:
                import webbrowser
                webbrowser.open(f"file://{save_html}")
        
        return fig
    
    def plot_trajectory_with_slider(
        self,
        result: Dict,
        r0: Optional[np.ndarray] = None,
        rf: Optional[np.ndarray] = None,
        title: str = "轨迹动画",
        save_html: Optional[str] = None,
    ) -> go.Figure:
        """
        带时间滑块的交互式轨迹动画
        
        Parameters:
            result: 优化结果
            r0: 初始位置
            rf: 目标位置
            title: 标题
            save_html: 保存路径
            
        Returns:
            Plotly Figure对象
        """
        positions = result.get('r', result.get('positions', None))
        if positions is None:
            raise ValueError("结果中未找到位置数据")
        
        times = result.get('t', np.linspace(0, 1, len(positions)))
        
        if r0 is None:
            r0 = positions[0]
        if rf is None:
            rf = positions[-1]
        
        # 创建带滑块的图形
        fig = go.Figure()
        
        # 全部轨迹（淡化显示）
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode='lines',
            name='完整轨迹',
            line=dict(width=1, color='rgba(100, 100, 100, 0.3)'),
            hoverinfo='skip',
        ))
        
        # 动态轨迹（随滑块变化）
        fig.add_trace(go.Scatter3d(
            x=positions[:1, 0], y=positions[:1, 1], z=positions[:1, 2],
            mode='lines+markers',
            name='当前轨迹',
            line=dict(width=4, color='#00ff88'),
            marker=dict(size=3, color=times[:1], colorscale='Plasma'),
        ))
        
        # 起点终点
        fig.add_trace(go.Scatter3d(
            x=[r0[0]], y=[r0[1]], z=[r0[2]],
            mode='markers', name='起点',
            marker=dict(size=6, color='#ff6b6b', symbol='circle'),
        ))
        fig.add_trace(go.Scatter3d(
            x=[rf[0]], y=[rf[1]], z=[rf[2]],
            mode='markers', name='目标',
            marker=dict(size=8, color='#4ecdc4', symbol='diamond'),
        ))
        
        # 创建滑块步骤
        n_frames = min(50, len(positions))
        indices = np.linspace(0, len(positions) - 1, n_frames, dtype=int)
        
        frames = []
        for i in indices:
            frame = go.Frame(
                name=f"frame_{i}",
                data=[go.Scatter3d(
                    x=positions[:i+1, 0],
                    y=positions[:i+1, 1],
                    z=positions[:i+1, 2],
                    mode='lines+markers',
                    line=dict(width=4, color='#00ff88'),
                    marker=dict(size=3, color=times[:i+1], colorscale='Plasma'),
                )],
            )
            frames.append(frame)
        
        fig.frames = frames
        
        # 滑块设置
        sliders = [dict(
            active=0,
            steps=[dict(
                method='animate',
                args=[[f'frame_{i}'], dict(mode='immediate', frame=dict(duration=0))],
                label=f'{times[i]:.0f}s'
            ) for i in indices],
            currentvalue=dict(prefix='时间: ', suffix=' s'),
            pad=dict(t=50),
        )]
        
        # 播放按钮
        updatemenus = [dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(label='播放', method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
                dict(label='暂停', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False))]),
            ],
            x=0.1, y=0, xanchor='left', yanchor='bottom',
        )]
        
        fig.update_layout(
            title=title,
            sliders=sliders,
            updatemenus=updatemenus,
            scene=dict(
                xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                bgcolor='#16213e' if self.dark_mode else '#ffffff',
            ),
            paper_bgcolor='#1a1a2e' if self.dark_mode else '#ffffff',
            font=dict(color='white' if self.dark_mode else 'black'),
        )
        
        if save_html:
            fig.write_html(save_html, include_plotlyjs='cdn')
            print(f"动画已保存至: {save_html}")
        
        return fig
    
    def plot_comparison(
        self,
        results: List[Dict],
        labels: List[str],
        r0: Optional[np.ndarray] = None,
        rf: Optional[np.ndarray] = None,
        title: str = "轨迹对比",
        save_html: Optional[str] = None,
        ply_file: Optional[str] = None,
        asteroid_radius: float = 700.0,
        asteroid_center: Optional[np.ndarray] = None,
    ) -> go.Figure:
        """
        多轨迹交互式对比
        
        Parameters:
            results: 多个优化结果
            labels: 标签列表
            r0: 初始位置
            rf: 目标位置
            title: 标题
            save_html: 保存路径
            ply_file: PLY小行星模型文件路径
            asteroid_radius: 小行星半径（Castalia约700m）
            
        Returns:
            Plotly Figure对象
        """
        fig = go.Figure()
        
        # 如果未提供目标点，尝试从结果中提取
        if rf is None and results:
            positions = results[0].get('r', results[0].get('positions', None))
            if positions is not None:
                rf = positions[-1]

        # 小行星中心位置：默认沿用旧逻辑（目标点），也可显式传入
        if asteroid_center is None:
            asteroid_center = rf if rf is not None else np.array([0, 0, 0])
        else:
            asteroid_center = np.array(asteroid_center)
        
        # 先添加小行星模型（作为背景）
        if ply_file and os.path.exists(ply_file):
            vertices, faces = load_asteroid_mesh(ply_file)
            if vertices is not None and faces is not None:
                # 计算缩放和位置 - 使用真实尺寸
                model_center = np.mean(vertices, axis=0)
                model_radius = np.max(np.linalg.norm(vertices - model_center, axis=1))
                
                # 缩放到真实大小（半径约700m）
                scale = asteroid_radius / model_radius if model_radius > 0 else 1.0
                
                # 缩放并移动到目标点附近
                vertices = (vertices - model_center) * scale + asteroid_center
                
                fig.add_trace(go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    name='小行星 (Castalia)',
                    opacity=0.6,
                    color='#8b5cf6',
                    hoverinfo='skip',
                    lighting=dict(
                        ambient=0.5,
                        diffuse=0.8,
                        fresnel=0.1,
                        specular=0.3,
                        roughness=0.7,
                    ),
                    lightposition=dict(x=500, y=500, z=500),
                    showlegend=True,
                ))
                print(f"已加载多面体模型: {os.path.basename(ply_file)}")
                print(f"  小行星中心位置: [{asteroid_center[0]:.1f}, {asteroid_center[1]:.1f}, {asteroid_center[2]:.1f}] m")
                print(f"  小行星半径: {asteroid_radius:.1f} m (真实尺寸)")
        
        # Distinct palette for multi-method comparison
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # 收集所有轨迹点用于计算边界
        all_positions = []
        
        for i, (result, label) in enumerate(zip(results, labels)):
            positions = result.get('r', result.get('positions', None))
            if positions is None:
                continue
            
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                mode='lines',
                name=label,
                line=dict(width=4, color=colors[i % len(colors)]),
            ))
            all_positions.append(positions)
            
            if r0 is None:
                r0 = positions[0]
            if rf is None:
                rf = positions[-1]
        
        # 起点终点
        if r0 is not None:
            fig.add_trace(go.Scatter3d(
                x=[r0[0]], y=[r0[1]], z=[r0[2]],
                mode='markers', name='起点',
                marker=dict(size=6, color='#ff6b6b', symbol='circle',
                           line=dict(color='white', width=1)),
            ))
        if rf is not None:
            fig.add_trace(go.Scatter3d(
                x=[rf[0]], y=[rf[1]], z=[rf[2]],
                mode='markers', name='目标点',
                marker=dict(size=8, color='#4ecdc4', symbol='diamond',
                           line=dict(color='white', width=1)),
            ))
        
        # 计算合适的相机位置 - 以小行星为中心
        # 相机距离应该能看到整个轨迹和小行星
        if all_positions:
            all_pos = np.vstack(all_positions)
            trajectory_span = np.max(np.ptp(all_pos, axis=0))
            camera_distance = max(trajectory_span * 1.5, asteroid_radius * 5)
        else:
            camera_distance = asteroid_radius * 5
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                bgcolor='#16213e' if self.dark_mode else '#ffffff',
                # 设置相机位置，让小行星居中
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.8),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1),
                ),
                # 设置宽高比为1:1:1，避免变形
                aspectmode='data',
            ),
            paper_bgcolor='#1a1a2e' if self.dark_mode else '#ffffff',
            font=dict(color='white' if self.dark_mode else 'black'),
            legend=dict(x=0.02, y=0.98),
        )
        
        if save_html:
            fig.write_html(save_html, include_plotlyjs='cdn')
            print(f"对比图已保存至: {save_html}")
        
        return fig


def plot_interactive_trajectory(
    result: Dict,
    r0: Optional[np.ndarray] = None,
    rf: Optional[np.ndarray] = None,
    save_html: Optional[str] = None,
    **kwargs
) -> Optional[go.Figure]:
    """
    快速绘制交互式3D轨迹图
    
    Parameters:
        result: 优化结果
        r0: 初始位置
        rf: 目标位置
        save_html: 保存路径
        **kwargs: 其他参数
        
    Returns:
        Plotly Figure对象（如果可用）
    """
    if not PLOTLY_AVAILABLE:
        print("警告: Plotly未安装，请运行 'pip install plotly'")
        return None
    
    viz = InteractiveTrajectoryVisualizer()
    return viz.plot_trajectory(result, r0, rf, save_html=save_html, **kwargs)


# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    
    # 生成螺旋下降轨迹
    t = np.linspace(0, 1800, 100)
    r = 500 + 200 * np.exp(-t / 600)
    theta = t / 50
    phi = np.pi / 4 + 0.2 * np.sin(t / 200)
    
    positions = np.column_stack([
        r * np.cos(theta) * np.sin(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(phi)
    ])
    
    result = {
        'r': positions,
        't': t,
        'fuel_consumption': 1.47,
        'pos_error': 0.0001,
        'vel_error': 0.0001,
    }
    
    r0 = positions[0]
    rf = positions[-1]
    
    # 创建可视化器
    viz = TrajectoryVisualizer3D(dark_mode=True)
    
    # 绘制单条轨迹
    fig1 = viz.plot_trajectory(
        result, r0, rf,
        title="小行星附着最优轨迹",
        show_thrust=True,
        show_asteroid=True,
        asteroid_radius=400
    )
    
    # 绘制多视角图
    fig2 = viz.plot_multi_view(result, r0, rf, title="多视角轨迹分析")
    
    plt.show()
    
    # 交互式可视化（如果可用）
    if PLOTLY_AVAILABLE:
        interactive_viz = InteractiveTrajectoryVisualizer(dark_mode=True)
        fig_interactive = interactive_viz.plot_trajectory(
            result, r0, rf,
            title="交互式轨迹可视化",
            save_html="interactive_trajectory.html"
        )
        fig_interactive.show()
