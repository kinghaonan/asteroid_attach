#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLY小行星模型和多面体引力计算

提供小行星多面体模型的加载和引力场计算功能。
"""

import numpy as np
from plyfile import PlyData
import os


class PLYAsteroidModel:
    """从PLY文件加载小行星多面体模型"""

    def __init__(self, ply_file_path):
        """
        初始化PLY模型

        Parameters:
            ply_file_path: PLY文件路径
        """
        self.ply_file_path = ply_file_path
        self.vertices = None
        self.faces = None
        self._load_ply_model()
        print(f"成功加载PLY模型：{os.path.basename(ply_file_path)}")
        print(f"模型包含 {len(self.vertices)} 个顶点，{len(self.faces)} 个面元")

    def _load_ply_model(self):
        """解析PLY文件"""
        if not os.path.exists(self.ply_file_path):
            raise FileNotFoundError(f"PLY文件不存在：{self.ply_file_path}")

        ply_data = PlyData.read(self.ply_file_path)

        # 提取顶点坐标
        vertex_element = ply_data["vertex"]
        self.vertices = np.column_stack(
            [vertex_element["x"], vertex_element["y"], vertex_element["z"]]
        )

        # 提取面元
        face_element = ply_data["face"]
        self.faces = []
        for face in face_element:
            face_data = face[0]
            if isinstance(face_data, tuple) and len(face_data) == 2:
                num_vertices, vertex_indices = face_data
            else:
                vertex_indices = face_data
                num_vertices = len(vertex_indices)

            if num_vertices != 3:
                raise ValueError(f"非三角形面元：{num_vertices}个顶点")
            self.faces.append(vertex_indices)

        self.faces = np.array(self.faces, dtype=int)

    def scale_to_real_size(self, target_diameter_m=1400.0):
        """缩放模型至真实大小"""
        center = np.mean(self.vertices, axis=0)
        vertex_distances = np.linalg.norm(self.vertices - center, axis=1)
        current_diameter = 2 * np.max(vertex_distances)
        scale_factor = target_diameter_m / current_diameter
        self.vertices = (self.vertices - center) * scale_factor + center
        print(f"模型已缩放至真实大小（直径{target_diameter_m}m）")

    def get_asteroid_radius(self):
        """计算小行星半径"""
        center = np.mean(self.vertices, axis=0)
        vertex_distances = np.linalg.norm(self.vertices - center, axis=1)
        radius = np.max(vertex_distances)
        return round(radius, 4)


class PolyhedralGravitySampler:
    """基于多面体模型的引力样本生成器"""

    def __init__(self, asteroid_model, asteroid_density=2670):
        """
        初始化采样器

        Parameters:
            asteroid_model: PLYAsteroidModel实例
            asteroid_density: 小行星密度（kg/m³）
        """
        self.asteroid_model = asteroid_model
        self.vertices = asteroid_model.vertices
        self.faces = asteroid_model.faces
        self.G_const = 6.67430e-11
        self.asteroid_density = asteroid_density
        self.asteroid_radius = asteroid_model.get_asteroid_radius()
        self.asteroid_center = np.mean(self.vertices, axis=0)

    def generate_sampling_points(
        self, num_samples=50000, min_r_ratio=1.1, max_r_ratio=5.0
    ):
        """
        生成采样点

        Parameters:
            num_samples: 采样点数量
            min_r_ratio: 最小半径倍数
            max_r_ratio: 最大半径倍数

        Returns:
            采样点坐标 (N, 3)
        """
        min_r = min_r_ratio * self.asteroid_radius
        max_r = max_r_ratio * self.asteroid_radius

        phi = np.random.uniform(0, 2 * np.pi, num_samples)
        theta = np.arccos(2 * np.random.uniform(0, 1, num_samples) - 1)
        r = np.random.uniform(min_r, max_r, num_samples)

        x = self.asteroid_center[0] + r * np.sin(theta) * np.cos(phi)
        y = self.asteroid_center[1] + r * np.sin(theta) * np.sin(phi)
        z = self.asteroid_center[2] + r * np.cos(theta)

        return np.column_stack((x, y, z))

    def calculate_polyhedral_gravity(self, points):
        """
        多面体法计算引力加速度

        Parameters:
            points: 采样点坐标 (N, 3)

        Returns:
            引力加速度 (N, 3)
        """
        gravity_list = []
        batch_size = 100

        for batch_idx in range(0, len(points), batch_size):
            batch_points = points[batch_idx : batch_idx + batch_size]
            batch_gravity = []

            for point in batch_points:
                if (
                    np.linalg.norm(point - self.asteroid_center)
                    < self.asteroid_radius * 0.95
                ):
                    batch_gravity.append(np.zeros(3))
                    continue

                total_gravity = np.zeros(3)
                for face_idx in self.faces:
                    face_vertices = self.vertices[face_idx]
                    v1, v2, v3 = face_vertices[0], face_vertices[1], face_vertices[2]

                    face_normal = np.cross(v2 - v1, v3 - v1)
                    normal_magnitude = np.linalg.norm(face_normal)
                    if normal_magnitude < 1e-12:
                        continue
                    face_normal = face_normal / normal_magnitude

                    face_center = np.mean(face_vertices, axis=0)
                    if np.dot(face_center - self.asteroid_center, face_normal) < 0:
                        face_normal = -face_normal

                    r1 = v1 - point
                    r2 = v2 - point
                    r3 = v3 - point
                    r1_mag = np.linalg.norm(r1)
                    r2_mag = np.linalg.norm(r2)
                    r3_mag = np.linalg.norm(r3)

                    cross_r1r2 = np.cross(r1, r2)
                    dot_cross_r3 = np.dot(cross_r1r2, r3)
                    dot_r1_cross_r2r3 = np.dot(r1, np.cross(r2, r3))
                    denominator = dot_r1_cross_r2r3 + r1_mag * r2_mag * r3_mag

                    if abs(denominator) < 1e-12:
                        solid_angle = 0.0
                    else:
                        solid_angle = np.arctan2(dot_cross_r3, denominator)

                    solid_angle = abs(solid_angle)
                    face_area = 0.5 * normal_magnitude
                    face_contribution = (
                        -self.G_const
                        * self.asteroid_density
                        * solid_angle
                        * face_normal
                        * face_area
                    )
                    total_gravity += face_contribution

                batch_gravity.append(total_gravity)
            gravity_list.extend(batch_gravity)

            # 打印进度（每500个点或最后一批）
            if (batch_idx + batch_size) % 500 == 0 or batch_idx + batch_size >= len(
                points
            ):
                progress = min(batch_idx + batch_size, len(points))
                print(
                    f"   引力计算: {progress}/{len(points)} ({progress / len(points) * 100:.1f}%)"
                )

        return np.array(gravity_list)

    def calculate_polyhedral_gravity_gradient(self, points):
        """
        多面体法计算引力梯度（3x3矩阵，展平为9维向量）

        引力梯度是引力加速度的雅可比矩阵：∂g_i/∂r_j
        对于多面体模型，可以通过对引力公式求导得到

        Parameters:
            points: 采样点坐标 (N, 3)

        Returns:
            引力梯度 (N, 9)，每行是一个3x3矩阵的展平
        """
        gradient_list = []
        batch_size = 50  # 梯度计算更复杂，减小批量

        for batch_idx in range(0, len(points), batch_size):
            batch_points = points[batch_idx : batch_idx + batch_size]
            batch_gradient = []

            for point in batch_points:
                # 跳过小行星内部的点
                if (
                    np.linalg.norm(point - self.asteroid_center)
                    < self.asteroid_radius * 0.95
                ):
                    batch_gradient.append(np.zeros(9))
                    continue

                # 初始化3x3梯度矩阵
                gradient_matrix = np.zeros((3, 3))

                for face_idx in self.faces:
                    face_vertices = self.vertices[face_idx]
                    v1, v2, v3 = face_vertices[0], face_vertices[1], face_vertices[2]

                    # 计算面元外法向量
                    face_normal = np.cross(v2 - v1, v3 - v1)
                    normal_magnitude = np.linalg.norm(face_normal)
                    if normal_magnitude < 1e-12:
                        continue
                    face_normal = face_normal / normal_magnitude

                    # 确保法向量指向外部
                    face_center = np.mean(face_vertices, axis=0)
                    if np.dot(face_center - self.asteroid_center, face_normal) < 0:
                        face_normal = -face_normal

                    # 计算立体角（用于权重）
                    r1 = v1 - point
                    r2 = v2 - point
                    r3 = v3 - point
                    r1_mag = np.linalg.norm(r1)
                    r2_mag = np.linalg.norm(r2)
                    r3_mag = np.linalg.norm(r3)

                    cross_r1r2 = np.cross(r1, r2)
                    dot_cross_r3 = np.dot(cross_r1r2, r3)
                    dot_r1_cross_r2r3 = np.dot(r1, np.cross(r2, r3))
                    denominator = dot_r1_cross_r2r3 + r1_mag * r2_mag * r3_mag

                    if abs(denominator) < 1e-12:
                        solid_angle = 0.0
                    else:
                        solid_angle = np.arctan2(dot_cross_r3, denominator)

                    solid_angle = abs(solid_angle)
                    face_area = 0.5 * normal_magnitude

                    # 计算引力梯度贡献
                    # 基于多面体引力理论，梯度与距离立方成反比
                    for i in range(3):
                        for j in range(3):
                            # 引力梯度分量计算
                            term1 = (
                                np.dot(face_normal, r1)
                                * r1[i]
                                * r1[j]
                                / (r1_mag**3 + 1e-12)
                            )
                            term2 = (
                                np.dot(face_normal, r2)
                                * r2[i]
                                * r2[j]
                                / (r2_mag**3 + 1e-12)
                            )
                            term3 = (
                                np.dot(face_normal, r3)
                                * r3[i]
                                * r3[j]
                                / (r3_mag**3 + 1e-12)
                            )

                            gradient_contribution = (
                                self.G_const
                                * self.asteroid_density
                                * face_area
                                * (term1 + term2 + term3)
                            )
                            gradient_matrix[i, j] += gradient_contribution

                # 将3x3矩阵展平为9维向量（按行优先）
                batch_gradient.append(gradient_matrix.flatten())

            gradient_list.extend(batch_gradient)

            # 打印进度（每200个点或最后一批）
            if (batch_idx + batch_size) % 200 == 0 or batch_idx + batch_size >= len(
                points
            ):
                progress = min(batch_idx + batch_size, len(points))
                print(
                    f"   梯度计算: {progress}/{len(points)} ({progress / len(points) * 100:.1f}%)"
                )

        return np.array(gradient_list)
