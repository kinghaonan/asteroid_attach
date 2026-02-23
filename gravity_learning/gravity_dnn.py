#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
引力场深度学习模型

使用神经网络近似小行星引力场。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import time


class GravityAndGradientDNN(nn.Module):
    """同时预测引力和引力梯度的双输出DNN模型"""

    def __init__(self, input_dim=3, gravity_dim=3, gradient_dim=9):
        super().__init__()
        # 共享特征提取器
        self.shared_features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # 引力预测分支
        self.gravity_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, gravity_dim), nn.Tanh()
        )

        # 引力梯度预测分支
        self.gradient_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, gradient_dim), nn.Tanh()
        )

        # 三个归一化器
        self.pos_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.grav_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.grad_scaler = MinMaxScaler(feature_range=(-1, 1))

    def forward(self, x):
        """前向传播"""
        features = self.shared_features(x)
        gravity = self.gravity_head(features)
        gradient = self.gradient_head(features)
        return gravity, gradient

    def fit_scalers(self, positions, gravity, gradient):
        """拟合归一化器"""
        print("拟合归一化器...")
        self.pos_scaler.fit(positions)
        self.grav_scaler.fit(gravity)
        self.grad_scaler.fit(gradient)

    def predict(self, positions):
        """预测引力和引力梯度"""
        self.eval()
        with torch.no_grad():
            pos_norm = self.pos_scaler.transform(positions)
            pos_tensor = torch.FloatTensor(pos_norm).to(next(self.parameters()).device)
            grav_norm, grad_norm = self.forward(pos_tensor)
            gravity = self.grav_scaler.inverse_transform(grav_norm.cpu().numpy())
            gradient = self.grad_scaler.inverse_transform(grad_norm.cpu().numpy())
        return gravity, gradient

    def save_model(self, save_path="best_gravity_gradient_dnn.pth", verbose=True):
        """保存模型

        Parameters:
            save_path: 保存路径
            verbose: 是否打印保存信息（默认True）
        """
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        model_data = {
            "model_state_dict": self.state_dict(),
            "pos_scaler": self.pos_scaler,
            "grav_scaler": self.grav_scaler,
            "grad_scaler": self.grad_scaler,
        }
        torch.save(model_data, save_path)
        if verbose:
            print(f"模型保存完成！路径：{os.path.abspath(save_path)}")

    @staticmethod
    def load_model(load_path="best_gravity_gradient_dnn.pth"):
        """加载模型"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件不存在：{load_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_data = torch.load(load_path, weights_only=False, map_location=device)

        model = GravityAndGradientDNN()
        model.load_state_dict(model_data["model_state_dict"])
        model.pos_scaler = model_data["pos_scaler"]
        model.grav_scaler = model_data["grav_scaler"]
        model.grad_scaler = model_data["grad_scaler"]
        model = model.to(device)

        print(f"模型加载完成！运行设备：{device}")
        return model


class GravityGradientTrainer:
    """引力和引力梯度训练器"""

    def __init__(self, model, gravity_weight=1.0, gradient_weight=1.0):
        """
        初始化训练器

        Parameters:
            model: GravityAndGradientDNN模型
            gravity_weight: 引力损失权重
            gradient_weight: 梯度损失权重
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.gravity_weight = gravity_weight
        self.gradient_weight = gradient_weight
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )
        self.history = {
            "train_total_loss": [],
            "train_gravity_loss": [],
            "train_gradient_loss": [],
            "val_total_loss": [],
            "val_gravity_loss": [],
            "val_gradient_loss": [],
        }

    def prepare_data(self, positions, gravity, gradient, batch_size=300):
        """准备数据加载器"""
        self.model.fit_scalers(positions, gravity, gradient)

        pos_norm = self.model.pos_scaler.transform(positions)
        grav_norm = self.model.grav_scaler.transform(gravity)
        grad_norm = self.model.grad_scaler.transform(gradient)

        pos_tensor = torch.FloatTensor(pos_norm)
        grav_tensor = torch.FloatTensor(grav_norm)
        grad_tensor = torch.FloatTensor(grad_norm)

        dataset = TensorDataset(pos_tensor, grav_tensor, grad_tensor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        print(f"数据准备完成：训练集{train_size}，验证集{val_size}")
        return train_loader, val_loader

    def train(self, train_loader, val_loader, epochs=10, print_freq=2):
        """训练模型"""
        print(f"在{self.device}上启动训练...")
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_total_loss = 0.0
            train_grav_loss = 0.0
            train_grad_loss = 0.0

            for inputs, grav_targets, grad_targets in train_loader:
                inputs = inputs.to(self.device)
                grav_targets = grav_targets.to(self.device)
                grad_targets = grad_targets.to(self.device)

                self.optimizer.zero_grad()
                grav_outputs, grad_outputs = self.model(inputs)

                loss_grav = self.criterion(grav_outputs, grav_targets)
                loss_grad = self.criterion(grad_outputs, grad_targets)
                loss = (
                    self.gravity_weight * loss_grav + self.gradient_weight * loss_grad
                )

                loss.backward()
                self.optimizer.step()

                train_total_loss += loss.item() * inputs.size(0)
                train_grav_loss += loss_grav.item() * inputs.size(0)
                train_grad_loss += loss_grad.item() * inputs.size(0)

            avg_train_total = train_total_loss / len(train_loader.dataset)
            avg_train_grav = train_grav_loss / len(train_loader.dataset)
            avg_train_grad = train_grad_loss / len(train_loader.dataset)

            self.history["train_total_loss"].append(avg_train_total)
            self.history["train_gravity_loss"].append(avg_train_grav)
            self.history["train_gradient_loss"].append(avg_train_grad)

            # 验证阶段
            self.model.eval()
            val_total_loss = 0.0
            val_grav_loss = 0.0
            val_grad_loss = 0.0

            with torch.no_grad():
                for inputs, grav_targets, grad_targets in val_loader:
                    inputs = inputs.to(self.device)
                    grav_targets = grav_targets.to(self.device)
                    grad_targets = grad_targets.to(self.device)

                    grav_outputs, grad_outputs = self.model(inputs)
                    loss_grav = self.criterion(grav_outputs, grav_targets)
                    loss_grad = self.criterion(grad_outputs, grad_targets)
                    loss = (
                        self.gravity_weight * loss_grav
                        + self.gradient_weight * loss_grad
                    )

                    val_total_loss += loss.item() * inputs.size(0)
                    val_grav_loss += loss_grav.item() * inputs.size(0)
                    val_grad_loss += loss_grad.item() * inputs.size(0)

            avg_val_total = val_total_loss / len(val_loader.dataset)
            avg_val_grav = val_grav_loss / len(val_loader.dataset)
            avg_val_grad = val_grad_loss / len(val_loader.dataset)

            self.history["val_total_loss"].append(avg_val_total)
            self.history["val_gravity_loss"].append(avg_val_grav)
            self.history["val_gradient_loss"].append(avg_val_grad)

            self.scheduler.step()

            if epoch % print_freq == 0 or epoch == epochs:
                print(
                    f"Epoch [{epoch:3d}/{epochs}]  "
                    f"Train: {avg_train_total:.4e}  "
                    f"Val: {avg_val_total:.4e}  "
                    f"(g:{avg_train_grav:.2e}, grad:{avg_train_grad:.2e})"
                )

            if avg_val_total < best_val_loss:
                best_val_loss = avg_val_total
                # 训练过程中静默保存，不打印
                self.model.save_model(
                    f"best_gravity_gradient_dnn_epoch.pth", verbose=False
                )

        print(f"\n训练完成！最优验证总损失: {best_val_loss:.6e}")
        # 最后保存一次最终模型（带打印）
        self.model.save_model(f"best_gravity_gradient_dnn_final.pth", verbose=True)
