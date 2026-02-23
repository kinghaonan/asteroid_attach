# 分阶段验证脚本说明

本目录包含三个分阶段验证脚本，用于验证项目的三个主要模块。

## 脚本列表

### 1. verify_phase1.py - 第一阶段验证（引力场学习）

**功能**：
- 加载PLY小行星模型
- 生成引力样本数据
- 训练DNN引力场模型
- 验证模型精度

**输入**：
- `Castalia Radar-based.ply` - 小行星模型文件

**输出**：
- `data/models/gravity_dnn_model.pth` - 训练好的DNN模型
- `data/samples/gravity_samples.npz` - 样本数据
- `results/phase1/verification_results.png` - 验证图表

**运行**：
```bash
python verify/verify_phase1.py
```

---

### 2. verify_phase2.py - 第二阶段验证（轨迹优化）

**功能**：
- 加载第一阶段训练的引力场模型
- 使用多种算法（伪谱法、打靶法、同伦法）进行轨迹优化
- 选择最优轨迹
- 保存轨迹供第三阶段使用

**输入**：
- `data/models/gravity_dnn_model.pth` - 第一阶段输出的模型

**输出**：
- `results/phase2/optimal_trajectory.pkl` - 最优轨迹数据
- `results/phase2/trajectory.png` - 轨迹图表

**数据衔接**：
- 从第一阶段读取：`data/models/gravity_dnn_model.pth`
- 输出到第三阶段：`results/phase2/optimal_trajectory.pkl`

**运行**：
```bash
python verify/verify_phase2.py
```

---

### 3. verify_phase3.py - 第三阶段验证（控制与仿真）

**功能**：
- 加载第二阶段优化的轨迹
- 使用PID控制器进行轨迹跟踪
- 进行蒙特卡洛鲁棒性分析
- 生成完整验证报告

**输入**：
- `results/phase2/optimal_trajectory.pkl` - 第二阶段输出的轨迹
- `data/models/gravity_dnn_model.pth` - 引力模型

**输出**：
- `results/phase3/control_simulation.pkl` - 仿真结果
- `results/phase3/final_report.txt` - 最终报告
- `results/phase3/simulation_results.png` - 仿真图表

**数据衔接**：
- 从第二阶段读取：`results/phase2/optimal_trajectory.pkl`

**运行**：
```bash
python verify/verify_phase3.py
```

---

### 4. run_all_verification.py - 统一运行脚本

**功能**：
- 依次运行所有三个阶段的验证
- 自动处理阶段间的依赖关系
- 生成汇总报告

**运行**：
```bash
python verify/run_all_verification.py
```

---

## 数据流衔接

```
┌─────────────────────────────────────────────────────────────┐
│                        数据流示意图                          │
└─────────────────────────────────────────────────────────────┘

Phase 1 (引力场学习)
    │
    ├── 输入: Castalia Radar-based.ply
    │
    └── 输出: data/models/gravity_dnn_model.pth
                  │
                  ▼
Phase 2 (轨迹优化)
    │
    ├── 输入: data/models/gravity_dnn_model.pth
    │
    └── 输出: results/phase2/optimal_trajectory.pkl
                  │
                  ▼
Phase 3 (控制与仿真)
    │
    ├── 输入: results/phase2/optimal_trajectory.pkl
    │         data/models/gravity_dnn_model.pth
    │
    └── 输出: results/phase3/final_report.txt
```

---

## 使用流程

### 方式一：分别运行（推荐开发和调试时使用）

```bash
# 第一阶段
python verify/verify_phase1.py

# 第二阶段（依赖第一阶段）
python verify/verify_phase2.py

# 第三阶段（依赖第二阶段）
python verify/verify_phase3.py
```

### 方式二：统一运行（推荐完整验证时使用）

```bash
# 一键运行所有阶段
python verify/run_all_verification.py
```

---

## 成功标准

### 第一阶段
- ✅ PLY模型正常加载
- ✅ 引力样本生成成功
- ✅ DNN模型训练收敛
- ✅ 预测误差 < 10%

### 第二阶段
- ✅ 伪谱法/打靶法至少一种成功
- ✅ 燃料消耗合理 (< 100 kg)
- ✅ 终端误差满足要求

### 第三阶段
- ✅ PID跟踪误差 < 50m
- ✅ 蒙特卡洛成功率 > 70%
- ✅ 生成完整报告

---

## 常见问题

### Q1: 第一阶段提示找不到PLY文件
**解决**：确保 `Castalia Radar-based.ply` 在项目根目录

### Q2: 第二阶段提示找不到模型文件
**解决**：先成功运行第一阶段，生成 `data/models/gravity_dnn_model.pth`

### Q3: 第三阶段提示找不到轨迹文件
**解决**：先成功运行第二阶段，生成 `results/phase2/optimal_trajectory.pkl`

### Q4: 训练时间过长
**解决**：在脚本中减少样本数量和epoch数量（已设置为验证友好的较小值）

---

## 输出文件结构

运行所有验证后，项目结构如下：

```
asteroid_contral/
├── data/
│   ├── models/
│   │   └── gravity_dnn_model.pth      # 训练好的模型
│   └── samples/
│       └── gravity_samples.npz         # 样本数据
│
├── results/
│   ├── phase1/
│   │   └── verification_results.png    # 引力场验证图
│   │
│   ├── phase2/
│   │   ├── optimal_trajectory.pkl      # 最优轨迹
│   │   └── trajectory.png              # 轨迹图
│   │
│   └── phase3/
│       ├── control_simulation.pkl      # 仿真结果
│       ├── final_report.txt            # 最终报告
│       └── simulation_results.png      # 仿真结果图
│
└── verify/
    ├── verify_phase1.py
    ├── verify_phase2.py
    ├── verify_phase3.py
    ├── run_all_verification.py
    └── README.md                       # 本文件
```

---

## 作者

Asteroid Mission Team
