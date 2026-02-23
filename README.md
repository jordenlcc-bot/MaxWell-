# Maxwell PINN MVP — PyTorch 最小验证实验

## 目标

用 1D Maxwell 方程验证"位移门控"架构的有效性：

| 对比项 | Baseline MLP PINN | Displacement-Gated PINN |
| ------ | ----------------- | ----------------------- |
| 网络结构 | Linear + sin × depth | DisplacementFieldCell × depth |
| 门控 | 无 | σ(W_g·h) 稀疏门控 |
| 精度 | L2 vs 解析解 | L2 vs 解析解 |
| 稀疏性 | — | 门控激活率 < 50% |

## 方程

```text
∂Ez/∂t = −∂Hy/∂x    (Faraday)
∂Hy/∂t = −∂Ez/∂x    (Ampere + 位移电流, c=1)
```

**解析解**（用于验证）：

```text
Ez(x,t) = sin(πx) · cos(πt)
Hy(x,t) = −cos(πx) · sin(πt)
```

## 文件结构

```text
maxwell-pinn-mvp/
├── models.py           # BaselinePINN + DisplacementPINN
├── pde.py              # Maxwell残差 + 采样器 + 解析解
├── train.py            # 训练循环（支持两种模型）
├── run_experiment.py   # 一键运行 + 绘图
├── requirements.txt    # torch, matplotlib, numpy
└── results/            # 输出图表（自动创建）
```

## 快速开始

```powershell
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行实验（CPU 约 10-20 分钟，GPU 约 2-3 分钟）
cd C:\Users\volum\.gemini\antigravity\scratch\maxwell-pinn-mvp
python run_experiment.py
```

## 输出

```text
results/
├── comparison_plots.png   # (a) Loss  (b) L2误差  (c) Gate稀疏率
├── field_comparison.png   # t=0, 0.5, 1.0 时刻的 Ez 场分布
├── baseline_history.pt    # Baseline 训练记录
└── displacement_history.pt  # DispField 训练记录
```

## 验证指标

| 指标 | 意义 | 目标 |
| ---- | ---- | ---- |
| L2 相对误差 | 与解析解的距离 | < 1e-2（两个模型相近） |
| 门控激活率 | 平均激活的 gate 比例 | < 0.5（稀疏） |
| Loss 收敛 | 训练曲线下降趋势 | 两者都应稳定下降 |

## 下一步（实验完成后）

1. **路线 A**：把 `DisplacementFieldCell` 移植到 MindSpore Elec 模板
2. **路线 B**：用实验结果图表支撑论文 Method 章节
3. **路线 C**：扩展到 2D Maxwell（增加 y 坐标维度）
