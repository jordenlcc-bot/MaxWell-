"""
models.py
=========
两个模型：
  1. BaselinePINN  —— 标准多层全连接 + sin 激活
  2. DisplacementPINN —— 每层替换为 DisplacementFieldCell（位移门控）

位移门控的物理动机：
  Maxwell 方程中"位移电流" ∂D/∂t 只在场变化显著时才有贡献。
  用同样的思路：gate = σ(W_g · h)，正则化使 gate → 0（稀疏）；
  只在当前隐层特征需要更新时才激活，否则直通（residual pass-through）。
"""

import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────
# 辅助：sin 初始化（SIREN 风格，提高 PINN 收敛）
# ─────────────────────────────────────────────
def siren_init(layer: nn.Linear, is_first=False):
    n = layer.in_features
    w0 = 30.0 if is_first else 1.0
    limit = (1.0 / n) if is_first else math.sqrt(6.0 / n) / w0
    with torch.no_grad():
        layer.weight.uniform_(-limit, limit)
        if layer.bias is not None:
            layer.bias.zero_()


# ─────────────────────────────────────────────
# 1. 基线 PINN（Baseline MLP）
# ─────────────────────────────────────────────
class BaselinePINN(nn.Module):
    """
    输入: (x, t)  → 形状 [N, 2]
    输出: (Ez, Hy) → 形状 [N, 2]
    结构: Linear → [sin → Linear] × depth → Linear
    """
    def __init__(self, hidden_dim: int = 64, depth: int = 4):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_dim)
        siren_init(self.input_layer, is_first=True)

        self.hidden_layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.Linear(hidden_dim, hidden_dim)
            siren_init(layer)
            self.hidden_layers.append(layer)

        self.output_layer = nn.Linear(hidden_dim, 2)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        u = torch.sin(self.input_layer(xt))
        for layer in self.hidden_layers:
            u = torch.sin(layer(u))
        return self.output_layer(u)


# ─────────────────────────────────────────────
# 2. 位移门控单元（DisplacementFieldCell）
# ─────────────────────────────────────────────
class DisplacementFieldCell(nn.Module):
    """
    把一层 sin-MLP 替换为"位移门控场单元"：

        h     = sin(W_h · u + b_h)       # field update candidate
        g     = σ(W_g · u + b_g)         # displacement gate  ∈ (0,1)
        output = g ⊙ h + (1 − g) ⊙ u    # gated residual（类 GRU）

    物理意义：
        g ≈ 0  →  场变化不显著，直通上一层特征（节省计算）
        g ≈ 1  →  场变化显著（类似位移电流激活），执行完整更新

    可训练的稀疏正则化：在 loss 里加 λ·mean(g) 使 gate 趋向稀疏。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.field_linear = nn.Linear(dim, dim)
        self.gate_linear  = nn.Linear(dim, dim)
        siren_init(self.field_linear)
        # gate 初始化偏向较小值 → 初始时 gate 较稀疏
        nn.init.xavier_uniform_(self.gate_linear.weight, gain=0.5)
        nn.init.constant_(self.gate_linear.bias, -1.0)  # σ(-1) ≈ 0.27

    def forward(self, u: torch.Tensor):
        h = torch.sin(self.field_linear(u))       # candidate update
        g = torch.sigmoid(self.gate_linear(u))    # displacement gate
        return g * h + (1.0 - g) * u             # gated residual

    def gate_activation_rate(self, u: torch.Tensor) -> float:
        """返回当前batch的平均门控激活率（越低 = 越稀疏）"""
        with torch.no_grad():
            g = torch.sigmoid(self.gate_linear(u))
        return g.mean().item()


# ─────────────────────────────────────────────
# 3. Maxwell 场 PINN（Displacement-Gated）
# ─────────────────────────────────────────────
class DisplacementPINN(nn.Module):
    """
    结构：
        Linear(2 → hidden) with sin
        → DisplacementFieldCell × depth
        → Linear(hidden → 2)
    """
    def __init__(self, hidden_dim: int = 64, depth: int = 4):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_dim)
        siren_init(self.input_layer, is_first=True)

        self.cells = nn.ModuleList(
            [DisplacementFieldCell(hidden_dim) for _ in range(depth)]
        )

        self.output_layer = nn.Linear(hidden_dim, 2)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        u = torch.sin(self.input_layer(xt))
        for cell in self.cells:
            u = cell(u)
        return self.output_layer(u)

    def mean_gate_rate(self, xt: torch.Tensor) -> float:
        """计算所有层的平均门控激活率"""
        with torch.no_grad():
            u = torch.sin(self.input_layer(xt))
            rates = []
            for cell in self.cells:
                rates.append(cell.gate_activation_rate(u))
                u = cell(u)
        return sum(rates) / len(rates)
