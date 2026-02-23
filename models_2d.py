"""
models_2d.py  [C]
=================
2D Maxwell PINN 模型（输入维度从 2 扩展到 3：x, y, t）。
输出维度从 2 扩展到 3：Ez, Hx, Hy。

与 models.py 结构一致，只改 I/O 维度，方便对比。
"""

import torch
import torch.nn as nn
import math


def siren_init(layer: nn.Linear, is_first=False):
    n = layer.in_features
    w0 = 30.0 if is_first else 1.0
    limit = (1.0 / n) if is_first else math.sqrt(6.0 / n) / w0
    with torch.no_grad():
        layer.weight.uniform_(-limit, limit)
        if layer.bias is not None:
            layer.bias.zero_()


class BaselinePINN2D(nn.Module):
    """
    输入: (x, y, t) → [N, 3]
    输出: (Ez, Hx, Hy) → [N, 3]
    """
    def __init__(self, hidden_dim: int = 64, depth: int = 4):
        super().__init__()
        self.input_layer = nn.Linear(3, hidden_dim)
        siren_init(self.input_layer, is_first=True)
        self.hidden_layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.Linear(hidden_dim, hidden_dim)
            siren_init(layer)
            self.hidden_layers.append(layer)
        self.output_layer = nn.Linear(hidden_dim, 3)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        u = torch.sin(self.input_layer(xyt))
        for layer in self.hidden_layers:
            u = torch.sin(layer(u))
        return self.output_layer(u)


class DisplacementFieldCell2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.field_linear = nn.Linear(dim, dim)
        self.gate_linear  = nn.Linear(dim, dim)
        siren_init(self.field_linear)
        nn.init.xavier_uniform_(self.gate_linear.weight, gain=0.5)
        nn.init.constant_(self.gate_linear.bias, -1.0)

    def forward(self, u: torch.Tensor):
        h = torch.sin(self.field_linear(u))
        g = torch.sigmoid(self.gate_linear(u))
        return g * h + (1.0 - g) * u

    def gate_activation_rate(self, u: torch.Tensor) -> float:
        with torch.no_grad():
            g = torch.sigmoid(self.gate_linear(u))
        return g.mean().item()


class DisplacementPINN2D(nn.Module):
    """
    输入: (x, y, t) → [N, 3]
    输出: (Ez, Hx, Hy) → [N, 3]
    """
    def __init__(self, hidden_dim: int = 64, depth: int = 4):
        super().__init__()
        self.input_layer = nn.Linear(3, hidden_dim)
        siren_init(self.input_layer, is_first=True)
        self.cells = nn.ModuleList(
            [DisplacementFieldCell2D(hidden_dim) for _ in range(depth)]
        )
        self.output_layer = nn.Linear(hidden_dim, 3)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        u = torch.sin(self.input_layer(xyt))
        for cell in self.cells:
            u = cell(u)
        return self.output_layer(u)

    def mean_gate_rate(self, xyt: torch.Tensor) -> float:
        with torch.no_grad():
            u = torch.sin(self.input_layer(xyt))
            rates = []
            for cell in self.cells:
                rates.append(cell.gate_activation_rate(u))
                u = cell(u)
        return sum(rates) / len(rates)
