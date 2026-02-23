"""
pde.py
======
1D Maxwell PDE 残差（用 torch.autograd 自动微分）

方程：
    ∂Ez/∂t = −∂Hy/∂x    (Faraday)
    ∂Hy/∂t = −∂Ez/∂x    (Ampere + 位移电流，c=1)

域：  x ∈ [0, 1],  t ∈ [0, T]

解析解（用于验证，初始条件 Ez(x,0) = sin(πx)，Hy(x,0) = 0）：
    Ez(x, t) = sin(πx) · cos(πt)
    Hy(x, t) = −sin(πx) · sin(πt)  （c = 1）
"""

import torch
from torch import Tensor


def _grad(y: Tensor, x: Tensor) -> Tensor:
    """∂y/∂x，保持图可微分"""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def maxwell_residual(model, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """
    计算Maxwell PDE残差。
    x, t 必须已设置 requires_grad=True。

    返回：
        r_faraday  = ∂Ez/∂t + ∂Hy/∂x   (应 = 0)
        r_ampere   = ∂Hy/∂t + ∂Ez/∂x   (应 = 0)
    """
    xt = torch.cat([x, t], dim=-1)
    out = model(xt)
    Ez, Hy = out[:, 0:1], out[:, 1:2]

    dEz_dt = _grad(Ez, t)
    dHy_dt = _grad(Hy, t)
    dEz_dx = _grad(Ez, x)
    dHy_dx = _grad(Hy, x)

    r_faraday = dEz_dt + dHy_dx   # ∂Ez/∂t = −∂Hy/∂x
    r_ampere  = dHy_dt + dEz_dx   # ∂Hy/∂t = −∂Ez/∂x
    return r_faraday, r_ampere


def exact_solution(x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """解析解"""
    Ez = torch.sin(torch.pi * x) * torch.cos(torch.pi * t)
    Hy = -torch.cos(torch.pi * x) * torch.sin(torch.pi * t)
    return Ez, Hy


def sample_collocation(n_pde: int, T: float, device: str) -> tuple[Tensor, Tensor]:
    """在 [0,1]×[0,T] 内均匀随机采样 PDE 配置点"""
    x = torch.rand(n_pde, 1, device=device)
    t = torch.rand(n_pde, 1, device=device) * T
    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t


def sample_ic(n_ic: int, device: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    初始条件采样（t = 0）：
        Ez(x, 0) = sin(πx)
        Hy(x, 0) = 0
    """
    x = torch.rand(n_ic, 1, device=device)
    t = torch.zeros(n_ic, 1, device=device)
    Ez0 = torch.sin(torch.pi * x)
    Hy0 = torch.zeros(n_ic, 1, device=device)
    return x, t, Ez0, Hy0


def sample_bc(n_bc: int, T: float, device: str) -> tuple[Tensor, Tensor]:
    """
    边界条件采样（x = 0 和 x = 1）：
        Ez(0, t) = 0,  Ez(1, t) = 0
    """
    t_vals = torch.rand(n_bc, 1, device=device) * T
    x_left  = torch.zeros(n_bc, 1, device=device)
    x_right = torch.ones(n_bc, 1, device=device)
    return torch.cat([x_left, x_right], dim=0), torch.cat([t_vals, t_vals], dim=0)
