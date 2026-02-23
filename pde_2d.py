"""
pde_2d.py  [C]
==============
2D 时域 Maxwell 方程（TM 模式）的 PDE 残差 + 采样器 + 解析解。

方程（TM 模式，Ez Hx Hy，c=1）：
    ∂Ez/∂t =  ∂Hy/∂x − ∂Hx/∂y     (Faraday z)
    ∂Hx/∂t = −∂Ez/∂y               (Ampere x)
    ∂Hy/∂t =  ∂Ez/∂x               (Ampere y)

域：  x ∈ [0,1], y ∈ [0,1], t ∈ [0,T]

解析解（最低阶谐振腔模式，m=n=1）：
    ω = π√2  (c=1)
    Ez(x,y,t) =  sin(πx) sin(πy) cos(ωt)
    Hx(x,y,t) = −sin(πx) cos(πy) sin(ωt) / √2
    Hy(x,y,t) =  cos(πx) sin(πy) sin(ωt) / √2

边界条件：
    Ez = 0  在 x=0, x=1, y=0, y=1  （PEC 边界）
"""

import torch
from torch import Tensor
import math

OMEGA = math.pi * math.sqrt(2.0)   # 谐振频率 ω = π√2


# ─────────────────────────────────────────────
# 辅助：自动微分梯度
# ─────────────────────────────────────────────
def _grad(y: Tensor, x: Tensor) -> Tensor:
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


# ─────────────────────────────────────────────
# PDE 残差
# ─────────────────────────────────────────────
def maxwell2d_residual(
    model,
    x: Tensor, y: Tensor, t: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    x, y, t 必须 requires_grad=True。
    返回：
        r_faraday  = ∂Ez/∂t − (∂Hy/∂x − ∂Hx/∂y)   (应 = 0)
        r_ampere_x = ∂Hx/∂t + ∂Ez/∂y               (应 = 0)
        r_ampere_y = ∂Hy/∂t − ∂Ez/∂x               (应 = 0)
    """
    xyt = torch.cat([x, y, t], dim=-1)
    out = model(xyt)
    Ez, Hx, Hy = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    dEz_dt = _grad(Ez, t)
    dHx_dt = _grad(Hx, t)
    dHy_dt = _grad(Hy, t)
    dEz_dx = _grad(Ez, x)
    dEz_dy = _grad(Ez, y)
    dHx_dy = _grad(Hx, y)
    dHy_dx = _grad(Hy, x)

    r1 = dEz_dt - (dHy_dx - dHx_dy)   # ∂Ez/∂t = ∂Hy/∂x − ∂Hx/∂y
    r2 = dHx_dt + dEz_dy               # ∂Hx/∂t = −∂Ez/∂y
    r3 = dHy_dt - dEz_dx               # ∂Hy/∂t = ∂Ez/∂x
    return r1, r2, r3


# ─────────────────────────────────────────────
# 解析解
# ─────────────────────────────────────────────
def exact2d(x: Tensor, y: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    pi = torch.pi
    s2 = math.sqrt(2.0)
    Ez = torch.sin(pi * x) * torch.sin(pi * y) * torch.cos(OMEGA * t)
    Hx = -torch.sin(pi * x) * torch.cos(pi * y) * torch.sin(OMEGA * t) / s2
    Hy =  torch.cos(pi * x) * torch.sin(pi * y) * torch.sin(OMEGA * t) / s2
    return Ez, Hx, Hy


# ─────────────────────────────────────────────
# 采样器
# ─────────────────────────────────────────────
def sample2d_collocation(n: int, T: float, device: str):
    """在 [0,1]² × [0,T] 内随机采样"""
    x = torch.rand(n, 1, device=device)
    y = torch.rand(n, 1, device=device)
    t = torch.rand(n, 1, device=device) * T
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    return x, y, t


def sample2d_ic(n: int, device: str):
    """初始条件：t=0"""
    x = torch.rand(n, 1, device=device)
    y = torch.rand(n, 1, device=device)
    t = torch.zeros(n, 1, device=device)
    Ez0, Hx0, Hy0 = exact2d(x, y, t)
    return x, y, t, Ez0.detach(), Hx0.detach(), Hy0.detach()


def sample2d_bc(n: int, T: float, device: str):
    """
    PEC 边界条件：Ez = 0 在四条边上
    返回各边采样点拼合后的 (x, y, t)
    """
    t = torch.rand(n, 1, device=device) * T
    pts = []
    for xi, yi in [
        (torch.zeros(n, 1, device=device), torch.rand(n, 1, device=device)),  # x=0
        (torch.ones(n, 1, device=device),  torch.rand(n, 1, device=device)),  # x=1
        (torch.rand(n, 1, device=device),  torch.zeros(n, 1, device=device)), # y=0
        (torch.rand(n, 1, device=device),  torch.ones(n, 1, device=device)),  # y=1
    ]:
        pts.append((xi, yi, torch.rand(n, 1, device=device) * T))

    x_bc = torch.cat([p[0] for p in pts])
    y_bc = torch.cat([p[1] for p in pts])
    t_bc = torch.cat([p[2] for p in pts])
    return x_bc, y_bc, t_bc
