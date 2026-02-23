"""
run_experiment_2d.py  [C]
=========================
2D Maxwell PINN 实验：对标 MindSpore Elec 的案例规模。

修改说明（相比 1D）：
  - 输入扩展：(x,t) → (x,y,t)
  - 输出扩展：(Ez,Hy) → (Ez,Hx,Hy)
  - PDE 残差增加：r_faraday, r_ampere_x, r_ampere_y
  - 采样点增加（2D 域面积是 1D 的 N 倍）

用法：
    python run_experiment_2d.py

输出：
    results/2d_comparison_plots.png
    results/2d_field_slice.png
    results/2d_baseline_history.pt
    results/2d_displacement_history.pt
"""

import os, time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models_2d import BaselinePINN2D, DisplacementPINN2D
from pde_2d import (
    maxwell2d_residual, exact2d,
    sample2d_collocation, sample2d_ic, sample2d_bc,
    OMEGA
)

os.makedirs("results", exist_ok=True)

# ────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_DIM = 64
DEPTH      = 4
EPOCHS     = 5000
LR         = 1e-3
N_PDE      = 3000      # 2D 需要更多配置点
N_IC       = 800
N_BC_SIDE  = 300       # 每条边 300 点，共 4 边 = 1200 点
T          = 0.5       # 2D 谐振腔跑半个周期
W_IC       = 10.0
W_BC       = 10.0
W_GATE     = 0.01

print(f"\n🖥  Device: {DEVICE}  (2D Maxwell)")
print(f"📐 Inputs: (x,y,t) → Outputs: (Ez,Hx,Hy)")
print(f"📐 Network: hidden={HIDDEN_DIM}, depth={DEPTH}, epochs={EPOCHS}, T={T}\n")

mse = nn.MSELoss()


# ─────────────────────────────────────────────
# L2 误差计算（在均匀网格上）
# ─────────────────────────────────────────────
def compute_l2_2d(model, T_eval: float = None, ng: int = 30):
    if T_eval is None:
        T_eval = T
    model.eval()
    with torch.no_grad():
        xs_1d = torch.linspace(0, 1, ng, device=DEVICE)
        ys_1d = torch.linspace(0, 1, ng, device=DEVICE)
        xx, yy = torch.meshgrid(xs_1d, ys_1d, indexing="ij")
        x_flat = xx.reshape(-1, 1)
        y_flat = yy.reshape(-1, 1)
        t_flat = torch.full_like(x_flat, T_eval)
        xyt = torch.cat([x_flat, y_flat, t_flat], dim=-1)
        pred = model(xyt)
        Ez_p, Hx_p, Hy_p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        Ez_t, Hx_t, Hy_t = exact2d(x_flat, y_flat, t_flat)
        err = (torch.norm(Ez_p - Ez_t) + torch.norm(Hx_p - Hx_t) + torch.norm(Hy_p - Hy_t))
        ref = (torch.norm(Ez_t) + torch.norm(Hx_t) + torch.norm(Hy_t)) + 1e-10
    model.train()
    return (err / ref).item()


# ─────────────────────────────────────────────
# 通用训练循环（2D）
# ─────────────────────────────────────────────
def train_2d(model, label: str, w_gate: float = 0.0):
    is_disp = isinstance(model, DisplacementPINN2D)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    hist = {"epochs_log": [], "loss_log": [], "l2_log": [],
            "gate_rate_log": [], "wall_time_log": []}
    t0 = time.time()
    print(f"\n{'='*58}")
    print(f"  {label}  ({'2D Displacement' if is_disp else '2D Baseline'})")
    print(f"{'='*58}")

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()

        # PDE
        x_c, y_c, t_c = sample2d_collocation(N_PDE, T, DEVICE)
        r1, r2, r3 = maxwell2d_residual(model, x_c, y_c, t_c)
        loss_pde = mse(r1, torch.zeros_like(r1)) \
                 + mse(r2, torch.zeros_like(r2)) \
                 + mse(r3, torch.zeros_like(r3))

        # IC
        x_ic, y_ic, t_ic, Ez0, Hx0, Hy0 = sample2d_ic(N_IC, DEVICE)
        xyt_ic = torch.cat([x_ic, y_ic, t_ic], dim=-1)
        p_ic = model(xyt_ic)
        loss_ic = mse(p_ic[:, 0:1], Ez0) + mse(p_ic[:, 1:2], Hx0) + mse(p_ic[:, 2:3], Hy0)

        # BC (Ez = 0 on 4 walls)
        x_bc, y_bc, t_bc = sample2d_bc(N_BC_SIDE, T, DEVICE)
        xyt_bc = torch.cat([x_bc, y_bc, t_bc], dim=-1)
        p_bc = model(xyt_bc)
        loss_bc = mse(p_bc[:, 0:1], torch.zeros_like(p_bc[:, 0:1]))

        # Gate 稀疏正则
        loss_gate = torch.tensor(0.0, device=DEVICE)
        gate_rate = 0.0
        if is_disp:
            xyt_g = torch.cat([x_c.detach(), y_c.detach(), t_c.detach()], dim=-1)
            u = torch.sin(model.input_layer(xyt_g))
            gate_vals = []
            for cell in model.cells:
                g = torch.sigmoid(cell.gate_linear(u))
                gate_vals.append(g)
                u = cell(u)
            loss_gate = torch.stack(gate_vals).mean()
            gate_rate  = loss_gate.item()

        loss = loss_pde + W_IC * loss_ic + W_BC * loss_bc + w_gate * loss_gate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0 or epoch == 1:
            l2 = compute_l2_2d(model)
            elapsed = time.time() - t0
            hist["epochs_log"].append(epoch)
            hist["loss_log"].append(loss.item())
            hist["l2_log"].append(l2)
            hist["gate_rate_log"].append(gate_rate)
            hist["wall_time_log"].append(elapsed)
            print(
                f"  Epoch {epoch:5d} | Loss={loss.item():.4e} | L2={l2:.4e}"
                + (f" | Gate={gate_rate:.3f}" if is_disp else "")
                + f" | {elapsed:.0f}s"
            )

    print(f"{'='*58}")
    print(f"  完成！最终 L2={hist['l2_log'][-1]:.4e}"
          + (f"  Gate={hist['gate_rate_log'][-1]:.3f}" if is_disp else ""))
    print(f"{'='*58}\n")
    return hist


# ══════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════
baseline_2d = BaselinePINN2D(hidden_dim=HIDDEN_DIM, depth=DEPTH)
bh2d = train_2d(baseline_2d, "2D Baseline MLP PINN")
torch.save({"model": baseline_2d.state_dict(), "history": bh2d},
           "results/2d_baseline_history.pt")

disp_2d = DisplacementPINN2D(hidden_dim=HIDDEN_DIM, depth=DEPTH)
dh2d = train_2d(disp_2d, "2D Displacement-Gated PINN", w_gate=W_GATE)
torch.save({"model": disp_2d.state_dict(), "history": dh2d},
           "results/2d_displacement_history.pt")

# ══════════════════════════════════════════════
# 绘图
# ══════════════════════════════════════════════
DARK_BG = "#0d1117"; PANEL_BG = "#161b22"; GRID_COLOR = "#21262d"
TEXT_COLOR = "#e6edf3"; MUTED = "#8b949e"
BLUE = "#58a6ff"; GREEN = "#3fb950"; ORANGE = "#d29922"; RED = "#f78166"

def apply_dark(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID_COLOR, lw=0.5, linestyle="--", alpha=0.6)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)

# 图1：对比曲线
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), facecolor=DARK_BG)
ep_b, ep_d = bh2d["epochs_log"], dh2d["epochs_log"]

for ax in axes: apply_dark(ax)

axes[0].semilogy(ep_b, bh2d["loss_log"], color=BLUE, lw=2, label="2D Baseline")
axes[0].semilogy(ep_d, dh2d["loss_log"], color=GREEN, lw=2, ls="--", label="2D Displacement")
axes[0].set_title("(a) Training Loss", color=TEXT_COLOR)
axes[0].legend(facecolor="#21262d", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
axes[0].set_xlabel("Epoch", color=MUTED)

axes[1].semilogy(ep_b, bh2d["l2_log"], color=BLUE, lw=2, label="2D Baseline")
axes[1].semilogy(ep_d, dh2d["l2_log"], color=GREEN, lw=2, ls="--", label="2D Displacement")
axes[1].set_title("(b) L2 Error (2D Maxwell)", color=TEXT_COLOR)
axes[1].legend(facecolor="#21262d", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
axes[1].set_xlabel("Epoch", color=MUTED)

axes[2].plot(ep_d, dh2d["gate_rate_log"], color=ORANGE, lw=2)
axes[2].fill_between(ep_d, dh2d["gate_rate_log"], alpha=0.15, color=ORANGE)
axes[2].axhline(0.5, color=MUTED, lw=1, ls=":")
axes[2].set_ylim(0, 0.7)
axes[2].set_title("(c) Gate Sparsity (2D)", color=TEXT_COLOR)
axes[2].set_xlabel("Epoch", color=MUTED)

fig.suptitle(f"2D Maxwell TM PINN · Cavity Mode (ω={OMEGA:.3f}) · GPU={DEVICE.upper()}",
             color=TEXT_COLOR, fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("results/2d_comparison_plots.png", dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close()

# 图2：Ez 场云图（2D 切片，t=T/2）
baseline_2d.eval(); disp_2d.eval()
ng = 60
xs_1d = torch.linspace(0, 1, ng, device=DEVICE)
ys_1d = torch.linspace(0, 1, ng, device=DEVICE)
xx, yy = torch.meshgrid(xs_1d, ys_1d, indexing="ij")
x_flat = xx.reshape(-1, 1)
y_flat = yy.reshape(-1, 1)
t_flat = torch.full_like(x_flat, T / 2)
xyt = torch.cat([x_flat, y_flat, t_flat], dim=-1)

with torch.no_grad():
    Ez_b = baseline_2d(xyt)[:, 0].cpu().numpy().reshape(ng, ng)
    Ez_d = disp_2d(xyt)[:, 0].cpu().numpy().reshape(ng, ng)
    Ez_t, _, _ = exact2d(x_flat, y_flat, t_flat)
    Ez_t = Ez_t.cpu().numpy().reshape(ng, ng)

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=DARK_BG)
titles = [f"Exact Ez (t={T/2})", "Baseline PINN Ez", "Displacement PINN Ez"]
datas  = [Ez_t, Ez_b, Ez_d]
for ax, dat, ttl in zip(axes2, datas, titles):
    ax.set_facecolor(PANEL_BG)
    im = ax.imshow(dat.T, origin="lower", extent=[0,1,0,1], cmap="RdBu_r",
                   vmin=-1, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(ttl, color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=MUTED)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)

fig2.suptitle(f"2D Maxwell Ez Field Slice at t = {T/2:.2f}",
              color=TEXT_COLOR, fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("results/2d_field_slice.png", dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close()

# 终端摘要
print("\n" + "═"*58)
print("  📊 2D Maxwell 实验结果")
print("═"*58)
print(f"  Baseline   L2: {bh2d['l2_log'][-1]:.4e}")
print(f"  DispField  L2: {dh2d['l2_log'][-1]:.4e}")
print(f"  Gate 稀疏比: {(1-dh2d['gate_rate_log'][-1])*100:.1f}%")
print(f"  输出文件: results/2d_comparison_plots.png")
print(f"           results/2d_field_slice.png")
print("═"*58 + "\n")
