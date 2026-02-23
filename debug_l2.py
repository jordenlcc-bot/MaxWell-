"""
debug_l2.py — 诊断 1D Maxwell PINN 的 L2 计算问题
"""
import torch, math
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 简单 2 层 PINN ───────────────────────────────────
class TinyPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 32)
        self.l2 = nn.Linear(32, 2)
        nn.init.xavier_uniform_(self.l1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.l2.weight, gain=0.1)
    def forward(self, xt):
        return self.l2(torch.sin(self.l1(xt)))

model = TinyPINN().to(DEVICE)

# ── 评估点 ────────────────────────────────────────────
n = 300
xs = torch.linspace(0, 1, n, device=DEVICE).unsqueeze(1)  # [300,1]
ts = torch.full((n, 1), 1.0, device=DEVICE)               # [300,1]  t=1
xt = torch.cat([xs, ts], dim=-1)                           # [300,2]

# 解析解
Ez_true = torch.sin(torch.pi * xs) * torch.cos(torch.pi * ts)
Hy_true = -torch.sin(torch.pi * xs) * torch.sin(torch.pi * ts)

print("=== 解析解检查 ===")
print(f"  Ez_true range: [{Ez_true.min():.3f}, {Ez_true.max():.3f}]")
print(f"  Hy_true range: [{Hy_true.min():.3f}, {Hy_true.max():.3f}]")
print(f"  ||Ez_true||_2 = {torch.norm(Ez_true):.4f}")  # 应该 ≈ 10
print(f"  ||Hy_true||_2 = {torch.norm(Hy_true):.4f}")

# 未训练模型输出
with torch.no_grad():
    pred = model(xt)
    Ez_p, Hy_p = pred[:, 0:1], pred[:, 1:2]

print("\n=== 未训练模型输出 ===")
print(f"  Ez_pred range: [{Ez_p.min():.4f}, {Ez_p.max():.4f}]")
print(f"  ||Ez_pred - Ez_true||_2 = {torch.norm(Ez_p - Ez_true):.4f}")
print(f"  Relative L2 (Ez) = {torch.norm(Ez_p - Ez_true) / (torch.norm(Ez_true) + 1e-10):.4f}")
print(f"  Relative L2 (Hy) = {torch.norm(Hy_p - Hy_true) / (torch.norm(Hy_true) + 1e-10):.4f}")

# ── 如果网络全输出 0，L2 应该是 1.0 ──────────────────
ez_if_zero = torch.norm(Ez_true) / (torch.norm(Ez_true) + 1e-10)
print(f"\n  如果模型全输出 0，Ez 相对 L2 应该 = {ez_if_zero:.4f}")

# ── 真正的问题：t=1 时的解析解 ───────────────────────
print(f"\n  cos(π·1) = {math.cos(math.pi):.4f}  ← Ez = sin(πx)·(-1) = -sin(πx)")
print(f"  sin(π·1) = {math.sin(math.pi):.6f}  ← Hy = -sin(πx)·(~0) ≈ 0")

print("\n⚠️  在 t=1 时 Hy ≈ 0 (sin(π)≈0)，会导致 ||Hy_true||→0，")
print("    分母趋于 0 → 相对 L2 爆炸！这是根本来源。")

# 验证
print(f"\n  ||Hy_true(t=1)|| = {torch.norm(Hy_true):.6e}  (几乎 = 0!)")
print(f"  1/||Hy_true||   = {1/(torch.norm(Hy_true)+1e-10):.2e}")
