#!/usr/bin/env python3
"""A1 figures: training curves and risk-coverage. Stdlib only (no numpy/matplotlib).

Reads metrics/A1_probe_results.json and writes
  figures/train_fit_and_cost_ranking.svg
  figures/risk_coverage.svg
"""
import json
import sys
from pathlib import Path

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
d = json.loads((RUN / "metrics" / "A1_probe_results.json").read_text(encoding="utf-8"))

W, H = 1180, 560
LEFT, RIGHT, TOP, BOTTOM = 78, 34, 74, 74

C_MIX = "#4c78a8"
C_PURE = "#e45756"
C_RAND = "#8c8c8c"
C_ACC = "#54a24b"


def axes(parts, x_off, pad_w, ylo, yhi, title, xlab, ylab, xticks, fmtx=lambda v: f"{v:g}"):
    plot_h = H - TOP - BOTTOM
    def ty(v):
        return TOP + plot_h * (1 - (v - ylo) / (yhi - ylo))
    parts.append(f'<rect x="{x_off}" y="{TOP}" width="{pad_w}" height="{plot_h}" fill="#fbfbfb" stroke="#dddddd"/>')
    for k in range(6):
        yv = ylo + (yhi - ylo) * k / 5
        y = ty(yv)
        parts.append(f'<line x1="{x_off}" y1="{y:.2f}" x2="{x_off+pad_w}" y2="{y:.2f}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{x_off-8}" y="{y+4:.2f}" text-anchor="end" font-size="11" fill="#666">{yv:.4g}</text>')
    parts.append(f'<line x1="{x_off}" y1="{TOP+plot_h}" x2="{x_off+pad_w}" y2="{TOP+plot_h}" stroke="#333"/>')
    parts.append(f'<line x1="{x_off}" y1="{TOP}" x2="{x_off}" y2="{TOP+plot_h}" stroke="#333"/>')
    for xv in xticks:
        x = x_off + (xv - xticks[0]) / (xticks[-1] - xticks[0]) * pad_w
        parts.append(f'<text x="{x:.1f}" y="{TOP+plot_h+20}" text-anchor="middle" font-size="11" fill="#555">{fmtx(xv)}</text>')
    parts.append(f'<text x="{x_off+pad_w/2:.0f}" y="{TOP-30}" text-anchor="middle" font-size="15" fill="#222">{title}</text>')
    parts.append(f'<text x="{x_off+pad_w/2:.0f}" y="{TOP+plot_h+46}" text-anchor="middle" font-size="12" fill="#444">{xlab}</text>')
    parts.append(f'<text x="{x_off-58}" y="{TOP+plot_h/2:.0f}" text-anchor="middle" font-size="12" fill="#444" transform="rotate(-90 {x_off-58} {TOP+plot_h/2:.0f})">{ylab}</text>')
    return ty


def line(parts, x_off, pad_w, ty, xs, ys, color, width=2.2, dash=None):
    x0, x1 = min(xs), max(xs)
    pts = []
    for x, y in zip(xs, ys):
        px = x_off + (x - x0) / (x1 - x0) * pad_w
        pts.append(f"{px:.2f},{ty(y):.2f}")
    da = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{width}"{da}/>')
    for x, y in zip(xs, ys):
        px = x_off + (x - x0) / (x1 - x0) * pad_w
        parts.append(f'<circle cx="{px:.2f}" cy="{ty(y):.2f}" r="3" fill="{color}"/>')


# ------------------------------------------------- figure 1: fit & ranking
cm, cp = d["training"]["mixed"], d["training"]["pure"]
steps = [r["step"] for r in cm]
fit_m = [r["loss_fit"] for r in cm]
fit_p = [r["loss_fit"] for r in cp]
sp_m = [r["spearman_cost_mean"] for r in cm]
sp_p = [r["spearman_cost_mean"] for r in cp]

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="DejaVu Sans, Arial, sans-serif">']
svg.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
svg.append(f'<text x="{LEFT}" y="34" font-size="19" font-weight="bold" fill="#111">A1 · 散点表示上的轻量学习性诊断（MLP 50-32-32-3，CPU，2048训练区域）</text>')
svg.append(f'<text x="{LEFT}" y="54" font-size="12" fill="#555">两副本共享同一初始权重（SHA c84ea6f5…）与相同批次顺序，仅损失不同：mixed=L_fit+0.1·L_policy，pure=L_fit</text>')
pad_w = (W - LEFT - RIGHT - 70) / 2
ty1 = axes(svg, LEFT, pad_w, min(min(fit_m), min(fit_p)) * 0.95, max(max(fit_m), max(fit_p)) * 1.02,
           "纯拟合损失 L_fit（越小越好）", "训练步 step", "L_fit", steps)
line(svg, LEFT, pad_w, ty1, steps, fit_m, C_MIX)
line(svg, LEFT, pad_w, ty1, steps, fit_p, C_PURE, dash="6 4")
x2 = LEFT + pad_w + 70
lo = min(min(sp_m), min(sp_p)) - 0.02
hi = max(max(sp_m), max(sp_p)) + 0.02
ty2 = axes(svg, x2, pad_w, lo, hi, "三动作平均 Spearman（训练内样本）", "训练步 step", "Spearman", steps)
line(svg, x2, pad_w, ty2, steps, sp_m, C_MIX)
line(svg, x2, pad_w, ty2, steps, sp_p, C_PURE, dash="6 4")
lx, ly = LEFT, H - 26
for i, (c, t, dash) in enumerate([(C_MIX, "mixed（含期望项）", None), (C_PURE, "pure（仅 L_fit）", "6 4")]):
    x = lx + i * 230
    da = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<line x1="{x}" y1="{ly-4}" x2="{x+30}" y2="{ly-4}" stroke="{c}" stroke-width="2.6"{da}/>')
    svg.append(f'<text x="{x+38}" y="{ly}" font-size="12.5" fill="#333">{t}</text>')
svg.append('</svg>')
(RUN / "figures").mkdir(parents=True, exist_ok=True)
(RUN / "figures" / "train_fit_and_cost_ranking.svg").write_text("\n".join(svg), encoding="utf-8")

# ------------------------------------------------- figure 2: risk coverage
rc = d["risk_coverage"]
cov = [c["coverage_area"] * 100 for c in rc["mixed"]]
am = [c["accepted_region_mean_improvement_raw"] for c in rc["mixed"]]
ap = [c["accepted_region_mean_improvement_raw"] for c in rc["pure"]]
ar = [c["same_action_random_accepted_mean_improvement_raw"] for c in rc["mixed"]]

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="DejaVu Sans, Arial, sans-serif">']
svg.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
svg.append(f'<text x="{LEFT}" y="34" font-size="19" font-weight="bold" fill="#111">A1 · 安全角色风险—覆盖诊断（强制覆盖，非部署）</text>')
svg.append(f'<text x="{LEFT}" y="54" font-size="12" fill="#555">按预测收益取前 1%/5%/10%/20% 区域，统计被接受区域的平均真实净改善（原始单位，>0 才是有益）</text>')
pad_w = W - LEFT - RIGHT - 210
vals = am + ap + ar + [0.0]
rngv = max(vals) - min(vals)
ylo = min(vals) - 0.08 * rngv
yhi = max(vals) + 0.20 * rngv
ty = axes(svg, LEFT, pad_w, ylo, yhi, "接受区域平均真实净改善（原尺度）", "覆盖面积 coverage (%)", "MSE 改善 (raw)",
          cov, fmtx=lambda v: f"{v:.0f}%")
line(svg, LEFT, pad_w, ty, cov, am, C_MIX)
line(svg, LEFT, pad_w, ty, cov, ap, C_PURE, dash="6 4")
line(svg, LEFT, pad_w, ty, cov, ar, C_RAND, width=1.8)
y0 = ty(0.0)
svg.append(f'<line x1="{LEFT}" y1="{y0:.2f}" x2="{LEFT+pad_w}" y2="{y0:.2f}" stroke="{C_ACC}" stroke-width="1.6" stroke-dasharray="4 3"/>')
svg.append(f'<text x="{LEFT+pad_w-6}" y="{y0-6:.2f}" text-anchor="end" font-size="11" fill="{C_ACC}">全拒绝/无变化 = 0</text>')
lx, ly = LEFT, H - 26
for i, (c, t, dash) in enumerate([(C_MIX, "mixed 预测", None), (C_PURE, "pure 预测", "6 4"), (C_RAND, "同动作随机参照", None)]):
    x = lx + i * 230
    da = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<line x1="{x}" y1="{ly-4}" x2="{x+30}" y2="{ly-4}" stroke="{c}" stroke-width="2.4"{da}/>')
    svg.append(f'<text x="{x+38}" y="{ly}" font-size="12.5" fill="#333">{t}</text>')
svg.append(f'<text x="{W-RIGHT}" y="{H-8}" text-anchor="end" font-size="11" fill="#888">signal@10%: mixed=false, pure=false（4场景中仅1个改善）</text>')
svg.append('</svg>')
(RUN / "figures" / "risk_coverage.svg").write_text("\n".join(svg), encoding="utf-8")
print("wrote", RUN / "figures" / "train_fit_and_cost_ranking.svg", RUN / "figures" / "risk_coverage.svg")
