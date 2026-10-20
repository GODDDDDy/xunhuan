#!/usr/bin/env python3
"""A2 figures: MLP (A1) + native head (A2) fit/ranking and risk-coverage.

Stdlib only (the server has no matplotlib). Reads
  metrics/A1_probe_results.json
  metrics/A2_native_fit_results.json
and rewrites
  figures/train_fit_and_cost_ranking.svg
  figures/risk_coverage.svg
so that both diagnostics appear in one place with the architecture named in the
legend, as required by the handoff (15.1 items 1-2).
"""
import json
import sys
from pathlib import Path

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
a1 = json.loads((RUN / "metrics" / "A1_probe_results.json").read_text(encoding="utf-8"))
a2 = json.loads((RUN / "metrics" / "A2_native_fit_results.json").read_text(encoding="utf-8"))

C_MIX = "#4c78a8"
C_PURE = "#e45756"
C_RAND = "#8c8c8c"
C_ACC = "#54a24b"
C_ORACLE = "#b279a2"


def panel(parts, x0, y0, w, h, ylo, yhi, title, xlab, ylab, xticks, fmtx=lambda v: f"{v:g}"):
    def ty(v):
        return y0 + h * (1 - (v - ylo) / (yhi - ylo))
    parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#fbfbfb" stroke="#dddddd"/>')
    for k in range(6):
        yv = ylo + (yhi - ylo) * k / 5
        y = ty(yv)
        parts.append(f'<line x1="{x0:.1f}" y1="{y:.2f}" x2="{x0+w:.1f}" y2="{y:.2f}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{x0-8:.1f}" y="{y+4:.2f}" text-anchor="end" font-size="10.5" fill="#666">{yv:.4g}</text>')
    parts.append(f'<line x1="{x0:.1f}" y1="{y0+h:.1f}" x2="{x0+w:.1f}" y2="{y0+h:.1f}" stroke="#333"/>')
    parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y0+h:.1f}" stroke="#333"/>')
    span = xticks[-1] - xticks[0] or 1
    for xv in xticks:
        x = x0 + (xv - xticks[0]) / span * w
        parts.append(f'<text x="{x:.1f}" y="{y0+h+18:.1f}" text-anchor="middle" font-size="10.5" fill="#555">{fmtx(xv)}</text>')
    parts.append(f'<text x="{x0+w/2:.0f}" y="{y0-24:.0f}" text-anchor="middle" font-size="13.5" fill="#222">{title}</text>')
    parts.append(f'<text x="{x0+w/2:.0f}" y="{y0+h+38:.0f}" text-anchor="middle" font-size="11.5" fill="#444">{xlab}</text>')
    parts.append(f'<text x="{x0-58:.0f}" y="{y0+h/2:.0f}" text-anchor="middle" font-size="11.5" fill="#444" '
                 f'transform="rotate(-90 {x0-58:.0f} {y0+h/2:.0f})">{ylab}</text>')
    return ty


def series(parts, x0, w, ty, xs, ys, color, width=2.2, dash=None, marker=True, marker_r=2.6):
    x_lo, x_hi = xs[0], xs[-1]
    span = (x_hi - x_lo) or 1
    da = f' stroke-dasharray="{dash}"' if dash else ""
    pts = [f"{x0 + (x - x_lo) / span * w:.2f},{ty(y):.2f}" for x, y in zip(xs, ys)]
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{width}"{da}/>')
    if marker:
        for x, y in zip(xs, ys):
            parts.append(f'<circle cx="{x0 + (x - x_lo) / span * w:.2f}" cy="{ty(y):.2f}" r="{marker_r}" fill="{color}"/>')


def vbar(parts, x0, w, ty, xs, ys, color, width=14, dash=None, offset=0.0):
    x_lo, x_hi = xs[0], xs[-1]
    span = (x_hi - x_lo) or 1
    da = f' stroke-dasharray="{dash}"' if dash else ""
    for x, y in zip(xs, ys):
        px = x0 + (x - x_lo) / span * w + offset
        parts.append(f'<line x1="{px:.2f}" y1="{ty(0.0):.2f}" x2="{px:.2f}" y2="{ty(y):.2f}" '
                     f'stroke="{color}" stroke-width="{width}"{da}/>')


def legend(parts, items, y, x_start, step=250):
    for i, (c, t, dash) in enumerate(items):
        x = x_start + i * step
        da = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{x}" y1="{y-4}" x2="{x+28}" y2="{y-4}" stroke="{c}" stroke-width="2.6"{da}/>')
        parts.append(f'<text x="{x+36}" y="{y}" font-size="12" fill="#333">{t}</text>')


# =============================================== figure 1: fit & ranking
W1, H1 = 1180, 720
LEFT, TOP, GAP = 78, 112, 88
PANEL_W = (W1 - LEFT - 44 - GAP) / 2
PANEL_H = (H1 - TOP - 104 - GAP) / 2

s1 = a1["training"]
steps1 = [r["step"] for r in s1["mixed"]]
a1_fit = {"mixed": [r["loss_fit"] for r in s1["mixed"]], "pure": [r["loss_fit"] for r in s1["pure"]]}
a1_sp = {"mixed": [r["spearman_cost_mean"] for r in s1["mixed"]], "pure": [r["spearman_cost_mean"] for r in s1["pure"]]}

steps2 = sorted(int(k) for k in a2["heads"]["mixed"]["monitors"])
a2_fit, a2_sp = {}, {}
for mode in ("mixed", "pure"):
    a2_fit[mode] = [a2["heads"][mode]["monitors"][str(st)]["pure_fit"] for st in steps2]
    a2_sp[mode] = [sum(v for v in a2["heads"][mode]["monitors"][str(st)]["spearman_by_action"].values() if v is not None)
                   / max(1, sum(1 for v in a2["heads"][mode]["monitors"][str(st)]["spearman_by_action"].values() if v is not None))
                   for st in steps2]

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W1}" height="{H1}" viewBox="0 0 {W1} {H1}" '
       f'font-family="DejaVu Sans, Arial, sans-serif">']
svg.append(f'<rect width="{W1}" height="{H1}" fill="#ffffff"/>')
svg.append(f'<text x="{LEFT}" y="34" font-size="18" font-weight="bold" fill="#111">'
           f'A1 线性/MLP 探针 与 A2 原生 50 通道判断头：纯拟合与三动作排序（两条曲线仅损失不同）</text>')
svg.append(f'<text x="{LEFT}" y="54" font-size="11.5" fill="#555">'
           f'mixed = L_fit + 0.1·L_policy（含期望项），pure = L_fit；两副本共享同一初值与相同样本顺序</text>')
svg.append(f'<text x="{LEFT}" y="70" font-size="11.5" fill="#a33">'
           f'注意纵轴分列：A1 使用自己的标准化尺度（1000 步、2048 训练区域），A2 使用第 22 项来源尺度（500 步、24 裁剪），两者数值不可直接比较</text>')

x_left, x_right = LEFT, LEFT + PANEL_W + GAP
y_top, y_bot = TOP, TOP + PANEL_H + GAP

ty = panel(svg, x_left, y_top, PANEL_W, PANEL_H,
           min(min(a1_fit["mixed"]), min(a1_fit["pure"])) * 0.96,
           max(max(a1_fit["mixed"]), max(a1_fit["pure"])) * 1.02,
           "A1 · MLP 探针：纯拟合损失 L_fit（越小越好）", "训练步 step", "L_fit", steps1)
series(svg, x_left, PANEL_W, ty, steps1, a1_fit["mixed"], C_MIX)
series(svg, x_left, PANEL_W, ty, steps1, a1_fit["pure"], C_PURE, dash="6 4")

ty = panel(svg, x_right, y_top, PANEL_W, PANEL_H,
           min(min(a2_fit["mixed"]), min(a2_fit["pure"])) * 0.985,
           max(max(a2_fit["mixed"]), max(a2_fit["pure"])) * 1.005,
           "A2 · 原生 50 通道头：纯拟合损失 L_fit（越小越好）", "训练步 step", "L_fit", steps2)
series(svg, x_right, PANEL_W, ty, steps2, a2_fit["mixed"], C_MIX)
series(svg, x_right, PANEL_W, ty, steps2, a2_fit["pure"], C_PURE, dash="6 4")

lo1, hi1 = min(min(a1_sp["mixed"]), min(a1_sp["pure"])), max(max(a1_sp["mixed"]), max(a1_sp["pure"]))
ty = panel(svg, x_left, y_bot, PANEL_W, PANEL_H, lo1 - 0.03, hi1 + 0.03,
           "A1 · MLP 探针：三动作平均 Spearman（训练内）", "训练步 step", "Spearman", steps1)
series(svg, x_left, PANEL_W, ty, steps1, a1_sp["mixed"], C_MIX)
series(svg, x_left, PANEL_W, ty, steps1, a1_sp["pure"], C_PURE, dash="6 4")

lo2, hi2 = min(min(a2_sp["mixed"]), min(a2_sp["pure"])), max(max(a2_sp["mixed"]), max(a2_sp["pure"]))
ty = panel(svg, x_right, y_bot, PANEL_W, PANEL_H, lo2 - 0.02, hi2 + 0.02,
           "A2 · 原生 50 通道头：三动作平均 Spearman（训练内）", "训练步 step", "Spearman", steps2)
series(svg, x_right, PANEL_W, ty, steps2, a2_sp["mixed"], C_MIX)
series(svg, x_right, PANEL_W, ty, steps2, a2_sp["pure"], C_PURE, dash="6 4")

legend(svg, [(C_MIX, "mixed（L_fit + 0.1·L_policy）", None), (C_PURE, "pure（仅 L_fit）", "6 4")], H1 - 24, LEFT, 300)
svg.append('</svg>')
(RUN / "figures").mkdir(parents=True, exist_ok=True)
(RUN / "figures" / "train_fit_and_cost_ranking.svg").write_text("\n".join(svg), encoding="utf-8")

# =============================================== figure 2: risk-coverage
W2, H2 = 1180, 640
LEFT2, TOP2, GAP2 = 78, 118, 52
PANEL_W2 = (W2 - LEFT2 - 44 - 2 * GAP2) / 3
PANEL_H2 = H2 - TOP2 - 104

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W2}" height="{H2}" viewBox="0 0 {W2} {H2}" '
       f'font-family="DejaVu Sans, Arial, sans-serif">']
svg.append(f'<rect width="{W2}" height="{H2}" fill="#ffffff"/>')
svg.append(f'<text x="{LEFT2}" y="34" font-size="18" font-weight="bold" fill="#111">'
           f'训练安全角色的风险—覆盖 / 净收益（小样本诊断，不是部署证据）</text>')
svg.append(f'<text x="{LEFT2}" y="54" font-size="11.5" fill="#555">'
           f'左 = A1 MLP 强制覆盖（按预测收益取前 1/5/10/20% 区域）；中/右 = A2 原生头旧门槛网格（0…12.8，单位为归一化成本 × 尺度 9.344e-4）</text>')
svg.append(f'<text x="{LEFT2}" y="70" font-size="11.5" fill="#a33">'
           f'A2 两条曲线均在 margin≥0.8 时覆盖降为 0；唯一合格工作点 = 全部拒绝，故 safe=true / useful=false</text>')

# --- panel 1 : A1 forced coverage
rc = a1["risk_coverage"]
cov = [c["coverage_area"] * 100 for c in rc["mixed"]]
p1_m = [c["accepted_region_mean_improvement_raw"] for c in rc["mixed"]]
p1_p = [c["accepted_region_mean_improvement_raw"] for c in rc["pure"]]
p1_r = [c["same_action_random_accepted_mean_improvement_raw"] for c in rc["mixed"]]
vals = p1_m + p1_p + p1_r + [0.0]
rngv = max(vals) - min(vals) or 1e-9
ty = panel(svg, LEFT2, TOP2, PANEL_W2, PANEL_H2, min(vals) - 0.08 * rngv, max(vals) + 0.22 * rngv,
           "A1 · MLP 强制覆盖（预测）", "覆盖面积 coverage (%)", "接受区域真实净改善 (raw MSE)",
           cov, fmtx=lambda v: f"{v:.0f}%")
series(svg, LEFT2, PANEL_W2, ty, cov, p1_m, C_MIX)
series(svg, LEFT2, PANEL_W2, ty, cov, p1_p, C_PURE, dash="6 4")
series(svg, LEFT2, PANEL_W2, ty, cov, p1_r, C_RAND, width=1.7)
y0 = ty(0.0)
svg.append(f'<line x1="{LEFT2}" y1="{y0:.2f}" x2="{LEFT2+PANEL_W2:.1f}" y2="{y0:.2f}" '
           f'stroke="{C_ACC}" stroke-width="1.6" stroke-dasharray="4 3"/>')
svg.append(f'<text x="{LEFT2+PANEL_W2-4:.1f}" y="{y0-6:.2f}" text-anchor="end" font-size="10" fill="{C_ACC}">全拒绝 = 0</text>')

# --- panel 2 : A2 coverage vs margin
grid = a2["heads"]["mixed"]["screening"]["grid"]
margins = [r["margin"] for r in grid]
cov_m = [r["coverage"] * 100 for r in grid]
cov_p = [r["coverage"] * 100 for r in a2["heads"]["pure"]["screening"]["grid"]]
ty = panel(svg, LEFT2 + PANEL_W2 + GAP2, TOP2, PANEL_W2, PANEL_H2, 0.0, max(cov_m + cov_p) * 1.18 or 1.0,
           "A2 · 原生头：接受覆盖率 vs 门槛（预测）", "旧门槛 margin（归一化成本）", "覆盖率 (%)", margins)
vbar(svg, LEFT2 + PANEL_W2 + GAP2, PANEL_W2, ty, margins, cov_m, C_MIX, width=11, offset=-7)
vbar(svg, LEFT2 + PANEL_W2 + GAP2, PANEL_W2, ty, margins, cov_p, C_PURE, width=11, offset=7, dash="3 2")

# --- panel 3 : A2 net improvement vs margin
imp_m = [r["mean_mse_improvement"] for r in grid]
imp_p = [r["mean_mse_improvement"] for r in a2["heads"]["pure"]["screening"]["grid"]]
oracle = a2["heads"]["mixed"]["screening"]["reference_accept_all"]["mean_mse_improvement"]
acc_all = a2["heads"]["mixed"]["screening"]["reference_reject_all"]["mean_mse_improvement"]
vals3 = imp_m + imp_p + [0.0, oracle]
rng3 = max(vals3) - min(vals3) or 1e-9
ty = panel(svg, LEFT2 + 2 * (PANEL_W2 + GAP2), TOP2, PANEL_W2, PANEL_H2,
           min(vals3) - 0.25 * rng3, max(vals3) + 0.25 * rng3,
           "A2 · 原生头：选定动作的真实净收益", "旧门槛 margin（归一化成本）", "平均 MSE 改善 (raw)", margins)
series(svg, LEFT2 + 2 * (PANEL_W2 + GAP2), PANEL_W2, ty, margins, imp_m, C_MIX, marker_r=2.2)
series(svg, LEFT2 + 2 * (PANEL_W2 + GAP2), PANEL_W2, ty, margins, imp_p, C_PURE, dash="6 4", marker_r=2.2)
y0 = ty(0.0)
svg.append(f'<line x1="{LEFT2+2*(PANEL_W2+GAP2):.1f}" y1="{y0:.2f}" x2="{LEFT2+2*(PANEL_W2+GAP2)+PANEL_W2:.1f}" '
           f'y2="{y0:.2f}" stroke="{C_ACC}" stroke-width="1.6" stroke-dasharray="4 3"/>')
yo = ty(oracle)
svg.append(f'<line x1="{LEFT2+2*(PANEL_W2+GAP2):.1f}" y1="{yo:.2f}" x2="{LEFT2+2*(PANEL_W2+GAP2)+PANEL_W2:.1f}" '
           f'y2="{yo:.2f}" stroke="{C_ORACLE}" stroke-width="1.8" stroke-dasharray="2 3"/>')
svg.append(f'<text x="{LEFT2+2*(PANEL_W2+GAP2)+4:.1f}" y="{yo-5:.2f}" font-size="10" fill="{C_ORACLE}">'
           f'GT 参照：每格最优动作上界 = {oracle:.2e}</text>')
svg.append(f'<text x="{LEFT2+2*(PANEL_W2+GAP2)+PANEL_W2-4:.1f}" y="{y0-6:.2f}" text-anchor="end" font-size="10" fill="{C_ACC}">'
           f'全拒绝 = {acc_all:g}</text>')

legend(svg, [(C_MIX, "mixed", None), (C_PURE, "pure", "6 4"), (C_RAND, "A1 同动作随机参照", None),
             (C_ORACLE, "GT 参照（oracle 上界）", "2 3")], H2 - 28, LEFT2, 285)
svg.append(f'<text x="{W2-40}" y="{H2-8}" text-anchor="end" font-size="10.5" fill="#888">'
           f'最多 32 张原尺寸验证筛查的风险—覆盖在 A3 后追加；本次 A2 未使用 test，未使用 validation 选择任何规则</text>')
svg.append('</svg>')
(RUN / "figures" / "risk_coverage.svg").write_text("\n".join(svg), encoding="utf-8")

print("wrote", RUN / "figures" / "train_fit_and_cost_ranking.svg", RUN / "figures" / "risk_coverage.svg")
