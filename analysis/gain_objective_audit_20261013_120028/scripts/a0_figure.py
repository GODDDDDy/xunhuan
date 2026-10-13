#!/usr/bin/env python3
"""A0 figure: judgement-cache channel magnitude disparity. Stdlib only (no numpy/matplotlib).

Reads metrics/A0_numeric_audit.json and writes figures/input_scale.svg.
"""
import json
import math
from pathlib import Path
import sys

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
data = json.loads((RUN / "metrics" / "A0_numeric_audit.json").read_text(encoding="utf-8"))
stats = data["channel_stats_judgement_only"]
mean = stats["mean"]
std = stats["std"]
n = len(mean)

W, H = 1180, 560
LEFT, RIGHT, TOP, BOTTOM = 70, 30, 70, 70
panel_w = (W - LEFT - RIGHT - 60) / 2
plot_h = H - TOP - BOTTOM


def band(x0, x1, color):
    return (x0, x1, color)


bands = [
    band(8.5, 11.5, "#f2cf5b"),    # delta mean 9-11
    band(18.5, 26.5, "#9ecae1"),   # projected feature 19-26
    band(26.5, 42.5, "#c7c7c7"),   # state 27-42
    band(42.5, 45.5, "#54a24b"),   # delta^2 43-45
    band(45.5, 48.5, "#b279a2"),   # |delta| 46-48
    band(48.5, 49.5, "#ff9da6"),   # ratio 49
]


def panel(x_off, values, title, ylab, logscale):
    parts = []
    # background bands
    for (x0, x1, color) in bands:
        bx0 = x_off + (x0 + 0.5) / n * panel_w
        bx1 = x_off + (x1 + 0.5) / n * panel_w
        parts.append(f'<rect x="{bx0:.2f}" y="{TOP}" width="{bx1-bx0:.2f}" height="{plot_h}" fill="{color}" opacity="0.35"/>')
    vmax = max(values)
    vmin = min(values)
    if logscale:
        lo = math.log10(max(min([v for v in values if v > 0]), 1e-6))
        hi = math.log10(vmax)
        span = hi - lo

        def ty(v):
            lv = math.log10(max(v, 1e-6))
            return TOP + plot_h * (1 - (lv - lo) / span)
    else:
        lo = min(0.0, vmin)
        hi = vmax * 1.05 if vmax > 0 else 1.0
        span = hi - lo

        def ty(v):
            return TOP + plot_h * (1 - (v - lo) / span)
    zero_y = ty(0.0) if not logscale else TOP + plot_h
    bw = panel_w / n * 0.72
    for i, v in enumerate(values):
        cx = x_off + (i + 0.5) / n * panel_w
        y = ty(v)
        color = "#4c78a8" if not logscale else "#e45756"
        parts.append(f'<rect x="{cx-bw/2:.2f}" y="{min(y,zero_y):.2f}" width="{bw:.2f}" height="{abs(zero_y-y):.2f}" fill="{color}"/>')
    # axes
    parts.append(f'<line x1="{x_off}" y1="{zero_y:.2f}" x2="{x_off+panel_w:.2f}" y2="{zero_y:.2f}" stroke="#333" stroke-width="1"/>')
    parts.append(f'<line x1="{x_off}" y1="{TOP}" x2="{x_off}" y2="{TOP+plot_h}" stroke="#333" stroke-width="1"/>')
    parts.append(f'<text x="{x_off+panel_w/2:.0f}" y="{TOP-24}" text-anchor="middle" font-size="15" fill="#222">{title}</text>')
    parts.append(f'<text x="{x_off-52}" y="{TOP+plot_h/2:.0f}" text-anchor="middle" font-size="12" fill="#444" transform="rotate(-90 {x_off-52} {TOP+plot_h/2:.0f})">{ylab}</text>')
    for i in range(0, n, 5):
        cx = x_off + (i + 0.5) / n * panel_w
        parts.append(f'<text x="{cx:.1f}" y="{TOP+plot_h+18}" text-anchor="middle" font-size="11" fill="#555">{i}</text>')
    return parts


svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="DejaVu Sans, Arial, sans-serif">')
svg.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
svg.append(f'<text x="{LEFT}" y="34" font-size="19" font-weight="bold" fill="#111">A0 · 判断训练缓存输入量级差异（仅训练角色，无GT）</text>')
svg.append(f'<text x="{LEFT}" y="54" font-size="12" fill="#555">50维判断输入：RGB/T1/候选类通道 std≈0.2–0.45；增量类通道 std≈4e-4–8e-3；无近零方差(&lt;1e-6)通道</text>')
svg += panel(LEFT, mean, "通道均值 channel mean", "mean", False)
svg += panel(LEFT + panel_w + 60, std, "通道标准差 std（对数刻度）", "std (log10)", True)

# legend
lx, ly = LEFT, H - 26
legend = [("#f2cf5b", "Δ均值 9-11"), ("#9ecae1", "投影特征 19-26"), ("#c7c7c7", "跨轮状态 27-42"),
          ("#54a24b", "Δ²均值 43-45"), ("#b279a2", "|Δ|均值 46-48"), ("#ff9da6", "抵消比 49")]
for i, (c, t) in enumerate(legend):
    x = lx + i * 175
    svg.append(f'<rect x="{x}" y="{ly-11}" width="12" height="12" fill="{c}" opacity="0.7"/>')
    svg.append(f'<text x="{x+17}" y="{ly}" font-size="12" fill="#333">{t}</text>')
svg.append('</svg>')

outp = RUN / "figures" / "input_scale.svg"
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text("\n".join(svg), encoding="utf-8")
print("wrote", outp, outp.stat().st_size)
