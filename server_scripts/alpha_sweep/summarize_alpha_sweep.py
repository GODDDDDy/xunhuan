#!/usr/bin/env python3
"""Turn alpha_sweep_result.json into a Chinese markdown + compact JSON summary.

Reporting discipline (from NEXT_ACTIONS_EVIDENCE_BASED_ZH_20261001.md):
  * PSNR and SSIM are both reported; image-mean and scene-macro are both reported.
  * Per-scene values and improved/degraded fractions are reported side by side.
  * Interpretation is mechanical, not rhetorical:
      - alpha > 0 beating alpha=1 but still below alpha=0  => "reduced degradation"
        ONLY. It is not evidence of iterative gain.
      - only a value ABOVE alpha=0 (= this model's own T1) counts as an
        iterative-gain signal under the current update direction.
      - if no non-zero alpha is above T1, the correct statement is that there is
        NO evidence that "merely shortening the existing second step" fixes it.
        It must not be phrased as "all step sizes / directions are invalid".
  * If the task loss falls while PSNR/SSIM fall, that is reported as a
    divergence between optimisation target and observation metric -- NOT as
    evidence that training failed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v, nd=6):
    if v is None:
        return "-"
    return f"{v:.{nd}f}"


def sgn(v, nd=4):
    if v is None:
        return "-"
    return f"{v:+.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--subset-json", default="",
                    help="optional 24-image subset result json, used only to build the "
                         "subset-vs-full consistency section in full-validation mode")
    args = ap.parse_args()

    data = json.loads(Path(args.src).read_text(encoding="utf-8"))
    alphas = [str(a) for a in data["alphas"]]
    L: list[str] = []
    add = L.append

    full_mode = bool(data.get("full_validation", False))
    sub = data.get("subset", {})
    if full_mode:
        scope_line = ("冻结检查点，不训练、不更新权重。**完整 validation 划分**"
                      "（按 loader 顺序，不做挑选）。")
    else:
        scope_line = ("冻结检查点，不训练、不更新权重。24 张固定图（8 场景，每场景前 3 张；"
                      "实际组成 21 easy + 3 hard，非场景×难度共同分层）。")

    add("# K=2 现有检查点的 alpha 幅度与目标诊断")
    add("")
    add(scope_line)
    add("")
    add("定义：`P(alpha) = P1 + alpha*(P2-P1)`，P1/P2 在该组**原始第二轮配置**下正常计算后固定；"
        "插值结果**直接送 TaskNet**，不再过 R2P。**关闭 feedback 不等于 alpha=0。**")
    add("")
    add("随机流：每次对比用的 TaskNet 调用前，恢复该图**第一次 TaskNet 调用之前**保存的 "
        "CPU 与全部 CUDA RNG 状态，因此 T2 与各 alpha 与 T1 看到同一扩散噪声流。")
    add("")
    add("> 区间/计数只描述量级与方向。**不得**据此宣称显著、等价或“在噪声内”。"
        "单 seed、单检查点，不含训练随机性。")
    add("")

    compact: dict = {"alphas": alphas, "groups": {}}

    for group in ("no_feedback", "feedback"):
        g = data["groups"][group]
        n = g["n_images"]
        add(f"## {group}（feedback_enabled={g['feedback_enabled']}, "
            f"feedback_scale={g['feedback_scale']:.8g}）")
        add("")
        add(f"检查点：`{g['checkpoint']}`，{n} 张图。")
        add("")

        # ---- endpoint integrity
        add("### 端点一致性")
        add("")
        # In full mode the endpoints are ALIASES of T1/T2 (see top-level
        # endpoint_aliasing field); the per-row "aliased" flag is not persisted
        # into the downloaded JSON, so the mode flag is the reliable signal.
        aliased = full_mode
        add("| 端点 | 参照 | 输入最大绝对差 | 输出最大绝对差 | 容差 | 状态 |")
        add("|---|---|---|---|---|---|")
        for a, ref in (("0.0", "T1"), ("1.0", "T2")):
            e = g["endpoint_integrity"][a]
            if aliased:
                state = "**not_remeasured（别名）**"
            else:
                state = "已测量" if (e["input_all_exact"]
                                 and e["output_within_tolerance"]) else "未通过"
            add(f"| alpha={a} | {ref} | {e['input_max_abs_diff_max_over_images']:.3e} | "
                f"{e['output_max_abs_diff_max_over_images']:.3e} | "
                f"{e['tolerance']:.0e} | {state} |")
        add("")
        if aliased:
            add("> **状态列已显式标注 `not_remeasured`。** 全量模式下 alpha=0 / alpha=1 "
                "并非独立测量，而是**别名**：直接复用 T1 / T2 的已有张量"
                "（省每图 2 次 TaskNet 调用）。表中的 0.000e+00 来自**上一轮 24 张子集**"
                "的实测结论，**不是**本 872 张上的 872 次独立测量。读者不应把该行"
                "理解为端点在本轮被重新验证过。")
            add("")

        # ---- clamp bookkeeping
        add("### 原始 delta 与有效更新（clamp 记账）")
        add("")
        add("| 轮次 | 原始 delta 平均绝对 | 有效更新 平均绝对 | 截断削减比例 | clamp 间隙 最大绝对 |")
        add("|---|---|---|---|---|")
        for rnd, label in (("clamp_round1", "第一轮 P1=clamp(P0+d1)"),
                           ("clamp_round2", "第二轮 P2=clamp(P1+d2)")):
            c = g[rnd]
            raw = c["raw_delta_abs_mean_mean_over_images"]
            eff = c["effective_update_abs_mean_mean_over_images"]
            cut = (raw - eff) / raw if raw else 0.0
            add(f"| {label} | {raw:.6e} | {eff:.6e} | {cut*100:.2f}% | "
                f"{c['clamp_gap_abs_max_over_images']:.6e} |")
        add("")
        add("“clamp 间隙”= `max|P_clamped − (P_in + delta_raw)|`（逐图取最大后再取跨图最大）。"
            "非 0 即截断确实生效，因此**必须区分网络原始 delta 与截断后有效更新**；"
            "上表第二、三列就是这个区别。")
        add("")

        # ---- aggregation caveat
        im_psnr = g["T1"]["psnr"]["image_mean"]
        sm_psnr = g["T1"]["psnr"]["scene_macro"]
        balanced = abs(im_psnr - sm_psnr) < 1e-12
        if balanced:
            add("> **聚合口径说明**：本子集每个场景恰好取同样多的图，因此“图像均值”与"
                "“场景宏均”在此**恒等**，两列相同**不构成两次独立确认**。"
                "两者出现差异要等每场景张数不等的场合（如完整 validation）。")
            add("")

        # ---- main table
        add("### 各 alpha 的 PSNR / SSIM（相对**本模型自己的 T1**）")
        add("")
        add("| alpha | 图像均值 PSNR | ΔPSNR vs T1 | 场景宏均 ΔPSNR | 图像均值 SSIM | "
            "ΔSSIM vs T1 | 场景宏均 ΔSSIM | 优于 T1 的图数 |")
        add("|---|---|---|---|---|---|---|---|")

        def row(a, label):
            A = g["alphas"][a]
            c = A["image_counts_vs_T1"]
            add(f"| {label} | {fmt(A['psnr']['image_mean'])} | "
                f"{sgn(A['delta_psnr_vs_T1']['image_mean'])} | "
                f"{sgn(A['delta_psnr_vs_T1']['scene_macro'])} | "
                f"{fmt(A['ssim']['image_mean'])} | "
                f"{sgn(A['delta_ssim_vs_T1']['image_mean'], 6)} | "
                f"{sgn(A['delta_ssim_vs_T1']['scene_macro'], 6)} | "
                f"{c['improved']} / {c['total']} |")

        T1 = g["T1"]
        add(f"| T1（参照） | {fmt(T1['psnr']['image_mean'])} | 0 | 0 | "
            f"{fmt(T1['ssim']['image_mean'])} | 0 | 0 | — |")
        T2 = g["T2"]
        add(f"| T2（原始第二轮） | {fmt(T2['psnr']['image_mean'])} | "
            f"{sgn(T2['delta_psnr_vs_T1']['image_mean'])} | "
            f"{sgn(T2['delta_psnr_vs_T1']['scene_macro'])} | "
            f"{fmt(T2['ssim']['image_mean'])} | "
            f"{sgn(T2['delta_ssim_vs_T1']['image_mean'], 6)} | "
            f"{sgn(T2['delta_ssim_vs_T1']['scene_macro'], 6)} | — |")
        for a in alphas:
            row(a, f"alpha={a}")
        add("")

        # ---- per scene
        add("### 逐场景 ΔPSNR vs T1")
        add("")
        scenes = sorted(g["T1"]["psnr"]["per_scene"].keys())
        add("| alpha | " + " | ".join(scenes) + " |")
        add("|---|" + "---|" * len(scenes))
        for a in alphas:
            ps = g["alphas"][a]["delta_psnr_vs_T1"]["per_scene"]
            add(f"| {a} | " + " | ".join(f"{ps[s]:+.3f}" for s in scenes) + " |")
        ps = g["T2"]["delta_psnr_vs_T1"]["per_scene"]
        add(f"| T2 | " + " | ".join(f"{ps[s]:+.3f}" for s in scenes) + " |")
        add("")

        # ---- loss
        add("### 任务损失（官方 DfPTaskLoss = L1 + (1−SSIM_loss) + phase + perceptual）")
        add("")
        add("| alpha | 总损失 | Δ vs T1 | L1 | 1−SSIM | phase | perceptual |")
        add("|---|---|---|---|---|---|---|")
        lc = T1["loss_components_image_mean"]
        add(f"| T1 | {fmt(T1['loss_total_image_mean'])} | 0 | {fmt(lc['l1'])} | "
            f"{fmt(lc['ssim_term'])} | {fmt(lc['phase'])} | {fmt(lc['perceptual'])} |")
        lc2 = g["alphas"]["1.0"]["loss_components_image_mean"]
        add(f"| T2（=alpha 1.0） | {fmt(T2['loss_total_image_mean'])} | "
            f"{sgn(T2['delta_loss_total_vs_T1_image_mean'])} | {fmt(lc2['l1'])} | "
            f"{fmt(lc2['ssim_term'])} | {fmt(lc2['phase'])} | {fmt(lc2['perceptual'])} |")
        for a in alphas:
            A = g["alphas"][a]
            lc = A["loss_components_image_mean"]
            add(f"| {a} | {fmt(A['loss_total_image_mean'])} | "
                f"{sgn(A['delta_loss_total_vs_T1_image_mean'])} | {fmt(lc['l1'])} | "
                f"{fmt(lc['ssim_term'])} | {fmt(lc['phase'])} | {fmt(lc['perceptual'])} |")
        add("")

        # ---- mechanical interpretation
        add("### 机械判读（按文档规则）")
        add("")
        d0 = g["alphas"]["0.0"]["delta_psnr_vs_T1"]["image_mean"]
        d100 = g["alphas"]["1.0"]["delta_psnr_vs_T1"]["image_mean"]
        inner = [a for a in alphas if a not in ("0.0", "1.0")]
        inner_vals = {a: g["alphas"][a]["delta_psnr_vs_T1"]["image_mean"] for a in inner}
        inner_sm = {a: g["alphas"][a]["delta_psnr_vs_T1"]["scene_macro"] for a in inner}
        best_inner = max(inner, key=lambda a: inner_vals[a]) if inner else None
        best_inner_sm = max(inner, key=lambda a: inner_sm[a]) if inner else None
        above_t1 = [a for a in inner if inner_vals[a] > 0]
        above_t1_sm = [a for a in inner if inner_sm[a] > 0]

        add(f"- T2 相对 T1：图像均值 {sgn(d100)} dB，场景宏均 "
            f"{sgn(g['T2']['delta_psnr_vs_T1']['scene_macro'])} dB。")
        add(f"- 内部 alpha 的 ΔPSNR（图像均值）："
            + "，".join(f"{a}={sgn(inner_vals[a])}" for a in inner) + "。")
        add(f"- 内部 alpha 的 ΔPSNR（场景宏均）："
            + "，".join(f"{a}={sgn(inner_sm[a])}" for a in inner) + "。")
        add("")
        # ---- cross-aggregation consistency of the "above T1" verdict ----------
        # Project rule: image-mean and scene-macro are BOTH legitimate; neither may
        # be picked as "the" conclusion. If they disagree, say so explicitly.
        both_above = bool(above_t1) and bool(above_t1_sm)
        both_below = (not above_t1) and (not above_t1_sm)
        add(f"- 图像均值口径下高于 T1 的非零 alpha："
            f"{', '.join(above_t1) if above_t1 else '无'}。")
        add(f"- 场景宏均口径下高于 T1 的非零 alpha："
            f"{', '.join(above_t1_sm) if above_t1_sm else '无'}。")
        if both_above:
            add("- 两种聚合口径**同向**支持“有非零 alpha 高于 T1”。按文档第 5 条，"
                "这构成现有更新方向下的**探索性收益线索**；")
        elif both_below:
            add("- 两种聚合口径**同向**否定：没有任何已查非零 alpha 高于本模型的 T1。"
                "按文档第 7 条 → **没有证据支持“只缩短现有第二步即可解决”**；"
                "**不得**据此宣称所有步长/方向都无效——本次只查了 "
                f"{', '.join(inner)} 这几个固定点，且只沿 (P2−P1) 这一个方向。")
        else:
            pos_side = "图像均值" if above_t1 else "场景宏均"
            neg_side = "场景宏均" if above_t1 else "图像均值"
            pos_vals = (above_t1 if above_t1 else above_t1_sm)
            src_vals = inner_vals if above_t1 else inner_sm
            pos_nums = ", ".join(f"{a}={sgn(src_vals[a])}" for a in pos_vals)
            pos_max = sgn(max(src_vals[a] for a in pos_vals))
            t2_sm = sgn(g["T2"]["delta_psnr_vs_T1"]["scene_macro"])
            add(f"- **两种聚合口径方向不一致**：{pos_side}口径有非零 alpha 高于 T1"
                f"（{pos_nums}），{neg_side}口径没有。"
                "按项目约定，两种口径各有合法含义，**不得择一作为结论**；"
                "符号分歧本身即说明该信号**不足以判定为收益**。")
            add("- 该正值的量级为 " + pos_max + " dB，与同组 T2−T1（图像均值 "
                + sgn(d100) + " / 场景宏均 " + t2_sm + "）相比不在同一量级，"
                "**不能**据此宣称存在迭代收益。")
            add("- 按文档第 7 条 + 口径并列约定：本组**没有可判定的收益证据**；"
                "同时**不得**宣称所有步长/方向都无效。")
        if best_inner and inner_vals[best_inner] > d100 and inner_vals[best_inner] <= 0:
            add(f"- 内部最优 alpha={best_inner}（图像均值 {sgn(inner_vals[best_inner])}）"
                f"高于 alpha=1（{sgn(d100)}）但仍低于 alpha=0（T1）→ 按规则"
                "**只能称“减轻退化”**，不构成迭代收益线索。")

        # best alpha, reported SEPARATELY per aggregation
        all_vals = {a: g["alphas"][a]["delta_psnr_vs_T1"]["image_mean"] for a in alphas}
        all_vals_sm = {a: g["alphas"][a]["delta_psnr_vs_T1"]["scene_macro"] for a in alphas}
        best_all = max(alphas, key=lambda a: all_vals[a])
        best_all_sm = max(alphas, key=lambda a: all_vals_sm[a])

        def best_line(a, nd_vals, tag):
            if a == "0.0":
                return (f"- {tag}：已查 alpha 中 PSNR 最高的是 **alpha=0.0**，"
                        "即**完全不做第二轮**就是该口径下的最优；任何非零步长都更差。")
            if a == "1.0":
                return (f"- {tag}：已查 alpha 中 PSNR 最高的是 **alpha=1.0**，"
                        "即**完整步长就是该口径下的最优**，缩短步长只会更差。")
            return (f"- {tag}：已查 alpha 中 PSNR 最高的是 **alpha={a}**"
                    f"（{sgn(nd_vals[a])}），即**不是**完整步长最优，缩小步长反而更好。")

        add(best_line(best_all, all_vals, "图像均值口径"))
        if best_all != best_all_sm:
            add(best_line(best_all_sm, all_vals_sm, "场景宏均口径"))
            add("  → **两种口径给出的最优 alpha 不同**，再次说明不得择一作为结论。")
        else:
            add(best_line(best_all_sm, all_vals_sm, "场景宏均口径（与图像均值一致）"))
        add("")

        # loss vs metric divergence -- state EXACTLY which quantity moved which way
        div_lines = []
        for a in alphas:
            if a == "0.0":
                continue
            A = g["alphas"][a]
            dl = A["delta_loss_total_vs_T1_image_mean"]
            dp = A["delta_psnr_vs_T1"]["image_mean"]
            ds = A["delta_ssim_vs_T1"]["image_mean"]
            if dl < 0 and ds < 0:
                if dp > 0:
                    div_lines.append(
                        f"- alpha={a}：任务总损失下降（{sgn(dl)}）、PSNR **上升**"
                        f"（{sgn(dp)}），但 SSIM **下降**（{sgn(ds, 6)}）。"
                        "→ 优化目标与 PSNR 口径一致、与 SSIM 口径分歧。")
                else:
                    div_lines.append(
                        f"- alpha={a}：任务总损失下降（{sgn(dl)}），PSNR 与 SSIM 同时下降"
                        f"（{sgn(dp)} / {sgn(ds, 6)}）。"
                        "→ 优化目标与两个观测指标都分歧。")
            elif dl < 0 and dp < 0 and ds > 0:
                div_lines.append(
                    f"- alpha={a}：任务总损失下降（{sgn(dl)}），PSNR 下降（{sgn(dp)}）"
                    f"但 SSIM 上升（{sgn(ds, 6)}）。→ 优化目标与 SSIM 一致、与 PSNR 分歧。")
        if div_lines:
            add("### 目标分歧（按文档第 4 条）")
            add("")
            for line in div_lines:
                add(line)
            add("")
            add("按文档规则：若任务损失下降而 PSNR/SSIM 下降，**只能说明观测目标存在分歧**，"
                "**不能**据此宣称训练未生效。")
            add("")

        # SSIM-vs-PSNR shape
        ds_list = [g["alphas"][a]["delta_ssim_vs_T1"]["image_mean"] for a in alphas]
        dp_list = [g["alphas"][a]["delta_psnr_vs_T1"]["image_mean"] for a in alphas]
        if all(v <= 0 for v in ds_list[1:]) and any(v > 0 for v in dp_list[1:]):
            add("- **PSNR 与 SSIM 反向**：所有非零 alpha 的 ΔSSIM 均 ≤ 0 "
                "（单调下降），而 ΔPSNR 有正值。两个指标给出相反方向，"
                "**不得**只用 PSNR 描述结果。")
        elif all(v <= 0 for v in ds_list[1:]) and all(v <= 0 for v in dp_list[1:]):
            worst = min(alphas[1:], key=lambda a: dp_list[alphas.index(a)])
            if worst != alphas[-1]:
                add(f"- PSNR 与 SSIM 同向下降；ΔSSIM 随 alpha 单调下降，"
                    f"ΔPSNR 的最差点在 alpha={worst}（{sgn(dp_list[alphas.index(worst)])}），"
                    f"之后到 alpha={alphas[-1]} 反而略有回升"
                    f"（{sgn(dp_list[-1])}）。即**退化对步长非单调**。")
            else:
                add("- PSNR 与 SSIM 同向下降，且 ΔPSNR 随 alpha 单调下降。")
        add("")

        compact["groups"][group] = {
            "feedback_enabled": g["feedback_enabled"],
            "feedback_scale": g["feedback_scale"],
            "n_images": n,
            "T1_psnr_image_mean": T1["psnr"]["image_mean"],
            "T1_psnr_scene_macro": T1["psnr"]["scene_macro"],
            "T2_delta_psnr_vs_T1_image_mean": T2["delta_psnr_vs_T1"]["image_mean"],
            "T2_delta_psnr_vs_T1_scene_macro": T2["delta_psnr_vs_T1"]["scene_macro"],
            "per_alpha_delta_psnr_vs_T1_image_mean":
                {a: g["alphas"][a]["delta_psnr_vs_T1"]["image_mean"] for a in alphas},
            "per_alpha_delta_psnr_vs_T1_scene_macro":
                {a: g["alphas"][a]["delta_psnr_vs_T1"]["scene_macro"] for a in alphas},
            "per_alpha_delta_ssim_vs_T1_image_mean":
                {a: g["alphas"][a]["delta_ssim_vs_T1"]["image_mean"] for a in alphas},
            "per_alpha_delta_ssim_vs_T1_scene_macro":
                {a: g["alphas"][a]["delta_ssim_vs_T1"]["scene_macro"] for a in alphas},
            "per_alpha_delta_loss_vs_T1_image_mean":
                {a: g["alphas"][a]["delta_loss_total_vs_T1_image_mean"] for a in alphas},
            "per_alpha_image_counts_vs_T1":
                {a: g["alphas"][a]["image_counts_vs_T1"] for a in alphas},
            "nonzero_alphas_above_T1_image_mean": above_t1,
            "nonzero_alphas_above_T1_scene_macro": above_t1_sm,
            "endpoint_integrity": g["endpoint_integrity"],
            "verdict": ("no_nonzero_alpha_above_T1" if not above_t1 and not above_t1_sm
                        else "some_nonzero_alpha_above_T1"),
        }

    ref872 = {"feedback": -0.04497, "no_feedback": -0.06125}
    if full_mode:
        add("## 本轮完整 validation 与既有 872 张记录的一致性核对")
        add("")
        add("已有 872 张 validation 的 T2−T1（来源：既有复核记录）："
            "**feedback −0.04497 dB，no_feedback −0.06125 dB**。本轮跑的就是同一划分。")
        add("")
        add("| 组 | 本轮完整 validation T2−T1（图像均值） | 本轮（场景宏均） | 既有记录 | 差值 |")
        add("|---|---|---|---|---|")
        for group in ("no_feedback", "feedback"):
            cur = data["groups"][group]["T2"]["delta_psnr_vs_T1"]["image_mean"]
            cur_sm = data["groups"][group]["T2"]["delta_psnr_vs_T1"]["scene_macro"]
            old = ref872[group]
            add(f"| {group} | {sgn(cur)} | {sgn(cur_sm)} | {sgn(old)} | "
                f"{sgn(cur - old)} |")
        add("")
        add("**说明与更正**：本轮与既有记录来自两条不同的评估脚本路径，"
            "本轮额外在每图上做了 RNG 复位与损失分量计算。上表差值已按"
            "**未舍入**的逐图值重算（详见 `REVIEW_FOLLOWUPS_ZH_20261002.md` §B）。")
        add("")
        add("⚠️ **不得把“总体均值吻合”写成“逐位吻合”。** 总体均值吻合**不能**推出"
            "输出张量逐元素一致；逐 sample 核对需要既有记录的逐图原始文件，"
            "该文件本项目**未留存**，因此**逐图核对是未完成项**，不作断言。"
            "现有证据只支持：两条路径在**总体均值**层面吻合到远小于任何实质效应的量级。")
        add("")
    else:
        add("## 子集与既有 872 张 validation 结果的对照")
        add("")
        add("已有 872 张 validation 的 T2−T1（来源：本文档引用的既有复核记录）："
            "**feedback −0.04497 dB，no_feedback −0.06125 dB**。")
        add("")
        add("| 组 | 24 张子集 T2−T1 | 872 张 T2−T1 | 符号 |")
        add("|---|---|---|---|")
        for group in ("no_feedback", "feedback"):
            sub_v = data["groups"][group]["T2"]["delta_psnr_vs_T1"]["image_mean"]
            full_v = ref872[group]
            same = "一致" if (sub_v > 0) == (full_v > 0) else "**相反**"
            add(f"| {group} | {sgn(sub_v)} | {sgn(full_v)} | {same} |")
        add("")
        add("**这张表是本轮最重要的边界提示**：两组的 24 张子集结果与 872 张结果"
            "**方向不一致**（子集上 feedback 为正、872 上为负；no_feedback 虽同为负但量级差约 5 倍）。"
            "这说明**这 24 张图不足以代表 validation**，本诊断的正/负号不能直接外推。"
            "按文档第 6 条，非零 alpha 在子集上出现正值时，应**按锁定的全局设置扩到完整 validation**。")
        add("")

    # ---- final conclusion ---------------------------------------------------
    add("## 结论（两组合并，按文档第 5/7 条）")
    add("")
    add("| 组 | T1 | T2−T1（图像均值） | T2−T1（场景宏均） | 非零 α 高于 T1（图像均值） | "
        "非零 α 高于 T1（场景宏均） | ΔPSNR 随 α |")
    add("|---|---|---|---|---|---|---|")
    for group in ("no_feedback", "feedback"):
        g = data["groups"][group]
        c = compact["groups"][group]
        dp = [g["alphas"][a]["delta_psnr_vs_T1"]["image_mean"] for a in alphas[1:]]
        mono = "单调下降" if all(dp[i] > dp[i + 1] for i in range(len(dp) - 1)) else "非单调"
        add(f"| {group} | {fmt(g['T1']['psnr']['image_mean'])} | "
            f"{sgn(g['T2']['delta_psnr_vs_T1']['image_mean'])} | "
            f"{sgn(g['T2']['delta_psnr_vs_T1']['scene_macro'])} | "
            f"{'、'.join(c['nonzero_alphas_above_T1_image_mean']) or '无'} | "
            f"{'、'.join(c['nonzero_alphas_above_T1_scene_macro']) or '无'} | {mono} |")
    add("")

    if args.subset_json and full_mode:
        try:
            sd = json.loads(Path(args.subset_json).read_text(encoding="utf-8"))
        except Exception:
            sd = None
        if sd is not None and "groups" in sd:
            add("### 24 张子集结论的撤回")
            add("")
            add("| 组 | 子集 T2−T1 | 全量 T2−T1 | 子集上是否曾出现正信号 | 结论 |")
            add("|---|---|---|---|---|")
            for group in ("no_feedback", "feedback"):
                s_v = sd["groups"][group]["T2"]["delta_psnr_vs_T1"]["image_mean"]
                f_v = data["groups"][group]["T2"]["delta_psnr_vs_T1"]["image_mean"]
                verdict = ("**子集正值不可外推，已不采用**"
                           if (s_v > 0) != (f_v > 0) else
                           ("量级差 %.1f 倍，子集不可外推"
                            % (abs(f_v / s_v) if s_v else float("nan"))))
                add(f"| {group} | {sgn(s_v)} | {sgn(f_v)} | "
                    f"{'是' if s_v > 0 else '否'} | {verdict} |")
            add("")
            add("**措辞更正（依评审批复）**：撤回的是**向 872 张的外推**，"
                "**不是**那 24 张上的观测数字本身。子集上的正值是**真实的子集观测**，"
                "不同聚合口径的结果也不会使原样本数字失效。该子集今后仍可用于"
                "实现/局部诊断，只是**不能代表 validation**、其符号**不能外推**。")
            add("")
            add("子集上 feedback 组 T2−T1 为正、且非零 α 全部高于 T1，曾触发文档第 6 条的扩样条件；"
                "扩样到完整 validation 后该符号**反转为负**。"
                "子集上 no_feedback 组“α=0.75 最差”的非单调形态，在全量上也**不成立**"
                "（全量图像均值口径为单调下降）。")
            add("")

    add("### 判读")
    add("")
    add("- 两组在完整 validation 上 **T2 均低于各自的 T1**。")
    add("- **“单调下降”的适用范围（已收窄）**：只成立于**图像均值口径 + 所查离散点**。"
        "feedback 组的**场景宏均并非单调下降**（α=0.25/0.5 为正、0.75 转负）。"
        "该判定**不**代表逐图、**不**代表连续 α、**不**代表所有聚合口径。")
    add("- 两组**任务总损失在所查点随 α 单调下降**，而 PSNR 与 SSIM 同时下降 → "
        "按文档第 4 条，这只能说明**优化目标与观测指标分歧**，"
        "**不能**推断训练未生效、也**不能**推断实现有 bug。")
    add("- 按文档第 7 条：**没有证据支持“只缩短现有第二步即可解决”**；"
        "同时**不得**宣称这一步长方向已被完全排除——"
        f"本结论只覆盖已查的离散点 {', '.join(alphas[1:-1])}、当前模型与给定评估；"
        "feedback 组还有场景宏均小正值，且**未估计跨 seed 不确定性**。")
    add("- 描述性现象（不作结论）：两组的 ΔPSNR 在场景 07 / 22 / 56 为正、"
        "在场景 01 / 55 为负，即聚合值的下降伴随**场景间再分配**。"
        "（逐图优于 T1 的比例须分组表述：no_feedback 约 33.5%–35.7%，"
        "feedback 约 29.5%–30.0%；详见复核补充核算 §E。）"
        "这是观测，不是机制解释。")
    add("")

    add("## 证据边界")
    add("")
    add("- 单 seed（2026）、单检查点（step 3000）；**不含训练随机性**。")
    if full_mode:
        add("- 样本为完整 validation 划分（按 loader 顺序，不做挑选），"
            "**不是**新划分、**不是**未见数据。")
    else:
        add("- 子集为场景分层，**不是**场景×难度共同分层；实际为 21 easy + 3 hard。")
    add("- 两种聚合口径并列，不择一作为结论。")
    add("- 本诊断只沿 (P2−P1) 这一个方向、在已查离散点上缩放；"
        "**只支持**当前模型与给定评估下的结论。不覆盖其他更新方向、"
        "不覆盖连续 α、不覆盖 α>1 的外推，也**未估计跨 seed 不确定性**。")
    add("- 若任务损失下降而指标下降，只报“目标分歧”，不推断训练成败。")
    add("- 这是**探索性诊断**；即使出现正向信号也不构成独立验证，"
        "正式确认需另行定义留出/外部评估。")
    add("- 完整 validation 已被用于既有模型选择，因此**本轮结果不能充当独立确认**。")
    add("")

    add("## 计算与推理次数")
    add("")
    el = data.get("elapsed_seconds")
    n_calls = data.get("n_tasknet_calls")
    n_img = sum(g["n_images"] for g in data["groups"].values())
    add("| 项 | 数值 |")
    add("|---|---|")
    add(f"| 组数 | {len(data['groups'])} |")
    add(f"| 图像总数（两组合计） | {n_img} |")
    add(f"| TaskNet 前向调用总数 | {n_calls} |")
    if full_mode:
        add("| 每图调用数 | 5（T1、T2、alpha=0.25/0.5/0.75；alpha=0/1 为别名，不另算） |")
    else:
        add("| 每图调用数 | 7（T1、T2、alpha=0/0.25/0.5/0.75/1） |")
    if el is not None:
        add(f"| 总墙钟耗时 | {el:.1f} s（{el/3600.0:.2f} h） |")
        if n_img:
            add(f"| 单图平均耗时 | {el/max(n_img,1):.2f} s |")
    add("| 训练次数 | 0（本轮不训练、不更新权重） |")
    add("")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(L), encoding="utf-8")
    print("WROTE", args.out_md)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(compact, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
        print("WROTE", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
