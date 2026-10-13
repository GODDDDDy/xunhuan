#!/usr/bin/env python3
# Follow-ups requested by ALPHA_SWEEP_REVIEW_ZH_20261002.md.
#
# READ-ONLY, CPU-ONLY. No GPU forward, no backward, no training, no checkpoint
# modification. Operates purely on the downloaded full-validation result JSON.
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

COMPONENTS = ("l1", "ssim_term", "phase", "perceptual")
LQ = "\u201c"
RQ = "\u201d"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def q(text):
    return LQ + text + RQ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="alpha_sweep_full_result.json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    d = json.loads(Path(args.src).read_text(encoding="utf-8"))
    alphas = [str(a) for a in d["alphas"]]
    out: dict = {"source": args.src, "alphas": alphas}

    L: list[str] = []
    add = L.append
    add("# 复核意见的执行结果（补充核算）")
    add("")
    add("本文件是对 `ALPHA_SWEEP_REVIEW_ZH_20261002.md` 各项意见的**逐条执行**。"
        "**只读、纯 CPU**：不新增 GPU 前向/反传，不训练，不修改检查点；"
        "全部数字从既有 `alpha_sweep_full_result.json`（10.3 MB，含逐图分项）重算。")
    add("")
    add("> 原始报告与原始结果**全部保留**，本文件是补充，不覆盖任何既有产物。")
    add("")

    # ---------------------------------------------------------------- A
    add("## A. 逐场景 / 逐难度的损失分项差值（T2−T1，不新增前向）")
    add("")
    add("正值 = 该项变差。分项之和 = 总损失变化（代数分解，非因果归因）。")
    add("")

    scene_tables: dict = {}
    for group in ("no_feedback", "feedback"):
        g = d["groups"][group]
        rows = g["per_image"]
        by_scene = defaultdict(list)
        by_diff = defaultdict(list)
        for r in rows:
            dl = {c: r["T2"]["loss"][c] - r["T1"]["loss"][c] for c in COMPONENTS}
            dl["total"] = sum(dl.values())
            by_scene[r["scene"]].append(dl)
            by_diff[r["difficulty"]].append(dl)

        scene_tables[group] = {}
        for label, buckets in (("per_scene", by_scene), ("per_difficulty", by_diff)):
            res = {}
            for key in sorted(buckets):
                items = buckets[key]
                entry = {c: mean(x[c] for x in items) for c in COMPONENTS}
                entry["total"] = mean(x["total"] for x in items)
                entry["n"] = len(items)
                res[key] = entry
            scene_tables[group][label] = res

        for title, buckets in ((chr(0x6309) + "场景", by_scene),
                               (chr(0x6309) + "难度", by_diff)):
            add("### " + group + " — " + title)
            add("")
            add("| 分组 | n | L1 | SSIM项 | phase | perceptual | 总计 |")
            add("|---|---|---|---|---|---|---|")
            for key in sorted(buckets):
                items = buckets[key]
                vals = {c: mean(x[c] for x in items) for c in COMPONENTS}
                tot = mean(x["total"] for x in items)
                add("| " + str(key) + " | " + str(len(items)) + " | "
                    + "%.9f" % vals["l1"] + " | " + "%.9f" % vals["ssim_term"] + " | "
                    + "%.9f" % vals["phase"] + " | " + "%.9f" % vals["perceptual"]
                    + " | " + "%.9f" % tot + " |")
            add("")
        add("（" + group + "：n=" + str(len(rows)) + "）")
        add("")

    out["loss_component_deltas"] = scene_tables

    # ---------------------------------------------------------------- B
    add("## B. T2−T1 的**未舍入**精确值与既有记录核对")
    add("")
    add("完整报告中的 −0.0612 / −0.0450 是 4 位小数展示值。下表给出未舍入值。")
    add("")
    prior = {"no_feedback": -0.06125, "feedback": -0.04497}
    add("| 组 | 本轮（未舍入，图像均值） | 本轮（未舍入，场景宏均） | 既有记录 | 绝对差 |")
    add("|---|---|---|---|---|")
    exact = {}
    for group in ("no_feedback", "feedback"):
        g = d["groups"][group]
        raw_rows = g["per_image"]
        im = mean(r["T2"]["delta_psnr_vs_T1"] for r in raw_rows)
        per_scene = defaultdict(list)
        for r in raw_rows:
            per_scene[r["scene"]].append(r["T2"]["delta_psnr_vs_T1"])
        sm = mean(mean(v) for v in per_scene.values())
        old = prior[group]
        exact[group] = {"image_mean_unrounded": im, "scene_macro_unrounded": sm,
                        "prior_872_record": old, "abs_diff_vs_prior": abs(im - old)}
        add("| " + group + " | " + ("%.12f" % im) + " | " + ("%.12f" % sm) + " | "
            + ("%.5f" % old) + " | " + ("%.3e" % abs(im - old)) + " |")
    add("")
    add("**复核意见要点（已采纳）**：总体均值吻合**不能**推出" + q("逐位吻合")
        + "，更不能推出输出张量逐元素一致。上表只说明两条评估路径在**总体均值**"
        "层面吻合到远小于任何实质效应的量级。逐 sample 核对需要既有记录的"
        "逐图原始文件（本项目当前**未留存**该逐图文件），因此**逐图核对列为"
        "未完成项**，不作断言。")
    add("")
    out["exact_deltas"] = {"per_group": exact, "prior_872_record": prior,
                           "per_sample_verification": "NOT_DONE_no_prior_per_image_file"}

    # ---------------------------------------------------------------- C
    add("## C. 反事实记账：phase 项改变权重后的 T2−T1 总损失")
    add("")
    add("做法：保持**已有输出张量不变**（不重新推理），按链式法则精确缩放 phase 分项，"
        "因为 `phase` 在 criterion 中以系数 1 线性相加。")
    add("")
    add("`总损失(w) = L1 + (1-SSIM) + w*phase + perceptual`")
    add("")
    add("> 这是**反事实记账**：只说明" + q("当前总损失的排序依赖权重取值")
        + "，**不预测**换损失重训后的结果，也**不证明**任何权重最优。")
    add("")
    cf_weights = [1.0, 0.5, 0.1, 0.0]
    add("| 组 | w=1.0（现行） | w=0.5 | w=0.1 | w=0.0 |")
    add("|---|---|---|---|---|")
    cf: dict = {}
    for group in ("no_feedback", "feedback"):
        g = d["groups"][group]
        row_vals = {}
        for w in cf_weights:
            vals = []
            for r in g["per_image"]:
                t2 = r["T2"]["loss"]
                t1 = r["T1"]["loss"]
                vals.append((t2["l1"] + t2["ssim_term"] + w * t2["phase"]
                             + t2["perceptual"])
                            - (t1["l1"] + t1["ssim_term"] + w * t1["phase"]
                               + t1["perceptual"]))
            row_vals[w] = mean(vals)
        cf[group] = row_vals
        add("| " + group + " | " + ("%.9f" % row_vals[1.0]) + " | "
            + ("%.9f" % row_vals[0.5]) + " | " + ("%.9f" % row_vals[0.1]) + " | "
            + ("%.9f" % row_vals[0.0]) + " |")
    add("")
    add("读法：w 从 1 降到 0.1 时，两组的 T2−T1 总损失由负转正——即在此反事实记账下"
        "T2 反而更差。**这只说明现行总损失的" + q("T2 更优")
        + "结论依赖 phase 项的权重**，不构成对任何替代权重的推荐。")
    add("")
    out["counterfactual_phase_weight"] = {"weights": cf_weights, "per_group": cf}

    # ---------------------------------------------------------------- D
    add("## D. phase 标量在总标量中的占比")
    add("")
    add("| 组 | 轮次 | phase | 总损失 | phase 占比 |")
    add("|---|---|---|---|---|")
    share: dict = {}
    for group in ("no_feedback", "feedback"):
        g = d["groups"][group]
        for tag, key, lc in (
                ("第一轮 T1", "T1", g["T1"]["loss_components_image_mean"]),
                ("第二轮 T2", "T2", g["alphas"]["1.0"]["loss_components_image_mean"])):
            tot = g[key]["loss_total_image_mean"]
            r = lc["phase"] / tot
            share[group + "_" + key] = r
            add("| " + group + " | " + tag + " | " + ("%.6f" % lc["phase"]) + " | "
                + ("%.6f" % tot) + " | " + ("%.2f" % (r * 100)) + "% |")
    add("")
    add("**复核意见要点（已采纳）**：该占比**不证明** phase 的梯度占比，也**不证明**"
        "实际参数更新占比。**不得**写成" + q("phase 主导训练已经证实")
        + "。要判断梯度是否冲突，需要 §G 的梯度诊断。")
    add("")
    out["phase_share_of_scalar"] = share

    # ---------------------------------------------------------------- E
    add("## E. 优于/劣于 T1 的比例（全精度，分组分 alpha）")
    add("")
    add("| 组 | alpha | 优于 T1 | 劣于 T1 | 不变 | 优于比例 | 劣于比例 |")
    add("|---|---|---|---|---|---|---|")
    frac: dict = {}
    for group in ("no_feedback", "feedback"):
        g = d["groups"][group]
        frac[group] = {}
        for a in alphas:
            vals = [r["alphas"][a]["delta_psnr_vs_T1"] for r in g["per_image"]]
            n_imp = sum(1 for v in vals if v > 0)
            n_deg = sum(1 for v in vals if v < 0)
            n_tie = len(vals) - n_imp - n_deg
            frac[group][a] = {"improved": n_imp, "degraded": n_deg, "unchanged": n_tie,
                              "improved_fraction": n_imp / len(vals),
                              "degraded_fraction": n_deg / len(vals)}
            add("| " + group + " | " + a + " | " + str(n_imp) + " | " + str(n_deg)
                + " | " + str(n_tie) + " | " + ("%.2f" % (n_imp / len(vals) * 100))
                + "% | " + ("%.2f" % (n_deg / len(vals) * 100)) + "% |")
    add("")
    nf = [frac["no_feedback"][a]["improved_fraction"] for a in alphas[1:]]
    fb = [frac["feedback"][a]["improved_fraction"] for a in alphas[1:]]
    add("**复核意见要点（已采纳）**：粗说法" + q("约 30%")
        + "应分组表述——no_feedback 为 " + ("%.1f%%-%.1f%%" % (min(nf) * 100, max(nf) * 100))
        + "，feedback 为 " + ("%.1f%%-%.1f%%" % (min(fb) * 100, max(fb) * 100))
        + "（非零 alpha）。")
    add("")
    out["improved_fractions"] = frac

    # ---------------------------------------------------------------- F
    add("## F. 单调性审计（按聚合口径分别判定）")
    add("")
    add("| 组 | 口径 | α=0.25 | 0.5 | 0.75 | 1.0 | 非零区间单调下降 |")
    add("|---|---|---|---|---|---|---|")
    mono: dict = {}
    for group in ("no_feedback", "feedback"):
        g = d["groups"][group]
        mono[group] = {}
        for agg_name in ("image_mean", "scene_macro"):
            vals = [g["alphas"][a]["delta_psnr_vs_T1"][agg_name] for a in alphas[1:]]
            is_mono = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
            mono[group][agg_name] = {"values": vals, "monotone_decreasing": is_mono}
            add("| " + group + " | " + agg_name + " | "
                + " | ".join("%+.6f" % v for v in vals) + " | "
                + ("是" if is_mono else "**否**") + " |")
    add("")
    add("**复核意见要点（已采纳）**：单调下降只能限定在**图像均值口径 + 所查离散点**上。"
        "feedback 组的**场景宏均不是单调下降**（α=0.25/0.5 为正、0.75 转负）。"
        "该判定**不**代表逐图、**不**代表连续 alpha、**不**代表所有聚合。")
    add("")
    out["monotonicity"] = mono

    # ---------------------------------------------------------------- G
    add("## G. 仍待执行的两类工作（本次未做）")
    add("")
    add("按复核意见第 2-4 条，以下**尚未执行**，需要单独授权：")
    add("")
    add("1. **训练损失源码只读核对**：确认实际训练所用任务损失的系数、FFT 输入与维度、"
        "phase 归约与周期处理；比较公开仓库现版本与历史运行依赖版本。"
        "已查的公开来源：PolarFree `ours_stage2.yml` 为 phase_weight=0.1、VGG_weight=0.02；"
        "PolarAPP `losses.py` 中 phase 以系数 1 相加、含 SSIM 项、perceptual_weight=0.1。"
        "⇒ 本项目的任务损失应称" + q("当前采用的 PolarAPP 适配任务损失")
        + "，**不是**完整的 PolarFree 原训练目标复现。")
    add("2. **梯度诊断**（少量预定训练批次、冻结检查点、**不执行 optimizer.step**）："
        "分别计算各分项对共享 R2P 参数的梯度范数与夹角，以及第一轮 vs 第二轮任务梯度；"
        "区分加权与裁剪前后；保留完整计算图与固定随机流。"
        "注意：现有量化预处理含 STE，梯度是**实现采用的代理梯度**。")
    add("")
    add("这两类都**还没有做**，因此不能据此判定" + q("phase 是唯一原因")
        + "，也不能认为替换损失必定解决问题。")
    add("")

    out["pending"] = ["training_loss_source_audit",
                      "gradient_diagnostics_on_predetermined_batches"]

    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
    Path(args.out_md).write_text("\n".join(L), encoding="utf-8")
    print("WROTE", args.out_md)
    print("WROTE", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
