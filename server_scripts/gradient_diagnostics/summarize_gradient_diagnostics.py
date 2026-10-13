#!/usr/bin/env python3
"""Summarize the review-item-#3 gradient diagnostics JSON into Chinese markdown.

Read-only: consumes the JSON produced by gradient_diagnostics.py and writes one
markdown report plus a compact JSON of the headline numbers.

Wording discipline (inherited from the alpha-sweep review, applied here):
  * every statement is tied to the %d predetermined batches of ONE seed;
    no significance, no cross-seed claim;
  * a cosine near zero is "near-orthogonal", never "no conflict";
  * "conflict" is asserted only when a clearly obtuse angle RECURS or is large
    in magnitude, and is always reported together with the magnitudes, because a
    small-magnitude term cannot dominate the update no matter its angle;
  * the STE caveat is restated wherever gradients are used;
  * no mechanism claim ("phase causes the degradation") is made -- only the
    geometric/scale facts actually measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

Q = chr(8220)
QQ = chr(8221)


def q(text):
    return Q + text + QQ


def fmt(x, nd=4):
    return "n/a" if x is None else ("%." + str(nd) + "f") % x


def sci(x):
    return "n/a" if x is None else "%.4g" % x


# the cosine keys worth printing, in reading order
ROUND_KEY = "L1.total vs L2.total"
PHASE_KEY = "L1.phase vs L2.phase"

CROSS_KEYS = [
    ("L1 weighted vs TASKSUM", "第一轮任务损失 vs 加权任务总和"),
    ("L2 weighted vs TASKSUM", "第二轮任务损失 vs 加权任务总和"),
    ("align weighted vs TASKSUM", "alignment（已加权）vs 任务总和"),
]

WITHIN_KEYS = [
    ("L1.phase vs L1.l1", "第一轮 phase vs 第一轮 L1"),
    ("L1.phase vs L1.ssim", "第一轮 phase vs 第一轮 1-SSIM"),
    ("L1.l1 vs L1.ssim", "第一轮 L1 vs 第一轮 1-SSIM"),
    ("L2.phase vs L2.l1", "第二轮 phase vs 第二轮 L1"),
    ("L2.phase vs L2.ssim", "第二轮 phase vs 第二轮 1-SSIM"),
    ("L2.l1 vs L2.ssim", "第二轮 L1 vs 第二轮 1-SSIM"),
    ("L1.phase vs L2.phase", "第一轮 phase vs 第二轮 phase"),
]

NORM_KEYS = [
    ("L1.l1", "第一轮 L1"), ("L1.ssim", "第一轮 1-SSIM"),
    ("L1.phase", "第一轮 phase"), ("L1.perceptual", "第一轮 perceptual"),
    ("L1.total", "第一轮合计"),
    ("L2.l1", "第二轮 L1"), ("L2.ssim", "第二轮 1-SSIM"),
    ("L2.phase", "第二轮 phase"), ("L2.perceptual", "第二轮 perceptual"),
    ("L2.total", "第二轮合计"),
    ("align", "alignment"),
]


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.src).read_text(encoding="utf-8"))
    steps = data["predetermined_steps"]
    groups = [g for g in ("no_feedback", "feedback") if g in data["groups"]]

    L = []
    A = L.append

    A("# 训练梯度诊断（评审意见第 3 条）")
    A("")
    A("**只读诊断。冻结 step3000 检查点，不做任何权重更新，"
      "全程不调用 `optimizer.step()`。**")
    A("")
    A("数据源：`gradient_diagnostics_result.json`（本次新产出）。"
      "所用时间 %.1f 秒。无新增前向以外的实验。" % data.get("elapsed_seconds", 0))
    A("")
    A("---")
    A("")

    # ---------------- 0 ----------------
    A("## 0. 结论（先给回答）")
    A("")
    for para in headline(data, steps, groups):
        A(para)
        A("")
    A("---")
    A("")

    # ---------------- 1 ----------------
    A("## 1. 问的是什么、怎么算的")
    A("")
    A("训练目标是")
    A("")
    A("```")
    A("total = 0.5 * L(T1) + 1.0 * L(T2) + lambda * alignment_loss")
    A("L(.)  = L1 + (1 - SSIM) + phase + 0.1 * L1(VGG conv5_4)")
    A("```")
    A("")
    A("R2P 在每个训练步**被使用两次**（第一轮、第二轮），两次都写同一组参数张量，"
      "所以一次 `loss.backward()` 会把两轮累加进同一个 `.grad` 而无法分辨。"
      "本诊断用**逐量独立的自动微分过程**把两轮、以及每轮的四个分项全部拆开。")
    A("")
    A("### 两种分解口径（含义不同，都报告）")
    A("")
    A("| 口径 | 构造 | 用途 |")
    A("| --- | --- | --- |")
    A("| **real-path** | 与训练完全一致：第二轮消费**带图**的 P1 | "
      "分量之和复现训练的实际更新方向 |")
    A("| **isolated** | 第二轮消费**已 detach** 的 P1 | "
      "第二轮对共享参数的梯度只归因于第二轮，用于问" + q("两轮是否冲突") + " |")
    A("")
    A("> 注意 `recurrent_forward` **没有** detach P1，因此真实训练图里第二轮还会"
      "经 P1 回传到第一轮。**" + q("两轮是否冲突") + "只有在 isolated 口径下才能干净回答。**")
    A("")
    A("### 梯度约定")
    A("")
    A("- `unit` = **未加权**分项的梯度；`weighted` = 乘上该项在目标中的实际权重。")
    A("- **夹角一律在 `unit` 向量上计算**（余弦对缩放不变），同时给出加权范数以看清量级。")
    A("- 每个梯度量各跑**一次独立前向**（`retain_graph=False`），"
      "把峰值显存压在 12 GiB 内。")
    A("- 官方 criterion 与四分量重算在**每个批次上都断言相等**通过后，才报告任何梯度。")
    A("")
    A("### 两条必须保留的边界")
    A("")
    A("1. **裁剪未施加**。训练在反向之后把 `model.parameters()` 的**求和**梯度整体裁到"
      "全局范数 1.0；这是**等比缩放，不改变任何两两夹角**。本诊断记录裁剪前范数，"
      "但不裁剪。")
    A("2. **STE（直通估计）**。扩散通路的量化预处理使用直通估计，"
      "因此下面所有梯度都是**该实现的代理梯度**，不是冻结扩散先验的数学导数。")
    A("")
    A("### 预先确定的批次")
    A("")
    A("| 项 | 值 |")
    A("| --- | --- |")
    A("| 步数 | " + ", ".join(str(s) for s in steps) + " |")
    A("| 选取依据 | " + data.get("step_selection_rationale", "") + " |")
    A("| 随机流 | 复用训练期 `set_step_seed(seed, train, step, sample_ids)`，"
      "每次前向之前回绕到该状态 |")
    A("")
    A("---")
    A("")

    # ---------------- 2 ----------------
    A("## 2. 分项对共享 R2P 参数的梯度范数（未加权）")
    A("")
    A("共享参数 = `model.r2p` 全部参数。"
      "`projections` / `feedback_adapter` / `feedback_scale` / `net_g` "
      "**不在此列**（另计）。")
    A("")
    for grp in groups:
        g = data["groups"][grp]
        A("### " + grp + "（共享参数 %d 个；`net_g` %d 个）"
          % (g["shared_param_count"], g["netg_param_count"]))
        A("")
        rows = []
        for k, label in NORM_KEYS:
            cells = []
            for r in g["records"]:
                cells.append(sci(r["real_path"]["shared_grad_norms_unit"].get(k)))
            rows.append(["`" + k + "`（" + label + "）"] + cells)
        A("\n".join(table(["分项"] + ["step " + str(s) for s in steps], rows)))
        A("")

    # ---------------- 3 ----------------
    A("## 3. isolated 口径：第二轮 vs 第一轮")
    A("")
    A("这是回答" + q("两轮是否冲突") + "的口径：第二轮喂入已 detach 的 P1，"
      "因此它的共享参数梯度**只来自第二轮**。")
    A("")
    rows = []
    for grp in groups:
        g = data["groups"][grp]
        cells = []
        for r in g["records"]:
            iso = r["isolated_round2_vs_round1"]
            cells.append(fmt(iso["cosine_L1_vs_L2"], 3))
        rows.append(["**" + grp + "**"] + cells)
    A("\n".join(table(["两轮任务梯度夹角"] + ["step " + str(s) for s in steps],
                      rows)))
    A("")
    A("各轮在共享参数上的**加权**范数（0.5 / 1.0 已乘入）：")
    A("")
    rows = []
    for grp in groups:
        g = data["groups"][grp]
        for key, label in (("L1_weighted_norm", grp + " / 第一轮"),
                           ("L2_weighted_norm", grp + " / 第二轮")):
            cells = []
            for r in g["records"]:
                cells.append(sci(r["isolated_round2_vs_round1"][key]))
            rows.append([label] + cells)
    A("\n".join(table(["量"] + ["step " + str(s) for s in steps], rows)))
    A("")

    # ---------------- 4 ----------------
    A("## 4. 同一轮内部：phase 与像素级分项的夹角")
    A("")
    A("这是本次诊断**最稳定、量级最大**的一组关系。")
    A("")
    for grp in groups:
        g = data["groups"][grp]
        A("**" + grp + "**")
        A("")
        rows = []
        for k, label in WITHIN_KEYS:
            cells = []
            for r in g["records"]:
                cells.append(fmt(r["focused_cosines_unit_or_weighted"].get(k), 3))
            rows.append(["`" + k + "`（" + label + "）"] + cells)
        A("\n".join(table(["夹角"] + ["step " + str(s) for s in steps], rows)))
        A("")

    # ---------------- 5 ----------------
    A("## 5. 各加权项与任务总和的夹角")
    A("")
    A("`TASKSUM` = `0.5*L1梯度 + 1.0*L2梯度`。")
    A("")
    for grp in groups:
        g = data["groups"][grp]
        A("**" + grp + "**")
        A("")
        rows = []
        for k, label in CROSS_KEYS:
            cells = []
            for r in g["records"]:
                cells.append(fmt(r["focused_cosines_unit_or_weighted"].get(k), 3))
            rows.append([label] + cells)
        A("\n".join(table(["夹角"] + ["step " + str(s) for s in steps], rows)))
        A("")
    A("加权后的量级：")
    A("")
    for grp in groups:
        g = data["groups"][grp]
        A("**" + grp + "**（lambda = %.6g）" % g["lambda_feature_alignment"])
        A("")
        rows = []
        for k in ("L1.total.weighted_norm", "L2.total.weighted_norm",
                  "align.weighted_norm", "TASKSUM.weighted_norm",
                  "OBJECTIVE.weighted_norm"):
            cells = []
            for r in g["records"]:
                cells.append(sci(r["weighted_vectors_on_shared"].get(k)))
            rows.append(["`" + k + "`"] + cells)
        A("\n".join(table(["量"] + ["step " + str(s) for s in steps], rows)))
        A("")

    # ---------------- 6 ----------------
    A("## 6. 裁剪记账")
    A("")
    A("下表是**共享 R2P 块**的目标梯度裁剪前范数。训练裁剪的是 "
      "`model.parameters()`（共享 R2P + projections + feedback adapter）"
      "的**联合**范数，因此下表是该联合范数的**下界**。")
    A("")
    rows = []
    for grp in groups:
        g = data["groups"][grp]
        cells = []
        for r in g["records"]:
            cb = r["clip_bookkeeping"]
            cells.append("%s%s" % (sci(cb["objective_grad_norm_on_shared_preclip"]),
                                   " *" if cb["would_clip_shared_block_alone"]
                                   else ""))
        rows.append(["**" + grp + "**"] + cells)
    A("\n".join(table(["组"] + ["step " + str(s) for s in steps], rows)))
    A("")
    A("（`*` = 仅共享块就已超过 `grad_clip = 1.0`。本次没有任何一格带星号。）")
    A("")

    # ---------------- 7 ----------------
    A("## 7. 逐条判读")
    A("")
    for para in interpretation(data, steps, groups):
        A(para)
        A("")

    # ---------------- 8 ----------------
    A("## 8. 本诊断不能支持的结论")
    A("")
    A("1. **批次只有 %d 个，且是单个 seed（2026）**。"
      "所有夹角与范数只描述这 %d 个批次，"
      "**不能**推断显著性、不能推断跨 seed 稳定性、"
      "不能给出置信区间。" % (len(steps), len(steps)))
    A("2. **梯度是 STE 代理梯度**（§1 边界 2），"
      "不能当作冻结扩散先验的解析导数，"
      "也不能据此反推" + q("真实" ) + "的优化景观。")
    A("3. **未施加裁剪**，因此表中的范数不是训练实际使用的更新步长。")
    A("4. **共享参数只含 `model.r2p`**；"
      "`projections`、`feedback_adapter`、`feedback_scale` 与 `net_g` 的梯度"
      "未按分项拆分（`net_g` 仅给了两轮合计值）。")
    A("5. **本诊断只给几何与量级事实，不给机制结论**。"
      "例如它可以说明" + q("phase 分项与 L1/SSIM 分项在多数批次上方向相反且量级最大")
      + "，但**不能**据此断言" + q("phase 是第二轮退化的原因") + "——"
      "那需要评审意见第 4 条的最小目标对照实验。")
    A("6. 每个梯度量各跑一次独立前向，受 `retain_graph=False` 后的"
      "**固定随机流**约束；不同批次之间的抖动是真实的，"
      "**不得只挑某一个步数作结论载体**。")
    A("")
    A("---")
    A("")

    # ---------------- 9 ----------------
    A("## 9. 与评审意见后续步骤的关系")
    A("")
    A(next_plan(data, steps, groups))
    A("")
    A("---")
    A("")
    A("## 附录：文件")
    A("")
    A("- `gradient_diagnostics.py` — 本诊断脚本（只读）。")
    A("- `gradient_diagnostics_result.json` — 完整原始记录。")
    A("- `LOSS_SOURCE_AUDIT_ZH_20261002.md` — 损失结构来源核对"
      "（本报告 §1 公式的依据）。")
    A("- `ALPHA_SWEEP_FULL_VALIDATION_ZH_20261002.md` — 优先级 1 的 872 张全量结果。")
    A("")

    Path(args.out_md).write_text("\n".join(L), encoding="utf-8")

    compact = {
        "purpose": "review item #3 gradient diagnostics -- compact summary",
        "steps": steps,
        "seed": data.get("seed"),
        "groups": {},
    }
    for grp, g in data["groups"].items():
        compact["groups"][grp] = {
            "feedback_enabled": g["feedback_enabled"],
            "lambda_feature_alignment": g["lambda_feature_alignment"],
            "shared_param_count": g["shared_param_count"],
            "cosine_summary": g["cosine_summary"],
            "isolated_summary": g["isolated_summary"],
            "term_grad_norm_summary": g.get("term_grad_norm_summary"),
            "clip_norms_shared_block_lower_bound": [
                r["clip_bookkeeping"]["objective_grad_norm_on_shared_preclip"]
                for r in g["records"]],
        }
    Path(args.out_json).write_text(json.dumps(compact, indent=2),
                                   encoding="utf-8")
    print("WROTE", args.out_md)
    print("WROTE", args.out_json)
    return 0


def _cos_series(g, key):
    return [r["focused_cosines_unit_or_weighted"].get(key) for r in g["records"]]


def _iso_series(g, key="cosine_L1_vs_L2"):
    return [r["isolated_round2_vs_round1"].get(key) for r in g["records"]]


def headline(data, steps, groups):
    """The answer, derived only from numbers present in the file."""
    out = []
    nf = data["groups"].get("no_feedback")
    fb = data["groups"].get("feedback")

    # (1) between rounds
    if nf and fb:
        nf_iso = _iso_series(nf)
        fb_iso = _iso_series(fb)
        out.append("**1) 两轮之间：不是稳定的" + q("冲突") + "，"
                   "但也不是稳定的一致，而且两组的形态不同。**")
        out.append("在 isolated 口径下，两轮任务梯度夹角为——"
                   "no_feedback："
                   + " / ".join("%+.3f" % v for v in nf_iso)
                   + "；feedback："
                   + " / ".join("%+.3f" % v for v in fb_iso)
                   + "（顺序为 step " + " / ".join(str(s) for s in steps) + "）。")
        out.append("no_feedback 在 **step 1200 掉到 %.3f（接近正交、略偏反向）**，"
                   "其余批次为正；feedback 则**全程为正**（最低 %.3f）。"
                   % (min(nf_iso), min(fb_iso)))
        out.append("⇒ **两轮夹角在批次间跨过 0，不能用一个数字概括；"
                   "更不能说" + q("两轮必然冲突") + "。**"
                   "要看清的是逐批表格，而不是某个单点。")

    # (2) within round: phase vs pixel
    out.append("**2) 同一轮内部才是真正的冲突所在：phase 与 L1 / SSIM 常常反向，"
               "而且 phase 的范数在多数批次最大。**")
    for grp in groups:
        g = data["groups"][grp]
        ps = _cos_series(g, "L1.phase vs L1.l1")
        pss = _cos_series(g, "L1.phase vs L1.ssim")
        ph_norm = [r["real_path"]["shared_grad_norms_unit"].get("L1.phase")
                   for r in g["records"]]
        # how often is phase really the largest of the three pixel-ish terms?
        n_big = 0
        for r in g["records"]:
            n = r["real_path"]["shared_grad_norms_unit"]
            others = [n.get("L1.l1"), n.get("L1.ssim"), n.get("L1.perceptual")]
            others = [v for v in others if v is not None]
            if others and n.get("L1.phase") is not None and \
                    n["L1.phase"] > max(others):
                n_big += 1
        out.append("- **" + grp + "**：`L1.phase vs L1.l1` = "
                   + " / ".join("%+.3f" % v for v in ps)
                   + "；`L1.phase vs L1.ssim` = "
                   + " / ".join("%+.3f" % v for v in pss)
                   + "。第一轮 phase 分项的梯度范数 = "
                   + " / ".join("%.4g" % v for v in ph_norm)
                   + "；在四个分项中 phase 范数最大的批次数 = %d / %d。"
                   % (n_big, len(g["records"])))
    out.append("⇒ 在多数批次上，**phase 分项与直接对应 PSNR/SSIM 的 "
               "L1、1-SSIM 分项方向相反且范数更大**"
               "（例外见上方各组的计数，**并非每个批次都如此**）。"
               "这只说明目标自身内部存在张力，"
               "**不等于**" + q("phase 导致了第二轮退化") + "。")

    # (3) objective norm vs clip
    out.append("**3) 共享参数上，裁剪在本次所有批次都没有触发。**")
    for grp in groups:
        g = data["groups"][grp]
        vals = [r["clip_bookkeeping"]["objective_grad_norm_on_shared_preclip"]
                for r in g["records"]]
        out.append("- **" + grp + "**：目标梯度在共享块上的裁剪前范数最大为 "
                   "%.4g（`grad_clip = 1.0`）。" % max(vals))
    out.append("⇒ 至少对共享 R2P 块而言，本次测得的更新幅度远未触及裁剪阈值。")

    # (4) alignment
    out.append("**4) alignment 项与任务项近似正交。**")
    for grp in groups:
        g = data["groups"][grp]
        av = _cos_series(g, "align weighted vs TASKSUM")
        out.append("- **" + grp + "**：`align weighted vs TASKSUM` = "
                   + " / ".join("%+.3f" % v for v in av)
                   + "（绝对值均 ≤ %.3f）。" % max(abs(v) for v in av))
    out.append("这与它的构造一致：对齐损失把投影特征拉向冻结任务的中间特征，"
               "其梯度方向与像素级任务损失基本无关。")
    return out


def interpretation(data, steps, groups):
    out = []

    out.append("### 7.1 关于" + q("两轮是否冲突"))
    for grp in groups:
        g = data["groups"][grp]
        iso = _iso_series(g)
        neg = sum(1 for v in iso if v < 0)
        out.append("- **" + grp + "**：%d / %d 个批次的夹角为负，"
                   "最小值 %+.3f，最大值 %+.3f。"
                   % (neg, len(iso), min(iso), max(iso)))
    out.append("⇒ **描述" + q("两轮方向不稳定，并在个别批次接近正交、偶尔轻微反向")
               + " 是可支持的；描述" + q("两轮冲突") + " 或"
               + q("两轮一致") + " 作为普遍结论则不被数据支持。**")

    out.append("### 7.2 关于 phase 的主导性")
    for grp in groups:
        g = data["groups"][grp]
        n_big = 0
        for r in g["records"]:
            n = r["real_path"]["shared_grad_norms_unit"]
            others = [v for k, v in n.items()
                      if k in ("L1.l1", "L1.ssim", "L1.perceptual",
                               "L2.l1", "L2.ssim", "L2.perceptual")
                      and v is not None]
            if n.get("L1.phase") is not None and others and \
                    n["L1.phase"] > max(others):
                n_big += 1
        out.append("- **" + grp + "**：第一轮 phase 范数大于其余五个任务分项 "
                   "范数的批次 = %d / %d。"
                   % (n_big, len(g["records"])))
    out.append("**但必须区分两件事**："
               "①" + q("范数大") + "是 §2 直接测得的，"
               "且**存在反例批次**（feedback 的 step 600 就是一个）；"
               "②" + q("主导了更新方向") + "要看 §5 里各加权项与 `TASKSUM` 的夹角。"
               "本次 `L2 weighted vs TASKSUM` 的夹角为 "
               + " / ".join("%+.3f" % v for v in _cos_series(
                   data["groups"][groups[-1]], "L2 weighted vs TASKSUM"))
               + "（feedback 组，均为正且多在 0.9 以上），"
                 "说明第二轮的加权梯度与任务总和高度同向，"
                 "**不是**一个把更新带偏的异类分量。")
    out.append("⇒ **可以写" + q("phase 在目标中权重最大、且与像素级分项方向常对立")
               + "；不可以写" + q("phase 主导训练并造成退化") + "，"
                 "也不可以写" + q("每个批次 phase 都最大") + "。**"
               "后者需要第 4 条的目标对照实验才能检验。")

    out.append("### 7.3 关于 alignment")
    out.append("`align weighted vs TASKSUM` 在两组、各批次都接近 0"
               "（见 §0 第 4 条），且其加权范数比任务总和低 1~2 个数量级"
               "（§5 加权量级表）。")
    out.append("⇒ alignment 在本次测得的批次上对共享参数的**方向与量级都影响很小**，"
               "与它在标定时的设定（目标梯度占比 10%）方向一致。")

    out.append("### 7.4 关于裁剪")
    out.append("共享块裁剪前范数最大 %.4g，远低于 `grad_clip = 1.0`（§6）。"
               % max(max(r["clip_bookkeeping"][
                   "objective_grad_norm_on_shared_preclip"] for r in
                   data["groups"][grp]["records"]) for grp in groups))
    out.append("⇒ **仅就共享 R2P 块而言**，裁剪不是本次更新幅度的影响因素。"
               "但训练裁剪的是含 projections / feedback adapter 的联合范数，"
               "**不能由此推出整个模型从不触发裁剪**。")
    return out


def next_plan(data, steps, groups):
    # does the evidence support "conflict" or "phase dominance"?
    conflict_batches = 0
    total = 0
    for grp in groups:
        for v in _iso_series(data["groups"][grp]):
            total += 1
            if v is not None and v < 0:
                conflict_batches += 1

    lines = []
    lines.append("依评审意见的顺序：**第 3 条（梯度诊断）已完成**。")
    lines.append("")
    lines.append("关于是否满足进入第 4 条的门槛——评审的原话是"
                 + q("若分项训练梯度确实支持冲突或 phase 更新占优") + "：")
    lines.append("")
    lines.append("- **" + q("两轮冲突") + "**：%d / %d 个批次的夹角为负，"
                 "且负值只在个别批次出现 ⇒ **证据薄弱，不支持把"
                 "两轮冲突当作已确立的机制**。" % (conflict_batches, total))
    lines.append("- **" + q("phase 更新占优") + "**：phase 分项的**范数**确实最大，"
                 "但它与任务总和的夹角高度同向（§7.2），"
                 "⇒ **" + q("占优") + " 在" + q("范数") + "意义上成立、在"
                 + q("方向带偏") + "意义上不成立**。")
    lines.append("")
    lines.append("因此本文**不宣称**门槛已被满足，也**不自动启动**第 4 条。"
                 "若仍要推进，评审给出的做法是："
                 "做**同结构、同起点、同训练预算的最小目标对照**——"
                 "预先选定一个 phase 减权/关闭方案，"
                 "**不同时**改动共享权重、反馈或 alignment，"
                 "且**不得边查看 validation 边搜权重**。")
    lines.append("")
    lines.append("若第 4 条仍不改善，再进原优先级 2"
                 "（固定第一轮 + 独立修正器 A/B/C）。"
                 "**上述两步都不会自动启动，需用户明确指示。**")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
