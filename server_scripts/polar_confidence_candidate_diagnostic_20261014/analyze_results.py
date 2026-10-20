#!/usr/bin/env python3
"""Summarize frozen-model C -> U interventions without causal overclaims.

Only Python's standard library is required.  The input contains ``rows`` with
sample_id, scene, and a conditions mapping.  Restoration gains are always
recomputed from the per-condition PSNR/SSIM; pixel-change measures are not
mistaken for restoration gains.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REAL = "real_quality"
DEFAULT_LABELS = {
    REAL: "真实质量图",
    "fixed_quality": "固定训练质量条件",
    "constant_quality": "固定训练质量条件",
    "shifted_quality": "空间错位质量图",
    "spatial_shift": "空间错位质量图",
}
FIELDS = (
    "t1_psnr", "t1_ssim", "u_psnr", "u_ssim",
    "u_minus_t1_psnr", "u_minus_t1_ssim", "u_t1_mae",
    "u_vs_real_mae", "u_vs_real_maxabs", "b_pred_mean", "c_pred_mean",
    "b_cond_mean", "c_cond_mean",
)


def native(value: Any) -> Any:
    """Convert scalar-like values and reject JSON NaN/Infinity on output."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if hasattr(value, "item"):
        return native(value.item())
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def difference(a: Any, b: Any) -> float | None:
    av, bv = number(a), number(b)
    return None if av is None or bv is None else av - bv


def summarize(values: list[Any]) -> dict[str, Any]:
    valid = [v for value in values if (v := number(value)) is not None]
    if not valid:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None,
                "std": None, "positive": 0, "negative": 0, "zero": 0}
    return {
        "n": len(valid), "mean": statistics.fmean(valid),
        "median": statistics.median(valid), "min": min(valid), "max": max(valid),
        "std": statistics.stdev(valid) if len(valid) > 1 else 0.0,
        "positive": sum(v > 0 for v in valid),
        "negative": sum(v < 0 for v in valid), "zero": sum(v == 0 for v in valid),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(native(data), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(native(rows))


def display(value: Any, digits: int = 6) -> str:
    v = number(value)
    return "缺失" if v is None else f"{v:.{digits}f}"


def signed(value: Any, digits: int = 6) -> str:
    v = number(value)
    return "缺失" if v is None else f"{v:+.{digits}f}"


def analyze(input_path: Path, run_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    source_rows = payload.get("rows")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("Input must contain a nonempty rows list.")
    metadata = payload.get("run_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {"original_run_metadata": metadata}
    conditions = list(dict.fromkeys(
        str(key) for row in source_rows if isinstance(row, dict)
        for key in row.get("conditions", {})
    ))
    if REAL not in conditions:
        raise ValueError(f"Input has no {REAL!r} condition.")
    conditions = [REAL] + [name for name in conditions if name != REAL]
    controls = [name for name in conditions if name != REAL]
    labels = {**DEFAULT_LABELS, **metadata.get("condition_labels", {})}
    expected_images = int(metadata.get("expected_images", metadata.get("expected_samples", 28)))
    issues: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    scene_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    t1_equal_count = 0
    all_u_equal_count = 0
    u_hash_equal_to_real = {name: 0 for name in controls}
    output_values: dict[str, dict[str, list[Any]]] = {
        name: {field: [] for field in FIELDS} for name in conditions
    }
    contrast_values = {
        name: {field: [] for field in ("u_psnr", "u_ssim")} for name in controls
    }
    for index, source in enumerate(source_rows):
        if not isinstance(source, dict):
            issues.append({"kind": "invalid_row", "row_index": index})
            continue
        sample_id = str(source.get("sample_id", ""))
        scene = str(source.get("scene", ""))
        if not sample_id or not scene:
            issues.append({"kind": "missing_identity", "row_index": index,
                           "sample_id": sample_id, "scene": scene})
        if sample_id in seen:
            issues.append({"kind": "duplicate_sample_id", "sample_id": sample_id})
        seen.add(sample_id)
        condition_data = source.get("conditions", {})
        flat: dict[str, Any] = {"sample_id": sample_id, "scene": scene}
        t1_hashes, u_hashes = [], []
        for name in conditions:
            current = condition_data.get(name)
            if not isinstance(current, dict):
                issues.append({"kind": "missing_condition", "sample_id": sample_id,
                               "condition": name})
                current = {}
            current = dict(current)
            current["u_minus_t1_psnr"] = difference(current.get("u_psnr"), current.get("t1_psnr"))
            current["u_minus_t1_ssim"] = difference(current.get("u_ssim"), current.get("t1_ssim"))
            for field in FIELDS:
                value = number(current.get(field))
                flat[f"{name}__{field}"] = value
                output_values[name][field].append(value)
                if field in ("t1_psnr", "t1_ssim", "u_psnr", "u_ssim") and value is None:
                    issues.append({"kind": "missing_or_nonfinite_metric", "sample_id": sample_id,
                                   "condition": name, "field": field})
            for field in ("t1_sha256", "u_sha256"):
                hash_value = str(current.get(field, ""))
                flat[f"{name}__{field}"] = hash_value
                if len(hash_value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in hash_value):
                    issues.append({"kind": "missing_or_invalid_hash", "sample_id": sample_id,
                                   "condition": name, "field": field})
                (t1_hashes if field == "t1_sha256" else u_hashes).append(hash_value.lower())
        t1_equal = all(t1_hashes) and len(set(t1_hashes)) == 1
        all_u_equal = all(u_hashes) and len(set(u_hashes)) == 1
        flat["all_conditions_t1_hash_equal"] = t1_equal
        flat["all_conditions_u_hash_equal"] = all_u_equal
        t1_equal_count += int(t1_equal)
        all_u_equal_count += int(all_u_equal)
        if not t1_equal:
            issues.append({"kind": "t1_hash_mismatch", "sample_id": sample_id})
        real_data = condition_data.get(REAL, {})
        for name in controls:
            data = condition_data.get(name, {})
            equal = bool(real_data.get("u_sha256")) and data.get("u_sha256") == real_data.get("u_sha256")
            u_hash_equal_to_real[name] += int(equal)
            flat[f"{name}__u_hash_equal_real"] = equal
            for field in ("u_psnr", "u_ssim"):
                delta = difference(real_data.get(field), data.get(field))
                flat[f"real_minus_{name}__{field}"] = delta
                contrast_values[name][field].append(delta)
            for field in ("t1_psnr", "t1_ssim"):
                delta = difference(real_data.get(field), data.get(field))
                if delta is not None and abs(delta) > 1e-8:
                    issues.append({"kind": "t1_metric_mismatch", "sample_id": sample_id,
                                   "condition": name, "field": field, "difference": delta})
        paired_rows.append(flat)
        scene_groups[scene].append(flat)
    if len(paired_rows) != expected_images:
        issues.append({"kind": "image_count_mismatch", "expected": expected_images,
                       "actual": len(paired_rows)})
    if len(conditions) != 3:
        issues.append({"kind": "condition_count_mismatch", "expected": 3,
                       "actual": len(conditions)})

    numeric_fields = [key for key in paired_rows[0] if key not in ("sample_id", "scene")
                      and "sha256" not in key and "hash_equal" not in key]
    scene_rows: list[dict[str, Any]] = []
    for scene in sorted(scene_groups):
        group = scene_groups[scene]
        row: dict[str, Any] = {"scene": scene, "image_count": len(group)}
        for key in numeric_fields:
            stats = summarize([entry.get(key) for entry in group])
            row[key] = stats["mean"]
            row[f"{key}__valid_images"] = stats["n"]
        row["all_conditions_t1_hash_equal_count"] = sum(
            entry["all_conditions_t1_hash_equal"] for entry in group)
        row["all_conditions_u_hash_equal_count"] = sum(
            entry["all_conditions_u_hash_equal"] for entry in group)
        scene_rows.append(row)
    condition_summary = {}
    for name in conditions:
        condition_summary[name] = {
            "label": labels.get(name, name),
            "image_equal_weight": {field: summarize(output_values[name][field]) for field in FIELDS},
            "scene_equal_weight": {
                field: summarize([row[f"{name}__{field}"] for row in scene_rows]) for field in FIELDS
            },
        }
    contrasts = {}
    for name in controls:
        contrasts[name] = {
            "definition": "real_quality U metric minus control U metric; positive favors real quality",
            "image_equal_weight": {field: summarize(contrast_values[name][field]) for field in ("u_psnr", "u_ssim")},
            "scene_equal_weight": {
                field: summarize([row[f"real_minus_{name}__{field}"] for row in scene_rows])
                for field in ("u_psnr", "u_ssim")
            },
        }
    summary = {
        "analysis_kind": "frozen_model_quality_condition_dependency_diagnostic",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path), "run_metadata": metadata,
        "expected_images": expected_images, "image_count": len(paired_rows),
        "scene_count": len(scene_rows), "conditions": condition_summary,
        "real_minus_control": contrasts,
        "integrity": {
            "passed": not issues, "issues": issues,
            "t1_all_condition_hash_equal_count": t1_equal_count,
            "all_condition_u_hash_equal_count": all_u_equal_count,
            "control_u_hash_equal_real_count": u_hash_equal_to_real,
            "restoration_gains_recomputed_from_metrics": True,
        },
        "limits": [
            "固定已训练模型的推理干预只检查条件依赖，不是匹配训练消融或因果收益证明。",
            "空间错位可能产生模型未见过的输入条件，错位变差不能独立证明真实置信训练有益。",
            "仅固定验证子集，未新增训练、未进行188张正式测试，不是正式实验完成或稳定有效证明。",
            "本诊断直接报告未门控候选U；最终动作判断器全部拒绝时的T2不能掩盖候选变化。",
            "GT仅用于评价；全部候选由观测输入及质量干预产生。",
            "不根据这些验证结果选择权重、阈值或追加训练；后续方案须单独锁定匹配对照。",
        ],
    }
    metrics_dir = output_dir or run_root / "metrics"
    report_dir = (output_dir / "reports") if output_dir else run_root / "reports"
    write_csv(metrics_dir / "paired_per_image.csv", paired_rows)
    write_csv(metrics_dir / "paired_per_scene.csv", scene_rows)
    write_json(metrics_dir / "summary.json", summary)

    lines = [
        "# 固定模型质量条件 → 第二轮候选诊断", "",
        f"这是一次只读推理诊断：固定 {len(paired_rows)} 张验证图、{len(scene_rows)} 个场景、"
        f"{len(conditions)} 种质量条件，没有新增训练，也没有运行188张正式测试。",
        "",
        "本报告直接比较未门控候选 U，不把最终全拒绝时的 T2=T1 当作候选没有变化。",
        "",
        "## 1. 完整性检查", "",
        f"- 完整性检查：{'通过' if not issues else '未通过，详见 summary.json 的 integrity.issues'}。",
        f"- 三条件第一轮浮点哈希相同：{t1_equal_count}/{len(paired_rows)} 张。",
        f"- 三条件候选浮点哈希全部相同：{all_u_equal_count}/{len(paired_rows)} 张。",
        "- 恢复收益使用 U 的 PSNR/SSIM 减去 T1 的对应指标；数值改动幅度单独报告。",
        "",
        "## 2. 候选是否比第一轮更好", "",
        "正数表示候选改善，负数表示候选变差。图像等权平均中每张图权重相同；"
        "场景等权平均先求各场景平均，再让各场景权重相同。", "",
        "| 条件 | 图像等权ΔPSNR(dB) | 场景等权ΔPSNR(dB) | 图像等权ΔSSIM | 场景等权ΔSSIM | PSNR正/负/零场景 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in conditions:
        image_stats = condition_summary[name]["image_equal_weight"]
        scene_stats = condition_summary[name]["scene_equal_weight"]
        ps = scene_stats["u_minus_t1_psnr"]
        lines.append(
            f"| {labels.get(name, name)} | {signed(image_stats['u_minus_t1_psnr']['mean'])} | "
            f"{signed(ps['mean'])} | {signed(image_stats['u_minus_t1_ssim']['mean'])} | "
            f"{signed(scene_stats['u_minus_t1_ssim']['mean'])} | {ps['positive']}/{ps['negative']}/{ps['zero']} |")
    lines += ["", "## 3. 真实质量条件与对照的配对差", "",
              "下表为真实质量候选减去对照候选。正数表示真实质量条件更好；"
              "这仍只是同一模型在条件替换后的输出差。", "",
              "| 对照 | 图像等权ΔPSNR(dB) | 场景等权ΔPSNR(dB) | 图像等权ΔSSIM | 场景等权ΔSSIM | PSNR真实条件胜/负/平场景 |",
              "|---|---:|---:|---:|---:|---|"]
    for name in controls:
        image_stats = contrasts[name]["image_equal_weight"]
        scene_stats = contrasts[name]["scene_equal_weight"]
        ps = scene_stats["u_psnr"]
        lines.append(
            f"| {labels.get(name, name)} | {signed(image_stats['u_psnr']['mean'])} | "
            f"{signed(ps['mean'])} | {signed(image_stats['u_ssim']['mean'])} | "
            f"{signed(scene_stats['u_ssim']['mean'])} | {ps['positive']}/{ps['negative']}/{ps['zero']} |")
    lines += ["", "## 4. 每个场景的结果", "",
              "| 场景 | 图数 | " + " | ".join(f"{labels.get(name, name)} U−T1 PSNR" for name in conditions) + " |",
              "|---|---:|" + "---:|" * len(conditions)]
    for row in scene_rows:
        lines.append(f"| {row['scene']} | {row['image_count']} | " + " | ".join(
            signed(row[f"{name}__u_minus_t1_psnr"]) for name in conditions) + " |")
    lines += ["", "## 5. 当前可以怎样理解", ""]
    if issues:
        lines.append("存在完整性问题，不能把下面的数值当作完成且有效的诊断结论；应先修复执行或记录问题，算法条件不得静默改变。")
    real_stats = condition_summary[REAL]["image_equal_weight"]
    real_p, real_s = real_stats["u_minus_t1_psnr"]["mean"], real_stats["u_minus_t1_ssim"]["mean"]
    if real_p is None or real_s is None:
        lines.append("真实质量条件的必要恢复指标缺失，无法判断候选是否改善第一轮。")
    elif real_p > 0 and real_s >= 0:
        lines.append("真实质量条件的候选在本验证子集的图像等权平均中，PSNR提高且SSIM未下降。"
                     "这是一项局部诊断信号；仍需看场景负面结果，不能称为稳定循环收益。")
    elif real_p > 0:
        lines.append("真实质量条件的候选平均PSNR提高，但平均SSIM下降，指标存在取舍，不能宣称全面改善。")
    else:
        lines.append("真实质量条件的候选平均PSNR没有提高，因此本子集不支持候选已经形成平均循环收益。")
    for name in controls:
        stats = contrasts[name]["image_equal_weight"]
        dp, ds = stats["u_psnr"]["mean"], stats["u_ssim"]["mean"]
        if dp is None or ds is None:
            text = "配对指标缺失，无法比较"
        elif dp > 0 and ds >= 0:
            text = "真实质量条件在平均PSNR上更好且SSIM未下降，说明该固定模型在这些图上存在有利的条件依赖"
        elif dp <= 0 and ds <= 0:
            text = "真实质量条件没有显示平均指标优势，本子集未支持该条件有益"
        else:
            text = "PSNR与SSIM方向不一致，不能把差异概括为一致收益"
        lines.append(f"- 相对{labels.get(name, name)}：{text}。")
    lines += ["", "## 6. 结论边界", ""] + [f"- {item}" for item in summary["limits"]]
    lines += ["", "## 7. 可复核文件", "",
              "- `metrics/per_image.json`：原始逐图数值、条件统计和浮点哈希。",
              "- `metrics/paired_per_image.csv`：三条件逐图指标及配对差。",
              "- `metrics/paired_per_scene.csv`：逐场景平均及有效样本数。",
              "- `metrics/summary.json`：两种等权汇总、正负数量和完整性记录。",
              "- 本次权重、固定清单和全部质量条件路径的执行证据以 run_metadata 及运行档案为准。", ""]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "RESULT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Diagnostic run directory")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = analyze(args.input or root / "metrics" / "per_image.json", root, args.output_dir)
    print(json.dumps({"image_count": summary["image_count"], "scene_count": summary["scene_count"],
                      "integrity_passed": summary["integrity"]["passed"],
                      "issue_count": len(summary["integrity"]["issues"])}, ensure_ascii=False))
    if not summary["integrity"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
