"""Stage A2 archive: lock record, artifact receipt, protocol execution section."""
import hashlib
import json
import sys
import time
from pathlib import Path

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def log_window(path):
    times = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if "time_unix" in obj:
            times.append(float(obj["time_unix"]))
    return min(times), max(times)


results = json.loads((RUN / "metrics/A2_native_fit_results.json").read_text(encoding="utf-8"))
equiv = json.loads((RUN / "receipts/online_cache_equivalence.json").read_text(encoding="utf-8"))
cache_audit = json.loads((RUN / "metrics/A2_cache_audit.json").read_text(encoding="utf-8"))
_, train_end = log_window(RUN / "logs/a2_train.log")
executed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(train_end))

# ---- 1. lock: A2 execution record ----
lock_path = RUN / "config/diagnostic_lock.json"
lock = json.loads(lock_path.read_text(encoding="utf-8"))
lock["A2_native_head_execution"] = {
    "status": "native_head_diagnostic_complete",
    "executed_at": executed_at,
    "run": str(RUN),
    "gpu_used": True,
    "test_used": False,
    "validation_used_for_any_rule": False,
    "source_final_sha256": results["shared_initialisation"]["source_final_sha256"],
    "shared_init_sha256": results["shared_initialisation"]["sha256"],
    "head_architecture": results["native_head_architecture"],
    "cache_npz_sha256": cache_audit["cache_npz_sha256"],
    "cache_crops": cache_audit["plan_crops"],
    "cache_per_role": cache_audit["per_role"],
    "online_cache_equivalence_passed": equiv["passed"],
    "online_cache_equivalence_bitwise_equal": all(r["raw_cost_bitwise_equal"] for r in equiv["per_crop"]),
    "scale": results["scale"],
    "steps": results["config"]["steps"],
    "train_seconds": results["timing"]["train_seconds"],
    "mixed_final_pure_fit": results["heads"]["mixed"]["train"]["pure_fit"],
    "pure_final_pure_fit": results["heads"]["pure"]["train"]["pure_fit"],
    "expectation_term_helps": False,
    "cross_scene_transfer_beats_zero_baseline": False,
    "chosen_screening": results["heads"]["mixed"]["screening"]["chosen"],
    "screening_safe": results["heads"]["mixed"]["screening"]["safe"],
    "screening_useful": results["heads"]["mixed"]["screening"]["useful"],
    "gt_oracle_upper_bound_raw": results["heads"]["mixed"]["screening"]["reference_accept_all"]["mean_mse_improvement"],
    "execution_defects_only_no_protocol_change": True,
    "execution_attempts": {"cache_and_equivalence": 4, "training": 3, "diagnostic": 2},
}
lock["stage_A_status"] = "native_head_diagnostic_complete"
lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- 2. A2 artifact receipt ----
ARTIFACTS = [
    ("metrics/A2_native_fit_results.json", "A2 machine-readable results (fit, ranking, roles, screening)"),
    ("metrics/A2_cache_audit.json", "A2 cache build audit and source tensor audit"),
    ("cache/a2_spatial_cache.npz", "48-crop minimal spatial cache of the 50-channel head input"),
    ("cache/a2_cache_index.json", "crop plan, role positions, channel layout, cache hash"),
    ("checkpoints/diagnostic_heads/a2_native_common_init.pt", "shared initial weights of both diagnostic heads"),
    ("checkpoints/diagnostic_heads/a2_native_mixed.pt", "mixed-head final weights (diagnostic, not a restoration method)"),
    ("checkpoints/diagnostic_heads/a2_native_pure.pt", "pure-head final weights (diagnostic, not a restoration method)"),
    ("receipts/online_cache_equivalence.json", "mandatory online/cache equivalence evidence"),
    ("receipts/A2_execution.json", "execution receipt with defect list and timing"),
    ("logs/a2_native_train.csv", "per-step loss, grad norm, lr, finite flag, sample ids"),
    ("logs/a2_cache.log", "cache build log"),
    ("logs/a2_train.log", "paired training log"),
    ("figures/train_fit_and_cost_ranking.svg", "MLP vs native head fit and ranking curves"),
    ("figures/risk_coverage.svg", "training-safety risk-coverage / net gain (A1 forced coverage + A2 grid)"),
    ("docs/A2_findings.md", "A2 findings report"),
    ("scripts/a2_common.py", "shared A2 helpers (50-channel head input reconstruction)"),
    ("scripts/a2_cache.py", "cache build + equivalence check"),
    ("scripts/a2_train.py", "paired mixed/pure native-head diagnostic"),
    ("scripts/a2_diag.py", "cache fidelity / RNG sensitivity diagnostic"),
    ("scripts/a2_figure.py", "A2 figures"),
    ("scripts/a2_finalize.py", "status, stage timing, execution receipt"),
]
receipt = {
    "stage": "A2",
    "created_at": executed_at,
    "run": str(RUN),
    "protocol_id": json.loads((RUN / "config/locked_run.json").read_text(encoding="utf-8"))["protocol_id"],
    "protocol_changed": False,
    "gpu_used": True,
    "test_used": False,
    "validation_used_for_any_rule": False,
    "probe_only": True,
    "restoration_method_score": False,
    "artifacts": [],
    "note": "A2 is a 24-crop native-head learnability and loss-variable diagnostic. It is not a full R2P "
            "score, not a convergence proof, and not a claim that the judgement module is theoretically useless.",
}
for rel, desc in ARTIFACTS:
    path = RUN / rel
    entry = {"path": rel, "exists": path.exists(), "description": desc}
    if path.exists():
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = sha(path)
    receipt["artifacts"].append(entry)
(RUN / "receipts/A2_artifacts.json").write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- 3. protocol: append A2 execution section ----
protocol_path = RUN / "docs/diagnostic_protocol.md"
text = protocol_path.read_text(encoding="utf-8")
if "## 10. A2 执行记录与结果" not in text:
    text += """
## 10. A2 执行记录与结果（已完成）

- 状态：`native_head_diagnostic_complete`；GPU；未读 188 测试；未用 validation 选择任何规则；未重跑 872 诊断。
- 预算：A2 首次 GPU 前登记的新 2 小时绝对预算（`budget.json`），未沿用旧实验预算。
- 判断头：生产构造路径的 50 输入 / 3 输出原生头（`feature_projection` 96→8 可训练），
  两份副本共享初始权重 SHA256 `6323d52851ad9e67c2020463eab5d3e859e9964758dfbc311f66e69ec6e200b0`（来源 `final.pt` `279d0a32…e9a2`）。
- 缓存：48 裁剪（judgement 24 / bias 8 / safety 16），一次冻结双轮前向；`feature_projection` 不缓存。
- **缓存等价：通过，且为逐比特相等**（成本输出、损失、梯度、一次更新后的参数 max|Δ| 全为 0），
  非判断张量仍严格等于来源，GT 替换只改标签，M0 精确复制 T1。
- 训练：Adam lr 1e-4、有效 batch 2、微批 1、clip_norm 1、固定 500 步；唯一变量 = 损失。
- 结果摘要：
  - 训练内 L_fit 0.139464 → mixed 0.127693 / pure 0.127553（−8.4% / −8.5%）；
    三动作成本 MAE 2.114e-4 / 2.113e-4，优于零基线 2.247e-4 与训练均值基线 2.247e-4；
    训练内 Spearman ≈ 0.24–0.27。
  - mixed 与 pure 在所有指标上同级（终点 L_fit 相对差 0.11%，**pure 略好**）→ **期望项没有带来可测收益**。
  - 跨场景角色（bias / safety）MAE 9.69e-5 / 8.58e-5，**均劣于零基线** 8.17e-5 / 8.39e-5 → 不迁移；
    幅值系统性低估（safety 上 a=1.0 预测 0.0409 vs 真实 0.0970）。
  - 旧门槛网格 0…12.8：无任何合格的非零覆盖 → **全部拒绝**（coverage 0，safe=true，useful=false，有害率 NA）；
    GT 参照（每格最优动作上界）仅 +4.575e-6（≈ 尺度的 0.49%）。
- 判定：原生头在固定小样本上**能收敛式拟合**，但**不跨场景迁移**，且**去掉期望项不改善**；
  未找到符合预定条件的非零修改。可进入 B 的信号（pure 优于 mixed / 安全角色净改善为正）**未出现**。
- 执行偏差：7 次进程尝试中 6 次失败，全部是新写脚本的执行缺陷（详见 `docs/A2_findings.md` §9 与
  `receipts/A2_execution.json`），**未改动算法/目标/数据/协议**，协议 ID 不变。
- 详见 `docs/A2_findings.md`、`metrics/A2_native_fit_results.json`、`receipts/A2_artifacts.json`。
"""
    protocol_path.write_text(text, encoding="utf-8")
else:
    print("protocol section 10 already present, skipped append")

print(json.dumps({"archived": True, "executed_at": executed_at, "artifacts": len(ARTIFACTS)}))
for rel, _ in ARTIFACTS:
    path = RUN / rel
    print(rel, path.exists(), path.stat().st_size if path.exists() else "-")
