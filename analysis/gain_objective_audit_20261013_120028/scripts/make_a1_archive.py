import json, sys
from pathlib import Path
RUN = Path(sys.argv[1])
import hashlib

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# 1. lock: add A1 execution record
lockp = RUN / "config" / "diagnostic_lock.json"
lock = json.loads(lockp.read_text(encoding="utf-8"))
lock["A1_probe_execution"] = {
    "status": "lightweight_probe_complete",
    "executed_at": "2026-10-13T12:23+08:00",
    "run": str(RUN),
    "cpu_only": True,
    "gpu_used": False,
    "common_init_sha256": "c84ea6f5d9fbc4b19fae3f22899f2f0c4d27825a8154168dfc52e44649ae7c65",
    "sample_indices_sha256": "fae6753b60a2f8b55604781786f72a47780c2193eb8a6dde28701d225054e10c",
    "train_seconds": 1.519399024080485,
    "loss_equivalence_torch_vs_numpy_abs_diff": 1.9566664261061462e-10,
    "parameters_source": "handoff document section 7 fixed defaults; not tuned on any result",
    "signal_present_at_10pct": {"mixed": False, "pure": False},
}
lock["stage_A_status"] = "lightweight_probe_complete"
lockp.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")

# 2. A1 artifacts receipt
arts = [
    ("metrics/A1_probe_results.json", "A1 probe machine-readable results"),
    ("checkpoints/diagnostic_heads/a1_mlp_mixed.pt", "mixed replica weights"),
    ("checkpoints/diagnostic_heads/a1_mlp_pure.pt", "pure replica weights"),
    ("checkpoints/diagnostic_heads/a1_mlp_common_init.pt", "shared initial weights"),
    ("figures/train_fit_and_cost_ranking.svg", "training curves"),
    ("figures/risk_coverage.svg", "safety risk-coverage"),
    ("docs/A1_findings.md", "A1 findings report"),
]
receipt = {
    "stage": "A1",
    "created_at": "2026-10-13T12:30+08:00",
    "run": str(RUN),
    "cpu_only": True,
    "gpu_used": False,
    "test_used": False,
    "validation_used_for_any_rule": False,
    "probe_only": True,
    "restoration_method_score": False,
    "artifacts": [],
    "note": "A1 is a learnability probe on the existing scattered 50-d cache; it is not a native-head "
            "result and not a full R2P score.",
}
for rel, desc in arts:
    p = RUN / rel
    e = {"path": rel, "exists": p.exists(), "description": desc}
    if p.exists():
        e["bytes"] = p.stat().st_size
        e["sha256"] = sha(p)
    receipt["artifacts"].append(e)
(RUN / "receipts" / "A1_artifacts.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")

# 3. stage timing
(RUN / "logs").mkdir(parents=True, exist_ok=True)
csvp = RUN / "logs" / "stage_timing.csv"
rows = [
    "stage,upstream_forward_seconds,head_train_seconds,data_load_seconds,metrics_seconds,calibration_seconds,cache_read_write_seconds,checkpoint_archive_seconds,wall_clock_seconds,note",
    "A0,,,,,,,,1.9,cpu-only source/cache/numeric audit (torch import + npz load + sha)",
    "A1,0,1.52,,,0,,,9.1,mlp 50-32-32-3 1000 steps x2 replicas on cpu; no upstream forward",
    "A2,,,,,,,,,not started",
    "A3,,,,,,,,,not started",
]
csvp.write_text("\n".join(rows) + "\n", encoding="utf-8")

# 4. protocol: append A1 execution section
prot = RUN / "docs" / "diagnostic_protocol.md"
t = prot.read_text(encoding="utf-8")
if "## 9. A1 执行记录与结果" not in t:
    t += """
## 9. A1 执行记录与结果（已完成）

- 状态：`lightweight_probe_complete`；CPU；未用 GPU；未读 188 测试；未重跑 872。
- 共同初始权重 SHA256 `c84ea6f5d9fbc4b19fae3f22899f2f0c4d27825a8154168dfc52e44649ae7c65`；样本索引 SHA256 `fae6753b60a2f8b55604781786f72a47780c2193eb8a6dde28701d225054e10c`。
- 损失等价性：torch（与 `action_gate.cost_loss` 同定义）对比独立 numpy 解析实现，最大绝对差 `1.96e-10`。核对时修正了 numpy 参照漏写负号（应为 `softmax(-costs)`）的错误。
- 训练耗时 1.52 秒（CPU）。
- 结果摘要：
  - 训练内（2048）：L_fit 0.09892 → mixed 0.02679 / pure 0.02685；成本 MAE 0.262 → 0.1085(mixed)/0.1049(pure)；训练内三动作平均 Spearman 升到约 0.42/0.41。
  - 跨角色：判断训练全集 MAE 0.2588/0.2564（Spearman≈0.21）、偏差 0.1742/0.1580（≈0.10）、安全 0.2432/0.2396（≈0.06）。全集/偏差/安全上 mixed 均未超过"最优常数动作"基线 0.1059。
  - 安全角色强制 1%–20% 覆盖：接受区平均真实净改善全为负，10% 处 mixed −2.56e-4 / pure −3.22e-4，**劣于**同动作随机参照 −5.08e-5；有害接受率约 42%。
  - `signal_present_at_10pct`：mixed=false，pure=false（4 场景仅 1 个为正）。
- 判定：散点表示在训练内可拟合（过拟合于 2048 抽样），但不跨角色/场景迁移；**去掉期望项没有实质改善**（L_fit 几乎不变、MAE 略优、排序略降）。
- 限制：本结果只说明该探针/预算/散点表示未形成可迁移信号，**不能**据此判定原生判断头坏掉；原生头判定见 A2。
- 详见 `docs/A1_findings.md`、`metrics/A1_probe_results.json`、`receipts/A1_artifacts.json`。
    """
    prot.write_text(t, encoding="utf-8")
else:
    print("protocol section 9 already present, skipped append")
print("updated lock, receipt, timing, protocol")
for rel, _ in arts:
    p = RUN / rel
    print(rel, p.exists(), p.stat().st_size if p.exists() else "-")
