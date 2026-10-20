"""CPU consistency audit for experiment 35; does not recompute image metrics."""
import argparse
import hashlib
import json
import math
import statistics
import traceback
from collections import Counter
from pathlib import Path

import torch


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def finite(value):
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert math.isfinite(value), value
    return float(value)


def close(actual, expected, label="value", tolerance=1e-10):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), label
        for key in expected:
            close(actual[key], expected[key], label + "/" + key, tolerance)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert abs(finite(actual) - finite(expected)) <= tolerance, (label, actual, expected)
    else:
        assert actual == expected, (label, actual, expected)


def summarize(rows, protocol_id):
    output = {"protocol_id": protocol_id, "n_rows": len(rows), "absolute": {}, "comparisons": {}}
    for name in ("T1", "F", "best", "final"):
        output["absolute"][name] = {
            metric: statistics.fmean(row["metrics"][name][metric] for row in rows)
            for metric in ("psnr", "ssim")
        }
    for name in ("F", "best", "final"):
        result = {}
        for metric in ("psnr", "ssim"):
            values = [row["metrics"][name][metric] - row["metrics"]["T1"][metric] for row in rows]
            result[metric] = {
                "mean": statistics.fmean(values),
                "positive_ids": [row["sample_id"] for row, value in zip(rows, values) if value > 0],
            }
        result["action_reference_pass"] = (
            result["psnr"]["mean"] >= 0.05
            and result["ssim"]["mean"] >= 0
            and len(result["psnr"]["positive_ids"]) >= 6
        )
        output["comparisons"][name + "_minus_T1"] = result
    return output


def check_variable(value, height, width, bound, label):
    assert isinstance(value, torch.Tensor), label
    assert list(value.shape) == [1, 12, height, width], (label, list(value.shape))
    assert value.device.type == "cpu" and torch.isfinite(value).all().item(), label
    maximum = value.detach().abs().max().item()
    assert maximum <= bound + 2e-8, (label, maximum, bound)
    return maximum, value.detach().abs().mean().item()


def audit(root):
    torch.set_num_threads(1)
    cfg = read(root / "config/locked.json")
    assert cfg["protocol_id"] and cfg["sources"] and cfg["code_sources"]
    assert cfg["steps"] == 100 and cfg["lr"] == 0.001 and cfg["bound"] == 0.02
    assert str(Path(__file__).resolve()) in {str(Path(item["path"]).resolve()) for item in cfg["code_sources"]}
    checked_sources = []
    revision_path = root / "config/implementation_revision_1.json"
    revision = read(revision_path) if revision_path.exists() else None
    changes = {} if revision is None else {r["path"]: r for r in revision["replacements"]}
    if revision:
        assert revision["protocol_id"] == cfg["protocol_id"]
        assert set(changes) == {str(root / "scripts/run_reachability.py"), str(root / "scripts/audit_reachability.py")}
        parity = read(root / "reports/execution_parity.json")
        assert parity["passed"] is True and parity["blocks"] > 0
        assert [r["delta"] for r in parity["cases"]] == ["zero", "ramp"]
        assert all(r["output_equal"] and r["loss_equal"] and r["input_gradient_equal"] and r["max_gradient_diff"] == 0 for r in parity["cases"])
    for item in cfg["sources"] + cfg["code_sources"] + [cfg["reference"]]:
        expected = item["sha256"]
        if item["path"] in changes:
            change = changes[item["path"]]
            assert change["original_sha256"] == expected and sha(change["preserved_original_path"]) == expected
            expected = change["repaired_sha256"]
        actual = sha(item["path"])
        assert actual == expected, ("source hash", item["path"], actual)
        checked_sources.append({"path": item["path"], "sha256": actual})

    items = cfg["items"]
    assert len(items) == 8 and len({item["sample_id"] for item in items}) == 8
    assert len(Counter(item["scene"] for item in items)) == 8
    rows = read(root / "metrics/rows.json")["rows"]
    assert len(rows) == 8
    assert [row["sample_id"] for row in rows] == [item["sample_id"] for item in items]
    references = {row["sample_id"]: row for row in read(cfg["reference"]["path"])["rows"]}
    assert len(references) == 872
    checked_artifacts = []
    for row, item in zip(rows, items):
        sample = row["sample_id"]
        assert row["scene"] == item["scene"]
        assert row["shape"] == [1, 3, item["height"], item["width"]], sample
        assert row["steps"] == cfg["steps"] and row["zero_identity"] is True, sample
        assert finite(row["first_grad_norm"]) > 0, sample
        reference = references[sample]
        assert row["shape"] == reference["shape"] and row["t1_sha256"] == reference["t1_sha256"], sample
        close(row["metrics"]["T1"], reference["metrics"]["T1"], sample + "/old33/T1")
        close(row["metrics"]["F"], reference["metrics"]["R1F"], sample + "/old33/F")
        for name in ("T1", "F", "best", "final"):
            assert set(row["metrics"][name]) == {"psnr", "ssim"}, sample
            for metric in ("psnr", "ssim"):
                finite(row["metrics"][name][metric])

        curve_path = Path(row["curve_path"])
        variables_path = Path(row["variables_path"])
        assert curve_path.is_absolute() and variables_path.is_absolute(), sample
        # Experimental variable files and curves must stay in this experiment directory.
        assert root.resolve() in curve_path.resolve().parents, sample
        assert root.resolve() in variables_path.resolve().parents, sample
        curve_object = read(curve_path)
        curve = curve_object["curve"] if isinstance(curve_object, dict) else curve_object
        assert isinstance(curve, list) and len(curve) == 101, sample
        assert [point["step"] for point in curve] == list(range(101)), sample
        losses = [finite(point["loss"]) for point in curve]
        best_step = 0
        for step in range(1, 101):
            if losses[step] < losses[best_step]:
                best_step = step
        assert row["best_step"] == best_step, sample
        close(row["initial_loss"], losses[0], sample + "/initial_loss")
        close(row["final_loss"], losses[100], sample + "/final_loss")
        close(row["best_loss"], losses[best_step], sample + "/best_loss")
        assert row["best_loss"] <= row["initial_loss"] + 1e-10, sample
        assert row["best_loss"] <= row["final_loss"] + 1e-10, sample

        state = torch.load(variables_path, map_location="cpu", weights_only=False)
        assert state["step"] == 100 and state["best_step"] == best_step, sample
        close(state["best_loss"], losses[best_step], sample + "/checkpoint/best_loss")
        assert state["rng"], sample
        noise = state["noise"]
        assert isinstance(noise, torch.Tensor) and list(noise.shape) == [1, 16, 256], sample
        assert torch.isfinite(noise).all().item(), sample
        assert tensor_sha(noise) == row["noise_sha256"], sample
        receipt = row["noise_receipt"]
        close(state["noise_receipt"], receipt, sample + "/checkpoint/noise_receipt")
        assert receipt["shape"] == [1, 16, 256] and receipt["used_sha256"] == row["noise_sha256"], sample
        assert len(receipt["calls"]) >= 103, sample
        for call in receipt["calls"]:
            assert call["scope"] == "full", sample
            assert call["native_sha256"] == call["used_sha256"] == row["noise_sha256"], sample
        close(state["first_grad_norm"], row["first_grad_norm"], sample + "/first_grad_norm")
        optimizer = state["optimizer"]
        assert len(optimizer["state"]) == 1 and len(optimizer["param_groups"]) == 1, sample
        group = optimizer["param_groups"][0]
        assert group["lr"] == 0.001 and tuple(group["betas"]) == (0.9, 0.999), sample
        assert group["eps"] == 1e-8 and group["weight_decay"] == 0 and len(group["params"]) == 1, sample
        moment = next(iter(optimizer["state"].values()))
        assert finite(float(moment["step"])) == 100, sample
        for name in ("exp_avg", "exp_avg_sq"):
            assert list(moment[name].shape) == [1, 12, item["height"], item["width"]], sample
            assert torch.isfinite(moment[name]).all().item(), sample
        assert (moment["exp_avg_sq"] >= 0).all().item(), sample
        for name, key in (("best", "best_delta"), ("final", "delta")):
            maximum, mean = check_variable(state[key], item["height"], item["width"], cfg["bound"], sample + "/" + key)
            description = row["delta"][name]
            close(description["max_abs"], maximum, sample + "/" + name + "/max_abs", 2e-8)
            close(description["mean_abs"], mean, sample + "/" + name + "/mean_abs", 2e-8)
            assert 0 <= finite(description["bound_fraction"]) <= 1, sample
            assert finite(description["tv"]) >= 0, sample
        if best_step == 0:
            assert torch.count_nonzero(state["best_delta"]).item() == 0, sample
        for path in (curve_path, variables_path):
            checked_artifacts.append({"path": str(path), "sha256": sha(path)})

    frozen = read(root / "reports/frozen.json")
    assert frozen["all_equal"] is True and frozen["all_grad_none"] is True
    assert frozen["before"] and frozen["before"] == frozen["after"]
    assert all(isinstance(value, str) and len(value) == 64 for value in frozen["before"].values())
    computed = summarize(rows, cfg["protocol_id"])
    close(read(root / "reports/summary.json"), computed, "summary")
    return {
        "status": "verified", "protocol_id": cfg["protocol_id"], "n_rows": len(rows),
        "scope": "CPU source/checkpoint/curve consistency and independent eight-example aggregation; no pixel-metric recomputation or global-optimality claim",
        "checked_sources": checked_sources, "checked_artifacts": checked_artifacts,
        "locked_sha256": sha(root / "config/locked.json"), "rows_sha256": sha(root / "metrics/rows.json"),
        "summary_sha256": sha(root / "reports/summary.json"), "frozen_sha256": sha(root / "reports/frozen.json"),
        "recomputed_summary": computed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    try:
        result = audit(arguments.root)
    except Exception:
        result = {"status": "failed", "error": traceback.format_exc()}
    output_path = arguments.root / "reports/independent_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps({key: value for key, value in result.items() if key != "recomputed_summary"}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "verified" else 1)
