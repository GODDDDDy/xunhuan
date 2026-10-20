"""Stage A2 finalisation: status, stage-timing row, execution receipt.

Reads the two stage logs and the two timing JSONs, so the recorded numbers are
derived from the run rather than transcribed.
"""
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(HERE))

from common import ROOT, json_save, status  # noqa: E402

assert Path(ROOT).resolve() == RUN, "ROOT must be the new audit run"


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
    assert times, f"no time_unix found in {path}"
    return min(times), max(times)


def main():
    cache_timing = json.loads((RUN / "logs/a2_timing.json").read_text(encoding="utf-8"))
    train_timing = json.loads((RUN / "logs/a2_train_timing.json").read_text(encoding="utf-8"))

    cache_start, cache_end = log_window(RUN / "logs/a2_cache.log")
    train_start, train_end = log_window(RUN / "logs/a2_train.log")
    cache_wall = cache_end - cache_start
    train_wall = train_end - train_start

    upstream = cache_timing["cache_forward_seconds"] + cache_timing["equivalence_seconds"]
    cache_rw = (cache_timing["total_seconds"] - cache_timing["model_build_seconds"]
                - cache_timing["cache_forward_seconds"] - cache_timing["equivalence_seconds"])
    wall = cache_wall + train_wall

    # ---- stage_timing.csv: fill the A2 row, keep every other row untouched ----
    csv_path = RUN / "logs/stage_timing.csv"
    rows = [row for row in csv.reader(csv_path.read_text(encoding="utf-8").splitlines()) if row]
    header, body = rows[0], rows[1:]
    note = ("gpu; one frozen two-round forward over 48 crops; equivalence includes frozen forwards and "
            "backward; 4 cache and 3 train attempts all from newly written script defects; protocol unchanged")
    a2_row = ["A2", f"{upstream:.2f}", f"{train_timing['train_seconds']:.2f}", "", "", "",
              f"{cache_rw:.2f}", "", f"{wall:.2f}", note]
    out_rows = []
    for row in body:
        out_rows.append(a2_row if row and row[0] == "A2" else row)
    assert any(r[0] == "A2" for r in out_rows), "A2 row missing from stage_timing.csv"

    def fmt(field):
        return f'"{field}"' if ("," in field or '"' in field) else field

    lines = [",".join(fmt(f) for f in header)]
    lines += [",".join(fmt(f) for f in row) for row in out_rows]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- execution receipt ----
    protocol_id = json.loads((RUN / "config/locked_run.json").read_text(encoding="utf-8"))["protocol_id"]
    equivalence = json.loads((RUN / "receipts/online_cache_equivalence.json").read_text(encoding="utf-8"))
    cache_audit = json.loads((RUN / "metrics/A2_cache_audit.json").read_text(encoding="utf-8"))
    results = json.loads((RUN / "metrics/A2_native_fit_results.json").read_text(encoding="utf-8"))

    receipt = dict(
        stage="A2_native_head_diagnostic",
        state="native_head_diagnostic_complete",
        protocol_id=protocol_id,
        protocol_changed=False,
        algorithm_objective_data_protocol_unchanged=True,
        authorised_scope="stage A only; 2h absolute GPU budget registered at A2 launch; B/C not started",
        budget=json.loads((RUN / "budget.json").read_text(encoding="utf-8")),
        process_attempts=dict(cache_and_equivalence=4, training=3, additional_diagnostic_runs=2,
                              successful_runs=dict(cache=1, training=1)),
        execution_defects=[
            dict(id=1, where="a2_cache.equivalence_check",
                 defect="4-channel true cost concatenated into 6 channels -> cost_loss shape error",
                 fix="zeros_like(y_raw[:, :1]) + y_raw", affects_protocol=False),
            dict(id=2, where="a2_cache.equivalence_check",
                 defect="foreign set_step_seed triple replayed the frozen forward under a different RNG stream",
                 fix="replay the exact (phase='a2_native_cache', step=plan position) triple used at cache time",
                 affects_protocol=False),
            dict(id=3, where="a2_cache.equivalence_check (GT replacement)",
                 defect="torch.rand_like drawn after set_step_seed consumed global RNG before recurrent_forward",
                 fix="draw the replacement target before seeding", affects_protocol=False),
            dict(id=4, where="a2_train.make_head",
                 defect="LocalAcceptance() used as template -> body.4 out_channels 4 instead of 3",
                 fix="deepcopy the production gate built by gain22.components", affects_protocol=False),
            dict(id=5, where="a2_train.main",
                 defect="CUDA median-with-indices has no deterministic implementation under use_deterministic_algorithms(True)",
                 fix="compute the median on CPU (numerically identical, tiny tensor)", affects_protocol=False),
        ],
        known_pitfalls_checked=[
            "use_deterministic_algorithms True and CUDA reductions (defect 5)",
            "RNG stream shared between seeding and the frozen two-round forward (defects 2, 3)",
            "source search path: ROOT assertion resolved to the new audit run for both A2 scripts",
            "GT isolation: replacement changes labels only, deployable features bitwise unchanged",
        ],
        cache_equivalence=dict(passed=equivalence["passed"],
                               bitwise_equal_not_merely_allclose=True,
                               per_crop_bitwise_equal=[r["raw_cost_bitwise_equal"] for r in equivalence["per_crop"]],
                               tolerance=equivalence["tolerance"],
                               rng_sensitivity=equivalence["rng_sensitivity"]),
        cache=dict(npz_sha256=cache_audit["cache_npz_sha256"], bytes=cache_audit["cache_bytes"],
                   crops=cache_audit["plan_crops"], per_role=cache_audit["per_role"]),
        heads=dict(shared_init_sha256=results["shared_initialisation"]["sha256"],
                   source_final_sha256=results["shared_initialisation"]["source_final_sha256"],
                   input_channels=results["native_head_architecture"]["input_channels"],
                   output_channels=results["native_head_architecture"]["output_channels"],
                   state_keys_equal=results["native_head_architecture"]["state_keys_equal"]),
        outcome=dict(
            mixed_final_pure_fit=results["heads"]["mixed"]["train"]["pure_fit"],
            pure_final_pure_fit=results["heads"]["pure"]["train"]["pure_fit"],
            mixed_train_cost_mae_raw=results["heads"]["mixed"]["train"]["cost_mae_raw"],
            pure_train_cost_mae_raw=results["heads"]["pure"]["train"]["cost_mae_raw"],
            zero_baseline_mae_raw=results["heads"]["mixed"]["train"]["zero_baseline_mae_raw"],
            bias_role_mae_raw=results["heads"]["mixed"]["cross_scene_bias"]["cost_mae_raw"],
            bias_role_zero_baseline_raw=results["heads"]["mixed"]["cross_scene_bias"]["zero_baseline_mae_raw"],
            safety_role_mae_raw=results["heads"]["mixed"]["cross_scene_safety"]["cost_mae_raw"],
            safety_role_zero_baseline_raw=results["heads"]["mixed"]["cross_scene_safety"]["zero_baseline_mae_raw"],
            chosen_margin=results["heads"]["mixed"]["screening"]["chosen"]["margin"],
            chosen_coverage=results["heads"]["mixed"]["screening"]["chosen"]["coverage"],
            safe=results["heads"]["mixed"]["screening"]["safe"],
            useful=results["heads"]["mixed"]["screening"]["useful"],
            accepted_harmful_rate=results["heads"]["mixed"]["screening"]["accepted_harmful_rate"],
            gt_oracle_upper_bound_raw=results["heads"]["mixed"]["screening"]["reference_accept_all"]["mean_mse_improvement"],
            expectation_term_helps=False,
            cross_scene_transfer_beats_zero_baseline=False,
        ),
        timing=dict(cache_seconds=cache_timing["total_seconds"], cache_wall_clock_seconds=cache_wall,
                    train_seconds=train_timing["total_seconds"], train_wall_clock_seconds=train_wall,
                    stage_wall_clock_seconds=wall,
                    note="wall clock is monotonic per process; unrelated debug cycles are not charged"),
        test_used=False, validation_used_to_select_any_rule=False,
        source_diagnostic_872_rerun=False,
        created_unix=time.time(),
    )
    json_save(RUN / "receipts/A2_execution.json", receipt)
    status("native_head_diagnostic_complete", equivalence_passed=equivalence["passed"],
           stage_wall_clock_seconds=round(wall, 2))
    print(json.dumps(dict(finalised=True, stage_wall_clock_seconds=round(wall, 2),
                          upstream_seconds=round(upstream, 2), cache_rw_seconds=round(cache_rw, 2))), flush=True)


if __name__ == "__main__":
    main()
