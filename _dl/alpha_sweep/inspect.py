import json
from pathlib import Path

d = json.loads(Path(r"F:/WORK/projects/pidsr_recurrent_joint_k2/_dl/alpha_sweep/alpha_sweep_result.json").read_text(encoding="utf-8"))
print("elapsed_seconds:", d["elapsed_seconds"])
print("n_tasknet_calls:", d["n_tasknet_calls"])
print("per_image_seconds_est:", d["elapsed_seconds"] / (24 * 2))

for g in ("no_feedback", "feedback"):
    s = d["groups"][g]
    print()
    print("==== %s  fs=%s  n=%d" % (g, s["feedback_scale"], s["n_images"]))
    print("clamp r1:", json.dumps(s["clamp_round1"]))
    print("clamp r2:", json.dumps(s["clamp_round2"]))
    T1 = s["T1"]
    print("T1 psnr img=%.6f sm=%.6f | ssim img=%.8f sm=%.8f | loss=%.6f"
          % (T1["psnr"]["image_mean"], T1["psnr"]["scene_macro"],
             T1["ssim"]["image_mean"], T1["ssim"]["scene_macro"],
             T1["loss_total_image_mean"]))
    print("T1 loss parts:", json.dumps(T1["loss_components_image_mean"]))
    T2 = s["T2"]
    print("T2 d_psnr img=%+0.6f sm=%+0.6f | d_ssim img=%+0.8f sm=%+0.8f | d_loss=%+0.6f"
          % (T2["delta_psnr_vs_T1"]["image_mean"], T2["delta_psnr_vs_T1"]["scene_macro"],
             T2["delta_ssim_vs_T1"]["image_mean"], T2["delta_ssim_vs_T1"]["scene_macro"],
             T2["delta_loss_total_vs_T1_image_mean"]))
    for a, v in s["alphas"].items():
        c = v["image_counts_vs_T1"]
        print("  a=%-5s dPSNR img %+0.6f sm %+0.6f | dSSIM img %+0.8f sm %+0.8f "
              "| imp %d/%d deg %d | dLoss %+0.6f | in[%.4f,%.4f]"
              % (a, v["delta_psnr_vs_T1"]["image_mean"],
                 v["delta_psnr_vs_T1"]["scene_macro"],
                 v["delta_ssim_vs_T1"]["image_mean"],
                 v["delta_ssim_vs_T1"]["scene_macro"],
                 c["improved"], c["total"], c["degraded"],
                 v["delta_loss_total_vs_T1_image_mean"],
                 v["input_range_over_images"]["min"],
                 v["input_range_over_images"]["max"]))
        print("        loss parts:", json.dumps(v["loss_components_image_mean"]))
    print("  endpoint:", json.dumps(s["endpoint_integrity"]))
    print("  per-scene dPSNR T2:", json.dumps(T2["delta_psnr_vs_T1"]["per_scene"]))
    for a in ("0.25", "0.5", "0.75"):
        print("  per-scene dPSNR a=%s:" % a,
              json.dumps({k: round(v_, 4) for k, v_ in
                          s["alphas"][a]["delta_psnr_vs_T1"]["per_scene"].items()}))

# difficulty split
print()
print("=== difficulty composition check ===")
for g in ("no_feedback", "feedback"):
    rows = d["groups"][g]["per_image"]
    from collections import Counter
    print(g, Counter(r["difficulty"] for r in rows))
