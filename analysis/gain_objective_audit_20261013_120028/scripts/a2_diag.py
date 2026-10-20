"""A2 diagnostic: piece-wise online/cache comparison + RNG sensitivity."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(HERE))

import gain22  # noqa: E402
import study  # noqa: E402
from acceptance import resize  # noqa: E402
from common import guard_gpu  # noqa: E402

from a2_common import build_head_inputs  # noqa: E402

spec = importlib.util.spec_from_file_location("a2c", HERE / "a2_cache.py")
a2c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a2c)


def pieces_online(model, out, batch):
    a, p = out.t1.detach().float(), out.ungated_t2.detach().float()
    d = p - a
    size = F.avg_pool2d(batch["official_rgb"].detach().float(), 4).shape[-2:]
    mp = F.avg_pool2d(d, 4)
    ab = F.avg_pool2d(d.abs(), 4)
    return {
        "img": F.avg_pool2d(batch["official_rgb"].detach().float(), 4),
        "t1p": F.avg_pool2d(a, 4),
        "Up": F.avg_pool2d(p, 4),
        "dp": mp,
        "polp": F.avg_pool2d(study.polarization_summary(batch["polar"]).detach().float(), 4),
        "err": resize(out.error_scale.detach(), size),
        "conf": resize(out.confidence_map.detach(), size),
        "proj": model.action_gate.feature_projection(resize(out.judgement_feature.detach(), size)),
        "state": resize(out.judgement_state.detach(), size),
        "dsq": F.avg_pool2d(d.square(), 4),
        "dabs": ab,
        "ratio": mp.abs().sum(1, keepdim=True) / (ab.sum(1, keepdim=True) + 1e-8),
    }


def pieces_cached(model, record):
    mp = record["dp"]
    ab = record["dabs"]
    return {
        "img": record["img"], "t1p": record["t1p"], "Up": record["Up"], "dp": mp,
        "polp": record["polp"], "err": record["err"], "conf": record["conf"],
        "proj": model.action_gate.feature_projection(record["feat"]),
        "state": record["state"], "dsq": record["dsq"], "dabs": ab,
        "ratio": mp.abs().sum(1, keepdim=True) / (ab.sum(1, keepdim=True) + 1e-8),
    }


def main():
    cfg = a2c.load_cfg()
    device = torch.device("cuda")
    model, task, final = a2c.build_model(cfg, device)
    roles, by_scene = a2c.role_rows(cfg)
    plan = a2c.build_plan(cfg, roles, by_scene)
    index = json.loads((RUN / "cache/a2_cache_index.json").read_text(encoding="utf-8"))
    payload = np.load(RUN / "cache/a2_spatial_cache.npz")

    for pos in (index["role_positions"]["judgement"][0], index["role_positions"]["judgement"][3]):
        entry = plan[pos]
        batch = a2c.load_crop(roles[entry["role"]], entry, device)
        guard_gpu()

        study.base.set_step_seed(cfg["seed"], "a2_native_cache", pos, batch["sample_id"])
        with torch.no_grad():
            out = study.recurrent_forward(model, task, batch)

        study.base.set_step_seed(cfg["seed"], "a2_equivalence", pos, batch["sample_id"])
        with torch.no_grad():
            out2 = study.recurrent_forward(model, task, batch)

        x_online = model.action_gate.inputs(*gain22.args(out, batch))
        record = {k: torch.from_numpy(payload[k][pos])[None].to(device) for k in payload.keys()}
        x_cached = build_head_inputs(record, model.action_gate)
        po = pieces_online(model, out, batch)
        pc = pieces_cached(model, record)

        print(json.dumps(dict(
            position=pos, sample_id=entry["sample_id"],
            x_online_channels=int(x_online.shape[1]), x_cached_channels=int(x_cached.shape[1]),
            x_online_vs_cached_max_abs=float((x_online - x_cached).abs().max()),
            token_effect_t1_max_abs=float((out.t1 - out2.t1).abs().max()),
            token_effect_t2_max_abs=float((out.ungated_t2 - out2.ungated_t2).abs().max()))), flush=True)

        for name in po:
            a, b = po[name], pc[name]
            print(json.dumps(dict(position=pos, group=name, online_chans=int(a.shape[1]),
                                  cached_chans=int(b.shape[1]),
                                  max_abs=float((a - b).abs().max()),
                                  shapes_match=bool(a.shape == b.shape))), flush=True)


if __name__ == "__main__":
    main()
