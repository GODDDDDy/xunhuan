"""CPU tests for figure availability and concurrent bookkeeping archive snapshots."""
import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load("eval_recovery", root / "recovery/evaluate_existing_final.py")
source = evaluation.patched_source((root / "run_evaluation.py").read_text(encoding="utf-8"))
namespace = {"__name__": "test_only", "__file__": str(root / "recovery/evaluate_existing_final.py")}
exec(compile(source, "patched_eval_test", "exec"), namespace)
rows = []
for scene in ("01", "07", "22", "30", "36", "41", "55", "56"):
    roles = ("easy",) if scene == "55" else ("hard",) if scene == "56" else ("easy", "hard")
    for difficulty in roles:
        for frame in range(12):
            rows.append(SimpleNamespace(scene=scene, difficulty=difficulty, split="validation", frame=frame,
                                        sample_id=f"{scene}/{difficulty}/{frame:03}"))
selected = namespace["example_selection"](rows)
assert len(set(selected)) == len(selected) == 16
assert all(x in {r.sample_id for r in rows} for x in selected)
assert sum(x.startswith("55/easy/") for x in selected) == 2
assert sum(x.startswith("56/hard/") for x in selected) == 2
assert namespace["example_selection"](list(reversed(rows))) == selected
assert '/ record["sample_id"].replace("/", "__")' in source
assert '/ manifest_row.sample_id.replace("/", "__")' in source

archive = load("archive_recovery", root / "recovery/archive_live_snapshot.py")
namespace = {"__name__": "test_only", "__file__": str(root / "recovery/archive_live_snapshot.py")}
exec(compile(archive.patched_source((root / "archive.py").read_text(encoding="utf-8")), "patched_archive_test", "exec"), namespace)
fixture = root / "diagnostics" / ("archive_snapshot_fixture_" + str(time.time_ns()))
fixture.mkdir(parents=True)
(fixture / "config").mkdir()
(fixture / "payload").mkdir()
(fixture / "status.json").write_text('{"stage":"failed"}', encoding="utf-8")
(fixture / "config/locked_run.json").write_text('{"arms":["meta"]}', encoding="utf-8")
for n in range(60):
    (fixture / "payload" / f"{n}.txt").write_text("static proof\n" * 100, encoding="utf-8")
watcher = fixture / "watchdog.json"
watcher.write_text('{"counter":0}', encoding="utf-8")
stop = threading.Event()


def mutate():
    n = 0
    while not stop.is_set():
        n += 1
        temporary = fixture / "watchdog.json.tmp"
        temporary.write_text(json.dumps({"counter": n}), encoding="utf-8")
        temporary.replace(watcher)
        time.sleep(.001)


thread = threading.Thread(target=mutate)
thread.start()
try:
    result = namespace["archive"](fixture)
finally:
    stop.set()
    thread.join()
assert result["archive_members_verified"] and not result["completed"]
print(json.dumps({"CPU_only": True, "examples_unique": 16, "actual_roles_preserved": True,
                  "selection_order_independent": True, "live_watchdog_snapshot_verified": True,
                  "training_or_GPU_used": False, "fixture": str(fixture)}))
