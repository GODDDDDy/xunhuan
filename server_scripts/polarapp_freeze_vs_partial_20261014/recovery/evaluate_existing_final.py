"""Only repair deterministic figure selection/paths; frozen evaluator stays intact."""
import hashlib
import sys
from pathlib import Path

SELECTION = '''def example_selection(manifest):
    """Two fixed frames per scene; preserve actual available difficulty labels."""
    strata = defaultdict(list)
    for row in manifest:
        if row.split == "validation":
            strata[(row.scene, row.difficulty)].append(row)
    scenes = sorted({scene for scene, _ in strata})
    if len(scenes) != 8:
        raise ValueError(f"Expected eight validation scenes, found {scenes}")
    result = []
    for scene in scenes:
        groups = [sorted(strata[(scene, d)], key=lambda r: (r.frame, r.sample_id))
                  for d in ("easy", "hard") if strata[(scene, d)]]
        if len(groups) == 2:
            result.extend(group[len(group)//2].sample_id for group in groups)
        elif len(groups) == 1 and len(groups[0]) >= 2:
            values = groups[0]
            result.extend(values[i].sample_id for i in (len(values)//3, 2*len(values)//3))
        else:
            raise ValueError(f"Cannot select two actual frames in scene {scene}")
    if len(result) != 16 or len(set(result)) != 16:
        raise ValueError("Fixed examples must contain 16 unique validation frames")
    return result
'''


def patched_source(source):
    start = source.index("def example_selection(manifest):")
    end = source.index("\n\ndef checkpoint_adapters", start)
    source = source[:start] + SELECTION + source[end:]
    edits = {
        'destination = root / "downstream/examples" / record["scene"] / record["difficulty"]':
        'destination = root / "downstream/examples" / record["scene"] / record["difficulty"] / record["sample_id"].replace("/", "__")',
        'example_path = root / "downstream/examples" / manifest_row.scene / manifest_row.difficulty':
        'example_path = root / "downstream/examples" / manifest_row.scene / manifest_row.difficulty / manifest_row.sample_id.replace("/", "__")',
        '"example_selection": "middle frame per scene/difficulty; frozen before reading GT",':
        '"example_selection": "one middle per existing difficulty, or 1/3 and 2/3 frames when only one exists; 16 unique; no GT selection",',
        '"evaluation_code_sha256": sha256_file(Path(__file__)),':
        '"evaluation_code_sha256": sha256_file(Path(__file__)), "frozen_base_evaluator_sha256": sha256_file(root / "scripts/run_evaluation.py"), "execution_recovery": "figure availability and unique sample paths only; model/metrics unchanged",',
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise RuntimeError(f"Unexpected frozen evaluator layout: {old}")
        source = source.replace(old, new)
    return source


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from runtime import read_locked_config
    config = read_locked_config(root)
    base = root / "scripts/run_evaluation.py"
    if hashlib.sha256(base.read_bytes()).hexdigest() != config["code_sha256"]["scripts/run_evaluation.py"]:
        raise RuntimeError("Frozen base evaluator changed")
    namespace = {"__name__": "__main__", "__file__": __file__}
    exec(compile(patched_source(base.read_text(encoding="utf-8")), str(base), "exec"), namespace)
