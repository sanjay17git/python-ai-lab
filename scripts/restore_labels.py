
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
tuning_path = root / "eval" / "tuning.json"
baseline_path = root / "eval" / "eval_set.json"

with baseline_path.open("r", encoding="utf-8") as f:
    baseline = json.load(f)

with tuning_path.open("r", encoding="utf-8") as f:
    tuning = json.load(f)

baseline_by_id = {q["id"]: q for q in baseline}

for q in tuning:
    if q["id"] in baseline_by_id:
        original = baseline_by_id[q["id"]]
        q["relevant_chunks"] = original["relevant_chunks"]
        q["source_passage"] = original.get("source_passage", "")

with tuning_path.open("w", encoding="utf-8") as f:
    json.dump(tuning, f, indent=2, ensure_ascii=False)

print("Restored original labels for q001-q010.")
print([
    (q["id"], q["relevant_chunks"])
    for q in tuning if q["id"] in baseline_by_id
])
