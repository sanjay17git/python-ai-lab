import json

with open("eval/tuning.json") as f:
    questions = json.load(f)

answerable = [q for q in questions if q["relevant_chunks"]]
negative   = [q for q in questions if not q["relevant_chunks"]]

print(f"Answerable: {len(answerable)}")
print(f"Negative  : {len(negative)}")
print(f"Unlabeled (not negative): "
      f"{[q['id'] for q in answerable if not q['relevant_chunks']]}")