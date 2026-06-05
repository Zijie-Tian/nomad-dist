"""CLI: score LongBench prediction jsonl files with official per-task metrics."""
import argparse
import glob
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longbench import scorer as S


def score_file(path, dataset):
    preds, answers, allc = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            preds.append(r["pred"])
            answers.append(r["answers"])
            allc.append(r.get("all_classes"))
    all_classes = allc[-1] if allc else None
    return S.scorer(dataset, preds, answers, all_classes)


def main():
    ap = argparse.ArgumentParser(description="Score LongBench predictions")
    ap.add_argument("--pred-dir", required=True)
    a = ap.parse_args()
    res = {}
    for p in sorted(glob.glob(os.path.join(a.pred_dir, "*.jsonl"))):
        ds = os.path.basename(p)[:-6]
        try:
            res[ds] = score_file(p, ds)
        except Exception as e:
            res[ds] = f"ERR:{e}"
    with open(os.path.join(a.pred_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
