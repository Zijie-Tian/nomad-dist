"""CLI: generate LongBench predictions for LLaMA-3.2 with baseline or NoMAD attention."""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longbench.runner import run


def main():
    ap = argparse.ArgumentParser(description="LongBench eval (LLaMA-3.2, baseline vs NoMAD)")
    ap.add_argument("--model", default="/mnt/data/tzj/models/Llama-3.2-1B-Instruct")
    ap.add_argument("--method", choices=["baseline", "nomad"], required=True)
    ap.add_argument("--codebooks", default="assets/llama-3.2-1b-dsub1")
    ap.add_argument("--datasets", default="trec,samsum,passage_count")
    ap.add_argument("--data-root", default="/mnt/data/tzj/data/LongBench")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-samples", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--full-nomad", dest="dense_prefill", action="store_false", help="NoMAD for prefill too (default: dense prefill + NoMAD decode)")
    ap.set_defaults(dense_prefill=True)
    ap.add_argument("--no-8bit", dest="use_8bit", action="store_false")
    ap.set_defaults(use_8bit=True)
    a = ap.parse_args()
    datasets = [d for d in a.datasets.split(",") if d]
    out_dir = a.output_dir or f"longbench_out/llama32-1b-instruct_{a.method}/pred"
    rep = run(a.model, a.method, a.codebooks, datasets, a.data_root, out_dir,
              a.max_samples, a.max_model_len, a.use_8bit, device=a.device, dense_prefill=a.dense_prefill)
    print(json.dumps({"method": a.method, "output_dir": out_dir, "datasets": list(rep)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
