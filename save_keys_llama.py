"""Save post-rotary attention keys from LLaMA-3.2 for NoMAD codebook learning.

Uses the capture attention_interface (records per-kv_head keys under GQA) and
writes one FAISS flat index per (layer, kv_head):
  idx = layer * n_kv_heads + kv_head ; file r0_i{idx}.index
This filename matches learn_codebooks.py's expected r{run}_i{i}.index layout.
"""
import os
import argparse
import numpy as np
import torch
import faiss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import nomad_llama


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/data/tzj/models/Llama-3.2-1B-Instruct")
    ap.add_argument("--data", default="data/wikitext-2-raw/wiki.train.raw")
    ap.add_argument("--out_dir", default="assets/llama-3.2-1b-keys")
    ap.add_argument("--max_samples", type=int, default=100)
    ap.add_argument("--chunk_size", type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    print(f"[info] layers={n_layers} kv_heads={n_kv} head_dim={head_dim}")

    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    limit = min(len(ids), args.max_samples * args.chunk_size)
    chunks = [ids[i:i + args.chunk_size].unsqueeze(0) for i in range(0, limit, args.chunk_size)]
    print(f"[info] {len(chunks)} chunks of {args.chunk_size} tokens")

    os.makedirs(args.out_dir, exist_ok=True)
    indices = {i: faiss.IndexFlatL2(head_dim) for i in range(n_layers * n_kv)}

    store = {}
    nomad_llama.register_capture(model, store)
    with torch.no_grad():
        for chunk in tqdm(chunks, desc="capture keys"):
            store.clear()
            model(chunk, use_cache=False)
            for layer_idx, keys in store.items():
                key = keys[0][0]  # (n_kv, tk, head_dim)
                for kv in range(n_kv):
                    indices[layer_idx * n_kv + kv].add(key[kv].numpy().astype(np.float32))

    for i, idx in indices.items():
        faiss.write_index(idx, os.path.join(args.out_dir, f"r0_i{i}.index"))
    print(f"[done] wrote {len(indices)} indices to {args.out_dir}; "
          f"idx0 ntotal={indices[0].ntotal}, idx{n_layers*n_kv-1} ntotal={indices[n_layers*n_kv-1].ntotal}")


if __name__ == "__main__":
    main()
