"""NoMAD LongBench runner: device-aware; dense-prefill + NoMAD-decode by default."""
import os
import sys
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import load_longbench_dataset
from .config import load_json_config
from .templates import infer_model_family, format_longbench_prompt, post_process

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nomad_llama  # noqa: E402

DATASET2PROMPT = load_json_config("dataset2prompt.json")
DATASET2MAXLEN = load_json_config("dataset2maxlen.json")


def load_model(model_path, device="cpu", dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, attn_implementation="eager")
    model.to(device)
    model.eval()
    return model, tok


def attach_method(model, method, codebooks_dir=None, use_8bit=True, device="cpu", dense_prefill=True):
    if method == "baseline":
        model.config._attn_implementation = "eager"
    elif method == "nomad":
        cfg = model.config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        cent = nomad_llama.load_cent_arr(codebooks_dir, cfg.num_hidden_layers, cfg.num_key_value_heads, head_dim)
        nomad_llama.register_nomad(model, cent, use_8bit=use_8bit, device=device, dense_prefill=dense_prefill)
    else:
        raise ValueError(f"unknown method: {method}")


def _eos_ids(tok, model, dataset):
    eos = model.config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else ([eos] if eos is not None else [tok.eos_token_id])
    if dataset == "samsum":
        nl = tok.encode("\n", add_special_tokens=False)
        if nl:
            eos = eos + [nl[-1]]
    return eos


def generate_dataset(model, tok, dataset, data, max_gen, max_model_len, model_family, out_path, device="cpu"):
    eos = _eos_ids(tok, model, dataset)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    rows = []
    for obj in data:
        prompt = DATASET2PROMPT[dataset].format(**obj)
        prompt = format_longbench_prompt(tok, prompt, dataset=dataset, model_family=model_family, max_model_len=max_model_len)
        inputs = tok(prompt, truncation=False, return_tensors="pt").to(device)
        ctx = inputs.input_ids.shape[-1]
        gk = dict(max_new_tokens=max_gen, num_beams=1, do_sample=False,
                  eos_token_id=eos, pad_token_id=pad, use_cache=True)
        if dataset == "samsum":
            gk["min_new_tokens"] = 1
        with torch.no_grad():
            out = model.generate(**inputs, **gk)[0]
        pred = post_process(tok.decode(out[ctx:], skip_special_tokens=True), model_family)
        rows.append({"pred": pred, "answers": obj.get("answers"),
                     "all_classes": obj.get("all_classes"), "length": obj.get("length")})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def run(model_path, method, codebooks_dir, datasets, data_root, out_dir, max_samples,
        max_model_len, use_8bit=True, device="cpu", dense_prefill=True):
    model, tok = load_model(model_path, device)
    attach_method(model, method, codebooks_dir, use_8bit, device, dense_prefill)
    family = infer_model_family(model_path)
    mode = ("dense-prefill+nomad-decode" if dense_prefill else "full-nomad") if method == "nomad" else "baseline"
    print(f"[runner] method={method} mode={mode} family={family} device={device} "
          f"attn={model.config._attn_implementation} max_model_len={max_model_len}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for ds in datasets:
        data = load_longbench_dataset(ds, data_root=data_root)
        if max_samples and max_samples > 0:
            data = data.select(range(min(max_samples, len(data))))
        out_path = os.path.join(out_dir, f"{ds}.jsonl")
        t = time.time()
        print(f"[{method}] {ds}: {len(data)} samples, max_gen={DATASET2MAXLEN[ds]}", flush=True)
        generate_dataset(model, tok, ds, data, DATASET2MAXLEN[ds], max_model_len, family, out_path, device)
        print(f"[{method}] {ds}: done in {time.time()-t:.0f}s", flush=True)
        report[ds] = out_path
    return report
