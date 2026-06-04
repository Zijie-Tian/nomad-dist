"""Final NoMAD-Attention PPL reproduction on StableLM-3B-4E1T.

Reproduces the paper's perplexity experiment: baseline (original attention) vs
NoMAD-Attention with per-(layer,head) PQ key codebooks. Uses
attn_implementation='eager' so the class-level monkey-patch actually intercepts
attention (the default SDPA subclass would bypass a class-level patch).

Supports the paper's Section 3.2 dynamic 8-bit LUT quantization (enabled by
default; pass --no_8bit for the un-quantized float-LUT variant).

Pipeline context:
  save_keys.py  ->  (repo root) learn_codebooks.py  ->  THIS SCRIPT
Run from the repository root, e.g.:
  python experiments/nomad_ppl_final.py
  python experiments/nomad_ppl_final.py --no_8bit --max_chunks 20
"""
import os, math, argparse, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.stablelm.modeling_stablelm import apply_rotary_pos_emb, repeat_kv, StableLmAttention
from tqdm import tqdm
import faiss

call_count = [0]


def load_cent_arr(assets_dir):
    """Load 1024 per-(layer,head) PQ codebooks (FAISS IndexPQFastScan) -> (32,32,80,16)."""
    codebooks = {}
    for i in range(1024):
        path = f'{assets_dir}/{i}.index'
        if not os.path.exists(path):
            continue
        idx = faiss.read_index(path)
        pq = idx.pq
        cent = faiss.vector_float_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub).squeeze(-1)
        codebooks[(i // 32, i % 32)] = cent.astype(np.float32)
    cent_arr = np.zeros((32, 32, 80, 16), dtype=np.float32)
    for (l, h), c in codebooks.items():
        cent_arr[l, h] = c
    return cent_arr


def nomad_attention_scores(query, key_codes, centroids, use_8bit=True):
    """Approximate q.K^T via PQ lookup tables (asymmetric: full-precision query, quantized keys)."""
    b, h, tq, d = query.shape
    _, _, tk, M = key_codes.shape
    q_sub = query.reshape(b, h, tq, M, 1)
    cent = torch.from_numpy(centroids).to(query.device, query.dtype).unsqueeze(0).unsqueeze(2).unsqueeze(-1)
    lut = (q_sub.unsqueeze(4) * cent).sum(dim=-1)  # (b,h,tq,M,16) query-centroid dot products
    if use_8bit:
        # Paper Section 3.2: dynamic per-(query, sub-quantizer) 8-bit LUT quantization.
        dp_min = lut.min(dim=4, keepdim=True)[0]
        dp_max = lut.max(dim=4, keepdim=True)[0]
        scale = (dp_max - dp_min) / 255.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        lut = torch.floor((lut - dp_min) / scale).clamp(0, 255).float() * scale + dp_min
    scores = torch.zeros(b, h, tq, tk, device=query.device, dtype=query.dtype)
    for j in range(tk):  # loop over key positions (tk ~ 512, cheap in Python)
        cj = key_codes[:, :, j, :].unsqueeze(2).unsqueeze(-1)
        gathered = torch.gather(lut, dim=4, index=cj.expand(-1, -1, tq, -1, -1))
        scores[:, :, :, j] = gathered.sum(dim=3).squeeze(-1)
    return scores


def make_nomad_fwd(cent_arr, use_8bit=True):
    """Class-level StableLmAttention.forward replacement using NoMAD score lookup."""
    def nomad_class_fwd(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                        output_attentions=False, use_cache=False, cache_position=None,
                        position_embeddings=None, **kwargs):
        call_count[0] += 1
        bsz, q_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if self.qk_layernorm:
            query = self.q_layernorm(query)
            key = self.k_layernorm(key)
        cos, sin = position_embeddings
        qr, qp = query[..., :self.rotary_ndims], query[..., self.rotary_ndims:]
        kr, kp = key[..., :self.rotary_ndims], key[..., self.rotary_ndims:]
        qr, kr = apply_rotary_pos_emb(qr, kr, cos, sin)
        query = torch.cat((qr, qp), dim=-1)
        key = torch.cat((kr, kp), dim=-1)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": self.rotary_ndims, "cache_position": cache_position}
            key, value = past_key_values.update(key, value, self.layer_idx, cache_kwargs)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        b, h, seq_k, d = key.shape
        # Quantize keys to PQ codes (L2-nearest centroid per sub-quantizer).
        key_sub = key.reshape(b, h, seq_k, d, 1).unsqueeze(4)
        cent = torch.from_numpy(cent_arr[self.layer_idx]).to(key.device, key.dtype).unsqueeze(0).unsqueeze(2).unsqueeze(-1)
        dist = (key_sub - cent).pow(2).sum(dim=-1)
        key_codes = dist.argmin(dim=-1)
        scores = nomad_attention_scores(query, key_codes, cent_arr[self.layer_idx], use_8bit=use_8bit)
        scores = scores / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores += attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = torch.softmax(scores, dtype=torch.float32, dim=-1).to(query.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights
    return nomad_class_fwd


def compute_ppl(model, tokenizer, text_path, max_chunks=100):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    seq_len = input_ids.size(1)
    nlls, total_trg_len, prev_end_loc = [], 0, 0
    stride, max_length = 512, 512
    chunks = list(range(0, seq_len, stride))[:max_chunks]
    for begin_loc in tqdm(chunks, desc="PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        chunk = input_ids[:, begin_loc:end_loc]
        target = chunk.clone()
        target[:, :-trg_len] = -100
        with torch.no_grad():
            loss = model(chunk, labels=target).loss * trg_len
        nlls.append(loss)
        total_trg_len += trg_len
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return torch.exp(torch.stack(nlls).sum() / total_trg_len).item()


def load_model(model_id):
    m = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map='cpu',
        trust_remote_code=True, attn_implementation='eager')
    m.eval()
    return m


def run_nomad(model_id, tokenizer, data, cb_dir, use_8bit, max_chunks, ppl_base, label):
    print(f"=== NoMAD ({label}, eager, 8bit={use_8bit}) ===")
    orig = StableLmAttention.forward
    StableLmAttention.forward = make_nomad_fwd(load_cent_arr(cb_dir), use_8bit=use_8bit)
    call_count[0] = 0
    try:
        ppl = compute_ppl(load_model(model_id), tokenizer, data, max_chunks=max_chunks)
    finally:
        StableLmAttention.forward = orig
    print(f"NoMAD PPL: {ppl:.4f} | forward calls: {call_count[0]} | "
          f"relative change: {((ppl - ppl_base) / ppl_base * 100):+.2f}%\n")
    return ppl


def main():
    p = argparse.ArgumentParser(description="NoMAD-Attention PPL reproduction (StableLM-3B-4E1T)")
    p.add_argument('--model', default='stabilityai/stablelm-3b-4e1t')
    p.add_argument('--data', default='data/wikitext-2-raw/wiki.test.raw')
    p.add_argument('--new_cb', default='assets/stablelm-3b-dsub1-transformers', help='codebooks trained on transformers keys')
    p.add_argument('--old_cb', default='assets/stablelm-3b-dsub1', help='repo-provided codebooks')
    p.add_argument('--max_chunks', type=int, default=100)
    p.add_argument('--no_8bit', dest='use_8bit', action='store_false',
                   help='disable paper Sec 3.2 8-bit LUT quantization (use float LUT)')
    p.set_defaults(use_8bit=True)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"[CONFIG] 8-bit LUT quantization = {args.use_8bit} | max_chunks = {args.max_chunks}\n")

    print("=== 1. Baseline (eager) ===")
    ppl_base = compute_ppl(load_model(args.model), tokenizer, args.data, max_chunks=args.max_chunks)
    print(f"Baseline PPL: {ppl_base:.4f}\n")

    ppl_new = run_nomad(args.model, tokenizer, args.data, args.new_cb, args.use_8bit, args.max_chunks, ppl_base, "new codebooks")
    ppl_old = run_nomad(args.model, tokenizer, args.data, args.old_cb, args.use_8bit, args.max_chunks, ppl_base, "old codebooks")

    print("=== Summary ===")
    print(f"Baseline:     {ppl_base:.4f}")
    print(f"NoMAD (new):  {ppl_new:.4f}  ({((ppl_new - ppl_base) / ppl_base * 100):+.2f}%)")
    print(f"NoMAD (old):  {ppl_old:.4f}  ({((ppl_old - ppl_base) / ppl_base * 100):+.2f}%)")


if __name__ == "__main__":
    main()
