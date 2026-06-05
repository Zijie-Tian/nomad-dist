"""NoMAD-Attention for LLaMA-family models via the transformers attention_interface.

Registers a 'nomad' attention function. Two modes (per layer flag):
  - dense_prefill=True (default): prefill (q_len>1) uses exact dense attention;
    decode (q_len==1) uses NoMAD PQ lookup. Matches a "GPU-prefill + NoMAD-decode"
    deployment and keeps prefill exact.
  - dense_prefill=False: prefill and decode both use NoMAD lookup (paper's pure NoMAD).

Codebooks are per (layer, kv_head); GQA codes/centroids are repeat-interleaved to
align query heads (query head h -> kv head h // num_key_value_groups). The PQ
lookup is vectorized via a one-hot matmul that is numerically identical to
per-key table lookup. Works on CPU or GPU (codebooks moved to model device).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import repeat_kv


def load_cent_arr(assets_dir, n_layers, n_kv_heads, head_dim, d_sub=1):
    """Load per-(layer,kv_head) PQ codebooks -> (n_layers, n_kv_heads, M, 16) float32."""
    M = head_dim // d_sub
    cent_arr = np.zeros((n_layers, n_kv_heads, M, 16), dtype=np.float32)
    missing = 0
    for i in range(n_layers * n_kv_heads):
        path = f"{assets_dir}/{i}.index"
        if not os.path.exists(path):
            missing += 1
            continue
        pq = faiss.read_index(path).pq
        cent = faiss.vector_float_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub).squeeze(-1)
        cent_arr[i // n_kv_heads, i % n_kv_heads] = cent.astype(np.float32)
    if missing:
        print(f"[nomad] WARNING: {missing}/{n_layers * n_kv_heads} codebooks missing in {assets_dir}")
    return cent_arr


def _nomad_scores(query, key, cent, use_8bit):
    """query (b,H,tq,d); key (b,Hkv,tk,d); cent (Hkv,M,16) tensor|ndarray -> (b,H,tq,tk)."""
    b, H, tq, d = query.shape
    _, Hkv, tk, _ = key.shape
    groups = H // Hkv
    if not torch.is_tensor(cent):
        cent = torch.from_numpy(cent)
    cent = cent.to(query.device, query.dtype)                          # (Hkv,M,16)
    M = cent.shape[1]
    key_sub = key.reshape(b, Hkv, tk, M, 1)
    dist = (key_sub - cent.unsqueeze(0).unsqueeze(2)).pow(2)           # (b,Hkv,tk,M,16)
    codes = dist.argmin(-1)                                            # (b,Hkv,tk,M)
    codes = codes.repeat_interleave(groups, dim=1)                    # (b,H,tk,M)
    cent_H = cent.repeat_interleave(groups, dim=0)                    # (H,M,16)
    q_sub = query.reshape(b, H, tq, M, 1)
    lut = q_sub * cent_H.unsqueeze(0).unsqueeze(2)                    # (b,H,tq,M,16)
    if use_8bit:
        dp_min = lut.amin(-1, keepdim=True)
        dp_max = lut.amax(-1, keepdim=True)
        scale = (dp_max - dp_min) / 255.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        lut = torch.floor((lut - dp_min) / scale).clamp(0, 255) * scale + dp_min
    onehot = F.one_hot(codes, 16).to(lut.dtype)                      # (b,H,tk,M,16)
    return torch.matmul(lut.reshape(b, H, tq, M * 16),
                        onehot.reshape(b, H, tk, M * 16).transpose(2, 3))


def _mask_softmax_value(module, scores, value, attention_mask, query, tq):
    tk = scores.shape[-1]
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :tk]
    elif tq > 1:
        scores = scores + torch.triu(
            torch.full((tq, tk), float("-inf"), dtype=scores.dtype, device=scores.device),
            diagonal=tk - tq + 1)
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(attn, repeat_kv(value, module.num_key_value_groups)).transpose(1, 2).contiguous()
    return out, attn


def nomad_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    if scaling is None:
        scaling = module.head_dim ** -0.5
    tq = query.shape[2]
    if getattr(module, "_nomad_dense_prefill", True) and tq > 1:
        # PREFILL: exact dense attention (GPU-friendly, no PQ error)
        key_h = repeat_kv(key, module.num_key_value_groups)
        scores = torch.matmul(query, key_h.transpose(2, 3)) * scaling
    else:
        # DECODE (or pure-NoMAD prefill): PQ lookup
        scores = _nomad_scores(query, key, module._nomad_cent, getattr(module, "_nomad_use_8bit", True)) * scaling
    return _mask_softmax_value(module, scores, value, attention_mask, query, tq)


def make_capture_forward(store):
    """attention_interface that records post-rotary keys (per kv_head) then runs correct causal attention."""
    def capture_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        store.setdefault(module.layer_idx, []).append(key.detach().to(torch.float32).cpu())
        if scaling is None:
            scaling = module.head_dim ** -0.5
        key_h = repeat_kv(key, module.num_key_value_groups)
        scores = torch.matmul(query, key_h.transpose(2, 3)) * scaling
        return _mask_softmax_value(module, scores, value, attention_mask, query, query.shape[2])
    return capture_attention_forward


def register_nomad(model, cent_arr, use_8bit=True, device=None, dense_prefill=True):
    """Install NoMAD attention. device: move codebooks to this device (e.g. 'cuda')."""
    ALL_ATTENTION_FUNCTIONS["nomad"] = nomad_attention_forward
    for i, layer in enumerate(model.model.layers):
        c = torch.from_numpy(cent_arr[i])
        if device is not None:
            c = c.to(device)
        layer.self_attn._nomad_cent = c
        layer.self_attn._nomad_use_8bit = use_8bit
        layer.self_attn._nomad_dense_prefill = dense_prefill
    model.config._attn_implementation = "nomad"
    return model


def register_capture(model, store):
    ALL_ATTENTION_FUNCTIONS["nomad_capture"] = make_capture_forward(store)
    model.config._attn_implementation = "nomad_capture"
    return model
