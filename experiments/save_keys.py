"""Save attention key embeddings from transformers StableLM for NoMAD codebook learning."""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_to_k(k, cos, sin):
    """Apply rotary embedding to key tensor only.
    k: (batch, num_kv_heads, seq, rotary_ndims)
    cos/sin: (batch, seq, rotary_ndims)
    """
    cos = cos.unsqueeze(1)  # (batch, 1, seq, rotary_ndims)
    sin = sin.unsqueeze(1)
    return (k * cos) + (rotate_half(k) * sin)


def main(max_samples=100):
    model_id = "stabilityai/stablelm-3b-4e1t"
    device = "cpu"

    print("[INFO] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    print("[INFO] Loading WikiText-2 train text...")
    with open("data/wikitext-2-raw/wiki.train.raw", "r", encoding="utf-8") as f:
        text = f.read()

    input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    total_tokens = len(input_ids)
    print(f"[INFO] Total tokens: {total_tokens}")

    chunk_size = 512
    chunks = []
    for i in range(0, min(total_tokens, max_samples * chunk_size), chunk_size):
        chunks.append(input_ids[i : i + chunk_size].unsqueeze(0))
    print(f"[INFO] Number of chunks to process: {len(chunks)}")

    num_layers = len(model.model.layers)
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    out_dir = "assets/stablelm-3b-keys-transformers"
    os.makedirs(out_dir, exist_ok=True)

    # One FAISS flat index per layer-head
    indices = {}
    for layer in range(num_layers):
        for head in range(num_heads):
            idx = layer * num_heads + head
            indices[idx] = faiss.IndexFlatL2(head_dim)

    # Monkey-patch self_attn to capture post-rotary keys
    orig_forwards = []
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        orig = attn.forward
        orig_forwards.append(orig)

        def make_hook(lidx, orig_fwd, attn_mod):
            def hooked_fwd(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=None,
                **kwargs,
            ):
                bsz, q_len, _ = hidden_states.size()
                key = (
                    attn_mod.k_proj(hidden_states)
                    .view(bsz, q_len, attn_mod.num_key_value_heads, attn_mod.head_dim)
                    .transpose(1, 2)
                )

                if attn_mod.qk_layernorm:
                    key = attn_mod.k_layernorm(key)

                cos, sin = position_embeddings
                key_rot, key_pass = key[..., : attn_mod.rotary_ndims], key[..., attn_mod.rotary_ndims :]
                key_rot = apply_rotary_to_k(key_rot, cos, sin)
                key = torch.cat((key_rot, key_pass), dim=-1)

                # Save keys to FAISS indices (batch 0 only)
                k_np = key[0].detach().cpu().numpy().astype(np.float32)
                for h in range(attn_mod.num_key_value_heads):
                    idx_faiss = lidx * num_heads + h
                    indices[idx_faiss].add(k_np[h])

                return orig_fwd(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            return hooked_fwd

        attn.forward = make_hook(layer_idx, orig, attn)

    print("[INFO] Running forward passes to collect keys...")
    with torch.no_grad():
        for chunk in tqdm(chunks, desc="Saving keys"):
            chunk = chunk.to(device)
            position_ids = torch.arange(chunk.size(1), device=device).unsqueeze(0)
            model(chunk, position_ids=position_ids)

    # Restore original forwards
    for layer_idx, layer in enumerate(model.model.layers):
        layer.self_attn.forward = orig_forwards[layer_idx]

    # Write indices to disk
    print(f"[INFO] Writing {num_layers * num_heads} FAISS indices to {out_dir}...")
    for layer in range(num_layers):
        for head in range(num_heads):
            idx = layer * num_heads + head
            faiss.write_index(indices[idx], os.path.join(out_dir, f"r0_i{idx}.index"))

    # Sanity check
    print("[INFO] Sanity check: reading back a few indices...")
    for idx in [0, 1, 1023]:
        path = os.path.join(out_dir, f"r0_i{idx}.index")
        if os.path.exists(path):
            test_idx = faiss.read_index(path)
            print(f"  idx {idx}: ntotal={test_idx.ntotal}")

    print("[DONE]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()
    main(args.max_samples)
