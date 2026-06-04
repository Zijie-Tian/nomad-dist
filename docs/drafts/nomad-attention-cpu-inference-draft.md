# NoMAD-Attention: Efficient LLM Inference on CPUs Through Multiply-Add-Free Attention

## Executive Summary

NoMAD-Attention (Zhang et al., 2024) is a hardware-aware attention algorithm that replaces the expensive multiply-add (MAD) operations in transformer attention with ultra-low-latency SIMD in-register lookups. By combining product quantization of key vectors, dynamic 8-bit quantization of dot-product lookup tables, and batched SIMD shuffle instructions, it achieves up to 2× end-to-end speedup on 4-bit quantized CodeLlama-7B at 16k context length on AVX2 CPUs, while increasing perplexity by less than 4%. The method requires no model fine-tuning and is compatible with pre-trained decoder-only transformers. However, the official implementation is currently distributed as binaries only (source withheld for patent application), and adoption in mainstream inference engines remains limited.

---

## 1. What Problem Does NoMAD-Attention Solve?

### 1.1 The Bottleneck: Multiply-Add Operations in Attention

In autoregressive decoder-only LLMs, the attention mechanism computes all-pair dot products between a query vector and all cached key vectors. At decoding step t, this requires O(t·d) MAD operations, where d is the head dimension. As sequence length grows, attention score computation dominates total inference time on CPUs: while MLPs, skip connections, and normalization remain O(1) per step, attention grows linearly with context length and quickly becomes the bottleneck.

CPUs are poorly suited to the repetitive, highly parallel MAD workloads that GPUs handle efficiently. Prior CPU optimizations (e.g., weight quantization, sparse attention) either require training from scratch or do not address the MAD bottleneck directly.

### 1.2 The Opportunity: SIMD In-Register Lookups

Modern CPUs provide SIMD registers (128-bit to 512-bit) that support single-instruction, multiple-data parallelism. Crucially, data stored in registers can be accessed in 1–2 CPU cycles, versus tens to hundreds of cycles for L1/L2 cache or main memory. NoMAD-Attention exploits this by shifting the computation paradigm from arithmetic (MAD) to memory lookups performed entirely within registers.

---

## 2. How NoMAD-Attention Works

NoMAD-Attention introduces three algorithmic innovations to make register-resident lookup-based attention feasible.

### 2.1 Product Quantization of Key Vectors

NoMAD adapts Product Quantization (PQ), originally designed for nearest-neighbor search, to estimate query-key dot products via table lookups.

- **Sub-quantizers**: A d-dimensional key vector is split into S sub-vectors of dimension d_sub = d/S.
- **Codebooks**: Each sub-quantizer has its own codebook of cluster centroids. A key vector is quantized to the index of its nearest centroid per sub-quantizer (measured by L2 distance).
- **Asymmetric dot-product estimation**: For a given query, a query-dependent Lookup Table (LUT) is computed once per sub-quantizer. The LUT stores dot products between the query sub-vector and each centroid. The full dot product is approximated by summing the LUT entries indexed by the key’s quantized codes.

This replaces the O(d) MADs per dot product with O(S) table lookups and additions.

### 2.2 Compressing Lookup Tables into SIMD Registers

A naive PQ approach with 256 centroids (8-bit codes) and FP32 LUT entries requires 8192 bits per sub-quantizer—far exceeding the 128-bit width of universal SIMD registers (ARM NEON, baseline AVX). NoMAD solves this with two constraints:

1. **Constrained codebook size**: Each sub-quantizer uses exactly 16 centroids, yielding 4-bit codes. The motivation is empirical: attention outputs are known to lose rank extremely quickly, suggesting key embeddings exhibit clustering amenable to small codebooks.
2. **Dynamic 8-bit quantization of LUT entries**: For each query and sub-quantizer, dot products to the 16 centroids are computed in FP32. The range [dp_min, dp_max] is divided into 256 buckets, and each dot product is quantized to an 8-bit index. This yields a 128-bit LUT (16 entries × 8 bits) that fits exactly into one 128-bit SIMD register.

The quantization and de-quantization overhead is minimal because it is performed once per query per sub-quantizer, amortized over the entire context length.

### 2.3 Memory Layout Reorganization for Batch Parallel Lookups

To maximize throughput, NoMAD reorganizes the key cache into a **transposed, blocked key-code cache**:

- **Block size**: 32 keys per block.
- **Interleaved storage**: Because each code is 4-bit and SIMD shuffle operates on byte indices, two 4-bit codes are packed per byte in alternating order.
- **Batch lookup**: For each block of 32 keys and each sub-quantizer:
  1. The 128-bit LUT is loaded into a register.
  2. The first 16 key codes are extracted via a 4-bit right shift and fed to `simd_shuffle` to retrieve 16 quantized dot products in parallel.
  3. The last 16 key codes are extracted via bitwise AND with `0x0f` and similarly retrieved.
  4. The 32 retrieved values are accumulated with `simd_add` into 16-bit accumulators (8-bit accumulators would overflow).

After processing all S sub-quantizers, the accumulated values are de-quantized, scaled by 1/√d, and passed through softmax.

---

## 3. Codebook Learning and Implementation

### 3.1 Learning Codebooks

NoMAD requires per-layer, per-head codebooks because the value distributions of key embeddings vary significantly across layers and heads (verified empirically on LLaMA-2-7B).

The learning procedure is:
1. Run inference with original attention on a small learning corpus (e.g., first 100 samples of WikiText-2 training set).
2. Save attention key embeddings for each layer and head.
3. Perform k-means clustering independently per head to learn 16 centroids per sub-quantizer.

For a 32-layer, 32-head model (e.g., CodeLLaMA-7B), this produces 1024 separate codebooks.

### 3.2 Software and Hardware

- **Implementation**: Built in C/C++ on top of llama.cpp and FAISS.
- **Distribution**: The repository provides pre-compiled binaries (`main`, `perplexity`, `quantize`) and Python scripts for codebook learning. The C++ source is withheld pending a patent application.
- **Hardware**: Requires a CPU with AVX2 support. Experiments used a server with 2× Intel Xeon E5-2695 V3 (14 cores each) and 512 GB DDR4 RAM.

---

## 4. Experimental Results

### 4.1 Model Quality (Perplexity)

NoMAD-Attention maintains model quality at aggressive compression ratios:

- At **8× key-cache compression** (d_sub = 1, 4 bits per float in key), perplexity increases by **consistently less than 4%** on WikiText-2 and PTB for both LLaMA-2-7B and StableLM-3B.
- **PCA-Attention**, a dimensionality-reduction baseline, fails catastrophically even at 2× compression. The authors attribute this to PCA’s symmetric dot-product computation versus NoMAD’s asymmetric computation (full-precision query, quantized keys).
- Beyond 8× compression (d_sub > 1), quality degrades, but NoMAD still outperforms PCA.

### 4.2 Inference Efficiency

End-to-end speedups are measured on CodeLlama-7B (4-bit and 16-bit weights) generating 4096 tokens after prompts up to 16k tokens:

| Metric | Baseline (4-bit) | NoMAD (d_sub=1, 4-bit) | Speedup |
|---|---|---|---|
| Decoding latency (16k context) | 450–600 ms/token | ~220 ms/token | **~2×** |
| Throughput (16k context, 4-bit) | ~0.8 tok/s | ~2.2 tok/s | **~2.75×** |
| Throughput (16k context, 16-bit) | ~2.2 tok/s | ~4.4 tok/s | **2×** |
| Prompt processing (16k) | ~2.8×10⁶ ms | ~1.8×10⁶ ms | **~1.5×** |

The speedup grows with context length because attention dominates runtime more strongly as the KV cache grows.

### 4.3 Ablation: Why In-Register Matters

An ablation comparing MAD-based Attention, PQ-Attention (8-bit codes, d_sub=2), and NoMAD-Attention (4-bit codes, d_sub=1) at 16k queries / 16k context on a single thread reveals:

- **PQ-Attention** achieves limited speedup over MAD-based attention because LUTs reside in cache/memory, causing stall cycles. Its key-caching overhead is also 2× higher than NoMAD because finding codes requires 256 distance computations per sub-quantizer versus 16.
- **NoMAD-Attention** achieves **8.3× speedup** on attention score computation alone versus MAD-based attention, demonstrating that the in-register lookup strategy is the critical factor.

---

## 5. Adoption, Follow-up Work, and Limitations

### 5.1 Adoption Status

- **Academic**: Accepted at NeurIPS 2024. Available through arXiv, OpenReview, and ML Anthology.
- **Open-source integration**: As of the research date, NoMAD-Attention has **not been upstreamed** into llama.cpp or other major inference frameworks. A llama.cpp issue (#7532, May 2024) raised the question of adoption, but no merge or implementation is evident.
- **Binary-only distribution**: The official repository (tonyzhang617/nomad-dist) provides binaries and pre-computed codebooks but not the core C++ source, limiting independent verification and portability.

### 5.2 Related Subsequent Work

- **EcoTransformer** (arXiv:2507.20096) also explores attention without multiplication, but uses an L1-distance-based Laplacian kernel rather than PQ+SIMD lookups. It is architecturally distinct from NoMAD.

### 5.3 Limitations and Open Questions

1. **Codebook overhead**: Learning and storing 1024 codebooks for a 7B model adds complexity. The memory overhead of codebooks is small, but the preparation pipeline is non-trivial.
2. **Source availability**: Binary-only distribution prevents community optimization, porting to other architectures (e.g., ARM NEON optimizations), and integration into existing serving stacks.
3. **Context-length dependence**: Speedup is modest at short contexts; the method is designed for long-context CPU inference.
4. **Value cache**: NoMAD compresses only the key cache; the value cache remains uncompressed. Future work could explore value-cache compression synergies.
5. **Quantization interaction**: The paper evaluates NoMAD on top of 4-bit and 8-bit weight quantization (GGUF q4_0, q8_0). It is unclear how it interacts with more aggressive quantization schemes or with recent KV-cache quantization methods.
6. **Single-architecture evaluation**: All experiments use Intel Xeon E5-2695 V3 (AVX2). Performance on newer AVX-512 CPUs, Apple Silicon (NEON), or AMD Zen architectures is unreported.

---

## 6. Bottom Line

NoMAD-Attention is a principled, hardware-aware method that demonstrates a viable path to eliminating MAD operations from CPU attention via SIMD register lookups. Its combination of product quantization, dynamic 8-bit LUT compression, and batched shuffle instructions yields substantial speedups at long context lengths with minimal quality loss. The primary barriers to broader impact are the binary-only distribution and the lack of integration into mainstream inference engines. For practitioners running long-context LLM inference on AVX2 CPUs, NoMAD offers a compelling ~2× speedup if they are willing to adopt a custom binary distribution and pre-compute layer/head-specific codebooks.
