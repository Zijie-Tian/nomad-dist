# Direct Research Notes: NoMAD-Attention (arXiv:2403.01273)

## Search Terms Used
1. "arxiv 2403.01273" — paper metadata and abstract
2. "NoMAD-Attention multiply-add-free attention citations follow-up" — follow-up work
3. "NoMAD-Attention CPU inference SIMD lookup subsequent work" — adoption and citations
4. "cpu attention optimization without multiply-add 2024 2025" — related work

## Primary Sources
- arXiv HTML v1: https://arxiv.org/html/2403.01273v1 (full text, methodology, experiments)
- GitHub repo (official code/dist): https://github.com/tonyzhang617/nomad-dist (local clone at /mnt/data/tzj/Code/nomad-dist)
- NeurIPS 2024 proceedings page: https://proceedings.neurips.cc/paper_files/paper/2024/hash/ccda3c632cc8590ee60ca5ba226a4c30-Abstract-Conference.html
- OpenReview: https://openreview.net/forum?id=4xDxVQHsbZ
- ML Anthology: https://mlanthology.org/neurips/2024/zhang2024neurips-nomadattention/
- llama.cpp issue #7532: https://github.com/ggerganov/llama.cpp/issues/7532

## Algorithm Mechanism (from paper + code)

### Core Idea
NoMAD-Attention replaces multiply-add (MAD) operations in attention score computation with in-register SIMD lookups. It targets CPU inference where MAD-based dot products become the bottleneck as sequence length grows.

### Three Technical Components

1. **Product Quantization (PQ) for Dot Products**
   - Key vectors are product-quantized into discrete codes.
   - Each d-dimensional key is split into S sub-vectors of dimension d_sub = d/S.
   - Each sub-quantizer has a codebook of 16 centroids (constrained by register size).
   - Codes are 4-bit (0-15) per sub-quantizer.
   - Asymmetric distance computation: query is kept in full precision; keys are quantized.
   - Query-dependent LUT stores dot products between query sub-vector and centroids.

2. **Compressing LUTs into SIMD Registers**
   - Standard PQ with 256 centroids (8-bit codes) needs 8192 bits per sub-quantizer LUT in FP32 — too large for 128-bit SIMD registers.
   - Solution: constrain codebooks to 16 centroids and quantize dot products dynamically to 8 bits per entry.
   - 16 × 8 bits = 128 bits, fitting exactly one 128-bit SIMD register.
   - Dynamic per-query quantization: range [dp_min, dp_max] mapped to 256 buckets.
   - LUT_s[c] = floor((dot(query, centroid_c) - dp_min) / ((dp_max - dp_min) / 256))

3. **Memory Layout Reorganization for Batch Parallel Lookups**
   - Key-code cache replaces key cache.
   - Stored in transposed, blocked format: 32 keys per block.
   - Codes are interleaved (alternating order) because each code is 4-bit and shuffle operates on byte indices.
   - SIMD shuffle retrieves 16 quantized dot products in parallel from the register-resident LUT.
   - For each block of 32 keys: bit-shift right by 4 to get first 16 codes, mask with 0xf to get last 16 codes.
   - Accumulated in 16-bit unsigned accumulators to avoid 8-bit overflow.

### Pseudocode (Algorithm 2 from paper)
- Compute key codes for current token (argmin over 16 centroids per sub-quantizer).
- Insert codes into key-code cache.
- For each query, compute 8-bit quantized LUTs for all S sub-quantizers.
- Process keys in batches of 32: load LUT into register, shuffle based on key codes, accumulate with SIMD add.
- De-quantize, scale by sqrt(d), apply softmax.

## Codebook Learning (from learn_codebooks.py + README)
- Codebooks are learned per layer and per head independently.
- Process: run inference on a learning corpus, save attention key embeddings, then k-means cluster.
- Implementation uses FAISS `IndexPQFastScan` with `METRIC_INNER_PRODUCT`.
- Default: d_sub=1, 4 bits per code (16 centroids), 100 k-means iterations.
- For CodeLLaMA-7B (32 layers, 32 heads): 1024 separate codebooks.
- Local assets/ contains pre-computed codebooks for codellama-7b, llama-2-7b, stablelm-3b at d_sub=1,2,4.

## Performance Claims (from paper Section 4)
- **Model quality**: At 8× key cache compression (d_sub=1), perplexity increase < 4% on WikiText-2 and PTB for LLaMA-2-7B and StableLM-3B.
- **PCA baseline fails**: PCA-Attention degrades catastrophically even at 2× compression.
- **Speedup**: CodeLlama-7B (4-bit weights) achieves 2× speedup at 16k context length.
  - Decoding latency: 450-600 ms/token → ~220 ms/token (d_sub=1).
  - Throughput: 0.8 tok/s → 2.2 tok/s (4-bit, 16k context).
  - Prompt processing: 2.8×10^6 ms → 1.8×10^6 ms (1.5× speedup).
- **Ablation**: PQ-Attention (8-bit codes, d_sub=2) yields limited speedup; NoMAD (4-bit, d_sub=1) achieves 8.3× speedup on attention score computation alone vs MAD-based attention at 16k/16k.

## Hardware Requirements
- CPU with AVX2 support (used in experiments: 2× Intel Xeon E5-2695 V3, 14-core each).
- 128-bit SIMD registers assumed in algorithm description; compatible with AVX2 (256-bit) and AVX-512 (512-bit) via wider parallelism.

## Software Stack
- Built on llama.cpp and FAISS.
- Currently ships binaries only (source code withheld for patent application).

## Follow-up & Adoption
- Paper accepted at NeurIPS 2024.
- llama.cpp issue #7532 (May 2024): user asked about adopting NoMAD-Attention into llama.cpp; no evidence of merge or upstream implementation found.
- Related but distinct subsequent work: EcoTransformer (arXiv:2507.20096) proposes attention without multiplication using L1 distance and Laplacian kernel — different mechanism from NoMAD's PQ+SIMD lookup.
- No other direct follow-up or widespread adoption detected in search.

## Key Limitations / Caveats
- Requires per-layer, per-head codebook learning (1024 codebooks for 7B model).
- Source code not available (binaries only); reproducibility limited to provided binaries.
- Speedup is most pronounced at long context lengths (16k); at shorter lengths, attention is less dominant in runtime.
- Key quantization adds a small perplexity penalty (<4%).
- Value cache is not compressed (only key cache).
