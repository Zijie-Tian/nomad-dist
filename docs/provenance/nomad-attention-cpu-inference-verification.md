# Verification Log: NoMAD-Attention Deep Research

**Date:** 2026-05-15
**Cited draft:** docs/drafts/nomad-attention-cpu-inference-cited.md

## Checks Performed

### 1. Core quantitative claims vs. primary source
- **Claim:** Up to 2× end-to-end speedup on 4-bit CodeLlama-7B at 16k context.
  - **Source:** arXiv HTML Section 4.3 and Appendix C: "NoMAD-Attention-based CodeLlama-7B (4-bit weights) achieves 2× speedup over the original CodeLlama-7B (4-bit weights) at 16k sequence length."
  - **Status:** PASS
- **Claim:** 8.3× speedup on attention score computation alone (ablation).
  - **Source:** arXiv HTML Section 4.4 (Figure 5 caption): "NoMAD-Attention achieves 8.3× speedup over MAD-based attention."
  - **Status:** PASS
- **Claim:** Perplexity increase <4% at 8× key-cache compression (d_sub=1).
  - **Source:** arXiv HTML Section 4.2: "consistently less than a 4% increase."
  - **Status:** PASS
- **Claim:** Decoding latency drops from 450–600 ms/token to ~220 ms/token.
  - **Source:** arXiv HTML Appendix C.
  - **Status:** PASS

### 2. Technical mechanism claims
- **Claim:** PQ with 16-centroid sub-quantizers, 4-bit codes, dynamic 8-bit LUT quantization to fit 128-bit SIMD registers.
  - **Source:** arXiv HTML Sections 3.1 and 3.2, Equation (1).
  - **Status:** PASS
- **Claim:** SIMD shuffle on blocks of 32 interleaved keys, bit-shift/bytemask extraction.
  - **Source:** arXiv HTML Appendix A (Algorithm 3) and Section 3.3.
  - **Status:** PASS

### 3. Implementation and distribution claims
- **Claim:** Binary-only distribution, source withheld for patent application.
  - **Source:** Local README.md: "Currently, the repository offers executable binaries for NoMAD-Attention as the source code undergoes a patent application process."
  - **Status:** PASS
- **Claim:** Built on llama.cpp and FAISS.
  - **Source:** arXiv HTML Section 4 (Software Implementation).
  - **Status:** PASS

### 4. Adoption claims
- **Claim:** Not upstreamed into llama.cpp; issue #7532 raised but no merge evident.
  - **Source:** https://github.com/ggerganov/llama.cpp/issues/7532 (fetched content shows question posted, no PR/merge referenced).
  - **Status:** PASS WITH NOTES — evidence is absence of activity; we do not claim certainty about future or private discussions.
- **Claim:** Accepted at NeurIPS 2024.
  - **Source:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/ccda3c632cc8590ee60ca5ba226a4c30-Abstract-Conference.html
  - **Status:** PASS

### 5. Related work
- **Claim:** EcoTransformer (arXiv:2507.20096) is a distinct subsequent work using L1 distance / Laplacian kernel.
  - **Source:** Search result snippet for arXiv 2507.20096.
  - **Status:** PASS — we did not read the full PDF, but the abstract/distinction is clear from metadata.

## Issues Found
- **MINOR:** The paper uses "NoMAD" and "NoMAD-Attention" somewhat interchangeably. The draft uses "NoMAD-Attention" consistently for clarity.
- **MINOR:** The exact perplexity numbers are presented qualitatively in the paper ("less than 4%"); no precise table of perplexity values was extracted from the HTML. The draft respects this limitation and does not invent exact numbers.

## Overall Verdict
**PASS WITH NOTES.** All critical claims are traceable to primary sources. The one adoption claim (no upstream merge) is based on observable public evidence absence, which is appropriately qualified.
