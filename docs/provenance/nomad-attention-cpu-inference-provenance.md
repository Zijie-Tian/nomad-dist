# Provenance: NoMAD-Attention Deep Research

- **Date:** 2026-05-15
- **Rounds:** 1 direct research round (lead-owned, no subagents)
- **Sources consulted:** 10 distinct sources
- **Sources accepted:** 10
- **Sources rejected:** 0
- **Verification:** PASS WITH NOTES
- **Plan:** docs/planning/nomad-attention-cpu-inference.md
- **Research files:**
  - docs/research-notes/nomad-attention-cpu-inference-research-direct.md
  - docs/drafts/nomad-attention-cpu-inference-draft.md
  - docs/drafts/nomad-attention-cpu-inference-cited.md
  - docs/provenance/nomad-attention-cpu-inference-verification.md

## Source Inventory

| # | Source | URL / Path | Used For |
|---|--------|-----------|----------|
| 1 | NoMAD-Attention paper (arXiv abstract) | https://arxiv.org/abs/2403.01273 | Metadata, citations |
| 2 | NoMAD-Attention paper (HTML full text) | https://arxiv.org/html/2403.01273v1 | Primary evidence: mechanism, results, ablation |
| 3 | NeurIPS 2024 proceedings | https://proceedings.neurips.cc/paper_files/paper/2024/hash/ccda3c632cc8590ee60ca5ba226a4c30-Abstract-Conference.html | Venue verification |
| 4 | OpenReview forum | https://openreview.net/forum?id=4xDxVQHsbZ | Peer review context |
| 5 | ML Anthology | https://mlanthology.org/neurips/2024/zhang2024neurips-nomadattention/ | Archival entry |
| 6 | Official GitHub repo (README) | https://github.com/tonyzhang617/nomad-dist /mnt/data/tzj/Code/nomad-dist/README.md | Distribution status, codebook learning, usage |
| 7 | learn_codebooks.py | /mnt/data/tzj/Code/nomad-dist/learn_codebooks.py | Implementation details of codebook learning |
| 8 | llama.cpp Issue #7532 | https://github.com/ggerganov/llama.cpp/issues/7532 | Adoption status |
| 9 | Jegou et al. (2010) PQ paper | https://ieeexplore.ieee.org/document/5432202 | Background on Product Quantization |
| 10 | Dong et al. (2021) rank collapse | https://proceedings.mlr.press/v139/dong21a.html | Motivation for small codebooks |
| 11 | EcoTransformer (arXiv:2507.20096) | https://www.arxiv.org/pdf/2507.20096 | Related subsequent work |

## Verification Notes
- All critical quantitative claims (2× speedup, 8.3× ablation, <4% perplexity increase) traced to arXiv HTML primary source.
- Binary-only distribution claim verified against local README.md.
- Adoption claim (no upstream merge) based on observable public evidence; qualified as "no evidence of merge" rather than absolute certainty.
- No fabricated numbers, tables, or figures. All tables reconstructed from explicit paper claims.
