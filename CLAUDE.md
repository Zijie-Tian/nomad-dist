# NoMAD-Attention — Project Guide

NoMAD-Attention (arXiv:2403.01273) + a LongBench evaluation harness for
**LLaMA-3.2-1B-Instruct** using NoMAD attention (dense-prefill on GPU + NoMAD PQ
lookup on decode).

## Key files
- `nomad_llama.py` — NoMAD `attention_interface` for LLaMA (GQA; vectorized PQ + 8-bit LUT; **dense-prefill / NoMAD-decode** split; CPU or GPU). Registers `ALL_ATTENTION_FUNCTIONS["nomad"]`.
- `longbench/` — ported LongBench framework (data/config/metrics/scorer/templates) + NoMAD `runner.py`.
- `eval_longbench.py` / `score_longbench.py` — generation + scoring CLIs.
- `save_keys_llama.py` — capture post-rotary keys (per kv-head) for codebook training.
- `learn_codebooks.py` — train PQ codebooks (FAISS `IndexPQFastScan`, nbits=4 → 16 centroids).
- `assets/llama-3.2-1b-dsub1/` — 128 NoMAD codebooks (16 layers × 8 kv-heads, head_dim 64, d_sub=1).

## conda environment `nomad` (exact versions, validated on A100 node)

If the node has no conda, install miniconda first:
```bash
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
bash /tmp/mc.sh -b -p ~/miniconda3
```

Create env + install (pinned to validated versions):
```bash
~/miniconda3/bin/conda create -y -n nomad python=3.11
PIP=~/miniconda3/envs/nomad/bin/pip
$PIP install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1 (A100)
$PIP install "transformers==4.56.0" faiss-cpu datasets numpy tqdm accelerate sentencepiece
```

| package | version |
|---|---|
| python | 3.11 |
| torch | 2.5.1+cu121 |
| transformers | **4.56.0** (required) |
| faiss-cpu | 1.14.2 |
| datasets / numpy / tqdm / accelerate / sentencepiece | latest |

> `transformers==4.56.0` is required: `nomad_llama.py` relies on the LlamaAttention
> forward signature and `ALL_ATTENTION_FUNCTIONS` dispatch present in 4.56.x.

## Run LongBench (GPU, dense-prefill + NoMAD-decode)
```bash
PY=~/miniconda3/envs/nomad/bin/python
CUDA_VISIBLE_DEVICES=<gpu> $PY eval_longbench.py --method nomad \
  --model /path/to/Llama-3.2-1B-Instruct --data-root /path/to/LongBench \
  --datasets trec,samsum,... --max-samples -1 --device cuda \
  --output-dir longbench_out/llama32-1b-instruct_nomad-full/pred
$PY score_longbench.py --pred-dir longbench_out/llama32-1b-instruct_nomad-full/pred
```
Flags: `--method baseline` (original attention), `--full-nomad` (NoMAD prefill too),
`--no-8bit`, `--max-model-len 2048`. Per-sample on a dedicated A100: ~5s (32-gen) /
~10s (64-gen) / ~80s (512-gen); 512-gen tasks dominate. Full 21-task ≈ 24-30h.

## ===== RESUME the interrupted full run on ANOTHER node =====

The full LongBench run (NoMAD, dense-prefill) was **paused at 13/21 tasks done**.

- Done (13, do NOT rerun): passage_count, passage_retrieval_en, passage_retrieval_zh,
  hotpotqa, 2wikimqa, musique, triviaqa, trec, multifieldqa_en, multifieldqa_zh, lsht, lcc, repobench-p
- Remaining (8): **samsum, narrativeqa, qasper, dureader, gov_report, qmsum, multi_news, vcsum** (~19-20h on a dedicated A100)

### Steps on the new node
1. **Clone**:
   ```bash
   mkdir -p ~/Code && cd ~/Code
   git clone -b tzj/nomad https://github.com/Zijie-Tian/nomad-dist.git && cd nomad-dist
   ```
2. **conda env `nomad`** — see section above (codebooks come with the clone).
3. **Model + data** (LLaMA-3.2 is gated — copy from a machine that has it, or `huggingface-cli login`):
   ```bash
   # model  -> ~/models/Llama-3.2-1B-Instruct   (e.g. rsync -a node:~/models/Llama-3.2-1B-Instruct ~/models/)
   # data   -> ~/data/LongBench/data/*.jsonl     (rsync, or download THUDM/LongBench and dump to jsonl)
   ```
4. **Download the 13 finished results** from aliyunpan and unpack into pred/:
   ```bash
   unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
   aliyunpan download /tmp/nomad/nomad-interrupted-results.tgz       # -> aliyunpan savedir
   tar xzf <savedir>/nomad-interrupted-results.tgz -C ~/Code/nomad-dist/longbench_out/
   # now longbench_out/llama32-1b-instruct_nomad-full/pred/ has the 13 jsonl + result.json
   ```
5. **Run ONLY the remaining 8** (keeps the 13; runner writes one jsonl per finished task):
   ```bash
   PY=~/miniconda3/envs/nomad/bin/python
   DS="samsum,narrativeqa,qasper,dureader,gov_report,qmsum,multi_news,vcsum"
   tmux new-session -d -s nomad_full "CUDA_VISIBLE_DEVICES=<gpu> $PY eval_longbench.py --method nomad \
     --model ~/models/Llama-3.2-1B-Instruct --data-root ~/data/LongBench \
     --datasets $DS --max-samples -1 --device cuda \
     --output-dir longbench_out/llama32-1b-instruct_nomad-full/pred 2>&1 | tee -a longbench_out/full_gpu2.log"
   ```
6. **Score** after all 21 are present:
   ```bash
   $PY score_longbench.py --pred-dir longbench_out/llama32-1b-instruct_nomad-full/pred
   ```

### Gotchas
- Runner writes jsonl **per task** (only after that task's 200 samples finish) — **no per-sample checkpoint**. A task interrupted mid-way must rerun (that's why `samsum` is in the remaining list).
- Training codebooks MUST cap FAISS threads: `OMP_NUM_THREADS=8` **and** `faiss.omp_set_num_threads(8)` — 1-dim k-means on 78 cores was ~160× slower (168s/index → 1s/index).
- GPU NoMAD here is a **quality** simulation; it does NOT reflect NoMAD's CPU-SIMD speed advantage (that needs the C++ kernel, which is patent-pending/expired in the official binaries).
