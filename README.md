# MemDefrag: Latent Memory Defragmentation for Large Language Models

MemDefrag is a **training-free, model-agnostic** framework for long-term *latent
memory* in LLMs (the paradigm of MemoryLLM / M+, where past knowledge is stored
as per-layer hidden-state prefixes). It addresses two failure modes of that
paradigm — accumulated positional-encoding distortion and the absence of any
mechanism to trace query-relevant memory — with two components:

1. **Inference-time memory defragmentation.** A small set of middle transformer
   layers concentrates the highest attention density on the query-relevant
   memory fragment (an inherent *tracing signal*). At each prompt, MemDefrag
   scores every stored fragment at a *tracer layer*, then **ranks, reorders,
   and Top-K filters** the memory before generation.
2. **Informativeness-guided proportional forgetting.** When the memory budget
   `N_max` is exceeded, eviction quotas are distributed proportionally across
   fragments and, within each fragment, the tokens with the lowest
   self-information are pruned.

Everything runs on frozen off-the-shelf checkpoints — no gradient updates, no
auxiliary modules.

## Repository layout

| Path | Purpose |
|---|---|
| `utilities.py` | Core library: memory formation / concatenation / forgetting, hidden-prefix injection & generation, `reorder()` (defragmentation with eager tracing), attention-density probes |
| `sdpa_tracing.py` | SDPA tracing: single-layer density probe (early-stop forward, no S×S materialization; supports GQA/RoPE/logit-softcapping/sliding-window) + slice-only defragmentation. Used for large models on a single GPU |
| `QA_benchmark.py` | Knowledge-retention benchmark (NaturalQA / SQuAD, multi-step memory updates). Supports `--tracing {eager,sdpa}`, `--adaptive_k`, `--eval_steps` checkpointed evaluation |
| `QA_benchmark_text.py` | Text-token counterpart (no latent memory; no positional distortion) |
| `QA_benchmark_investigation_acc.py` / `_rank.py` (+ `_text` variants) | Investigation experiments: accuracy vs. memory length, layer-wise attention-density heatmaps/correlations, tracer-layer rank statistics |
| `Long_benchmark.py`, `longbench_config/` | LongBench evaluation (6 datasets, string match / F1 / ROUGE-L / BERTScore) |
| `LoCoMo_benchmark.py`, `locomo_layer_sweep.py` | Multi-turn dialogue memory on LoCoMo (session-per-fragment injection, per-question defragmentation, tracer-layer sweep) |
| `big_model_layer_sweep.py` | Full-depth tracer-layer sweep on larger models (24B–34B), eager or SDPA capture, both last-/all-token attention in one forward |
| `latency_breakdown.py`, `peak_memory.py`, `cost_scaling.py` | Efficiency studies: per-prompt latency decomposition, peak GPU memory, per-query cost vs. injected-context scale (vs. full-context baseline) |
| `make_adaptive_table.py`, `make_cost_table.py`, `check_adaptive_progress.py`, `process_rank.py` | Result aggregation / table generation |
| `plot_investigation*.py`, `plot_formal/` | Paper-quality figures |
| `data_new/nqa`, `data_new/squad` | Processed evaluation data (see *Data*) |

## Setup

```bash
conda create -n generative-memory python=3.10
conda activate generative-memory
pip install torch transformers accelerate llmlingua datasets \
    rouge-score bert-score nltk scipy matplotlib
```

Experiments in the paper use 4× GPUs with 141 GB HBM3e; most 7–9B experiments
fit on a single such GPU.

### Tracing: SDPA (recommended, default) vs. eager

Two implementations of the tracer-layer density probe are provided:

- **`--tracing sdpa` (default, recommended).** A single model copy runs with
  SDPA attention; at the tracer layer, attention is computed manually **only
  for the prompt rows** (row-wise softmax is independent, so the densities are
  mathematically identical to taking rows of the full attention map). No
  `S × S` attention matrix is ever materialized and no second model copy is
  loaded — at `S ≈ 12.8K` this saves tens of GB of transient memory per layer
  plus a full set of model weights, and is also substantially faster.
- **`--tracing eager`.** Reference implementation: a second eager-attention
  model copy materializes full attention maps at the tracer layer. Kept for
  exactness checks; required for `--adaptive_k` and the rotate baseline.

Top-K selection has been verified identical between the two paths (densities
agree within bf16 rounding).

## Data

- **NaturalQA / SQuAD** (`data_new/nqa`, `data_new/squad`): processed
  (context, question, answer) triples. The NaturalQA subset follows
  [YuWangX/KnowledgeRetentionProcessed](https://huggingface.co/datasets/YuWangX/KnowledgeRetentionProcessed)
  to ensure low leakage and mutual independence among contexts.
- **LongBench**: loaded from Hugging Face automatically (`THUDM/LongBench`).
- **LoCoMo**: download the released data into `data_new/locomo/`:

```bash
mkdir -p data_new/locomo
wget -P data_new/locomo \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

## Reproducing the main experiments

**Knowledge retention** (50 steps; Top-1, informativeness-based forgetting,
`N_max = 12,800`; example backbone Qwen2.5-7B-Instruct with its tracer layer 14):

```bash
python QA_benchmark.py --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset nqa --unrelated_num 49 \
  --reorder_base_layer 14 --keep_num 1 --query_mode mean \
  --shuffle_knowledge --forget_strategy perplexity
```

Useful flags: `--keep_num K` (Top-K filtering; negative = reorder only),
`--adaptive_k --adaptive_tau 2.0 --adaptive_k_max 4` (margin-based adaptive K;
requires `--tracing eager`), `--eval_steps 1,10,20,30,40,50` (evaluate only at
checkpoints; memory evolution is unaffected),
`--using_llmlingua --lingua_rate 0.5` (prompt-compression compatibility).

**Tracer-layer investigation** (average density rank of the target fragment
across all layers/positions): `QA_benchmark_investigation_rank.py`, or for
larger backbones:

```bash
python big_model_layer_sweep.py --model_name Qwen/Qwen2.5-32B-Instruct \
  --group_num 100 --positions 0,4,9,14,19 --tracing sdpa
```

**LongBench**:

```bash
python Long_benchmark.py --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset hotpotqa --reorder_base_layer 14 --keep_num 2 \
  --query_mode mean --shuffle_knowledge
```

**LoCoMo (dialogue memory)**:

```bash
python LoCoMo_benchmark.py --model_name Qwen/Qwen2.5-7B-Instruct \
  --reorder_base_layer 14 --query_mode mean --keep_num 3   # MemDefrag Top-3
python LoCoMo_benchmark.py --model_name Qwen/Qwen2.5-7B-Instruct \
  --reorder_base_layer -1                                  # vanilla latent memory
```

**Efficiency** (all SDPA tracing):

```bash
BASE="--model_name Qwen/Qwen2.5-7B-Instruct --reorder_base_layer 14 --query_mode mean"
python latency_breakdown.py $BASE  # per-prompt latency decomposition at n=50
python peak_memory.py $BASE        # peak GPU memory, vanilla vs. MemDefrag
python cost_scaling.py $BASE       # per-query cost vs. injected context (vs. full-context)
```

## Paper ↔ code terminology

| Paper | Code |
|---|---|
| Tracer layer `l*` | `--reorder_base_layer` |
| Top-K filtering | `--keep_num` |
| Last-token / all-token attention (App. B.2) | `--query_mode last` / `mean` |
| Informativeness-based forgetting (Sec. 4.2) | `--forget_strategy perplexity` |
| Memory capacity `N_max` | `--max_knowledge_tokens` |
| Number of update steps `n` | `--unrelated_num` + 1 |

Reference tracer configurations (NaturalQA / SQuAD): Qwen2.5-7B-Instruct
layer 14 (all-token), Mistral-7B-Instruct-v0.3 layer 15 (all-token),
Gemma-2-9b-it layer 15 (last / all-token). For any other backbone, a
lightweight full-depth layer sweep (`big_model_layer_sweep.py`, one SDPA
forward per probe yields all layers × both attention modes) identifies a
high-quality tracer layer in minutes.

## Citation

```bibtex
@misc{yan2026memdefraglatentmemorydefragmentation,
      title={MemDefrag: Latent Memory Defragmentation for Large Language Models}, 
      author={Ruiyi Yan and Zhuoyuan Mao and Yiwen Guo},
      year={2026},
      eprint={2607.05969},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.05969}, 
}
```
