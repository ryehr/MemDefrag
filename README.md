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
fit on a single such GPU (two model copies are loaded when eager tracing is
used: an eager-attention copy for tracing and a default copy for generation;
`--tracing sdpa` uses a single copy).

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

**Knowledge retention** (50 steps; tracer layer 13, Top-1, informativeness-based
forgetting, `N_max = 12,800`):

```bash
python QA_benchmark.py --dataset nqa --unrelated_num 49 \
  --reorder_base_layer 13 --keep_num 1 --query_mode last \
  --shuffle_knowledge --forget_strategy perplexity
```

Useful flags: `--keep_num K` (Top-K filtering; negative = reorder only),
`--adaptive_k --adaptive_tau 2.0 --adaptive_k_max 4` (margin-based adaptive K),
`--eval_steps 1,10,20,30,40,50` (evaluate only at checkpoints; memory evolution
is unaffected), `--tracing sdpa` (single-model SDPA tracing),
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
python Long_benchmark.py --dataset hotpotqa --reorder_base_layer 13 \
  --keep_num 2 --query_mode mean --shuffle_knowledge
```

**LoCoMo (dialogue memory)**:

```bash
python LoCoMo_benchmark.py --keep_num 3            # MemDefrag Top-3
python LoCoMo_benchmark.py --reorder_base_layer -1 # vanilla latent memory
```

**Efficiency**:

```bash
python latency_breakdown.py   # per-prompt latency decomposition at n=50
python peak_memory.py         # peak GPU memory, vanilla vs. MemDefrag
python cost_scaling.py        # per-query cost vs. injected context (vs. full-context)
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

Per-model tracer configurations (NaturalQA / SQuAD): Llama-3.1-8B-Instruct
layer 13 (last / all-token), Qwen2.5-7B-Instruct layer 14 (all-token),
Mistral-7B-Instruct-v0.3 layer 15 (all-token), Gemma-2-9b-it layer 15
(last / all-token). For a new backbone, a lightweight layer sweep
(`big_model_layer_sweep.py`) identifies a high-quality tracer layer.

## Citation

The paper is currently under double-blind review; citation information will be
added upon publication.
