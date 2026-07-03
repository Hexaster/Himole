# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This repo is a **study reimplementation** of **HiMoLE** (Hierarchical Mixture of LoRA Experts),
from the NeurIPS 2025 paper *"HiMoLE: Towards OOD-Robust LoRA via Hierarchical Mixture of Experts"*
(Jiang et al., Zhejiang University / Ant Group). The goal is to recreate the method described in the
paper for learning purposes.

**Current state:** Build 1 has a learning scaffold for the plain-LoRA baseline. The framework,
tests, notebook, and script exist, with the pedagogically-critical lines left as `TODO` /
`NotImplementedError` for study. The paper lives in `paper/`:
- `paper/HiMoLE.pdf` — the original PDF (29 pages).
- `paper/himole.md` — full extracted text (authoritative reference; appendices have the implementation
  details: B = symbols, C = auxiliary loss, D = proofs, E = hyperparameters/metrics/datasets, F =
  training algorithm).
- `paper/claude-share-lora-himole-2026-06-01.md` — a plain-language Socratic summary of LoRA basics +
  HiMoLE intuition. Read this first for the "why"; read `himole.md` for exact equations and numbers.

Use `pytest -v` for the TODO-driven tests and
`python scripts/train_baseline.py --tiny --steps 5 --samples 8` for the eventual tiny smoke test
after the Build 1 TODOs are filled. The paper's stack is PyTorch + HuggingFace (`transformers`,
`peft`, `datasets`), Python 3.8.

## What HiMoLE is (so you don't have to re-parse the paper)

HiMoLE replaces the **FFN layer** in every transformer block of a frozen base LLM with a
**hierarchical mixture of LoRA experts**. It targets OOD robustness: plain LoRA and flat token-routed
MoE-LoRA (MixLoRA, HydraLoRA) overfit in-distribution and degrade out-of-distribution. The core
insight (paper Fig. 1d) is a bias-variance tradeoff in routing granularity:
- **Token-level routing** → better ID, worse OOD (exploits brittle local features).
- **Sentence-level routing** → worse ID, better OOD (coarse but stable global semantics).

HiMoLE does **both hierarchically**: sentence routing coarsely picks a knowledge group, token routing
refines within it. An expert fires only if global *and* local agree (a logical AND).

### Expert structure
- `N` **Knowledge Competition Groups (KCGs)**, each with `M` **Knowledge Collaboration Experts (KCEs)**.
- Each KCE is a LoRA module: `E = B·A`, with `A ∈ R^{din×r}`, `B ∈ R^{r×dout}`, `r ≪ min(din,dout)`.
- Total experts = `N×M`. All `M` KCEs in a KCG share the same Stage-1 initialization.
- Intra-group = collaboration; cross-group = competition (reduces negative transfer).

### Hierarchical routing (forward pass of one HiMoLE/FFN layer)
1. `h_sen = avg_pool(h_token)` over the sentence's token hidden states.
2. Sentence router (linear `W_sen`): `G_sen = W_sen · h_sen` → scores over the `N` KCGs.
3. Token router (linear `W_token`): `G_token = W_token · h_token` → scores over the `N×M` experts.
4. `G_hie = KeepTopK( Softmax( G_sen ⊙ G_token ) )`, top-k with `k=2`.
   - **Dimension subtlety to get right:** `G_sen` has `N` entries (one per group), `G_token` has `N×M`
     entries (one per expert). The element-wise `⊙` broadcasts each group's sentence score across its
     `M` token scores — the sentence score *gates* the token scores within that group.
5. Output: `o = W0·h_token + Σ_{i=1}^{N×M} G_hie^(i) · E_i · h_token`, where `W0` is the frozen
   original FFN weight.

### Two-stage training (critical — removing Stage 1 costs up to ~31.9% F1; see Table 3)
- **Stage 1 — initialize KCGs.** Embed every training sentence with a pretrained encoder, K-means into
  `N` clusters, then train each KCG **independently and in parallel** on its cluster's sub-dataset
  (LR `3e-4`). This gives experts genuinely distinct specializations before routing is learned —
  without it the router gets no useful gradient (chicken-and-egg: random experts → no router signal).
- **Stage 2 — joint optimization.** Train experts + routers together on the full dataset (LR `3e-5`)
  with `L = L_task + α·L_aux + β·L_diverse`.

### Losses (Stage 2)
- `L_task` — the downstream task loss (LM loss).
- `L_aux` — Switch/ST-MoE load-balancing loss: `L_aux = N · Σ_i F_i·P_i`, where `F_i` is the fraction
  of tokens dispatched to expert `i` and `P_i` the mean router probability for `i` (paper Appendix C).
- `L_diverse` — discourages expert redundancy. KCG output `e_n = Σ_m G_token^(m)·E_m·h_token`,
  normalize `e_n ← e_n / max(‖e_n‖₂, ε)` (`ε≈1e-8`), pairwise cosine sim `S_nl = ⟨e_n,e_l⟩`,
  `L_diverse =` mean of `S_nl` over all distinct KCG pairs. Computed every 10 layers on a sampled
  subset of expert outputs to control cost.

## Paper's experimental configuration (reproduction targets)

Three tasks, each with an ID dataset and an OOD dataset:

| Task | Base model | ID encoder for clustering | ID dataset | OOD dataset | N clusters |
|------|-----------|---------------------------|-----------|-------------|-----------|
| NER (biomedical) | OneKE-13B | BioBERT | BigBio-NER | RareDis | 4 |
| SA (social science) | Llama-2-7B | BERTweet | SocialiteInstructions | OPTIMISM | 3 |
| EQA (general) | Llama-2-7B | DeBERTa-v3-large | SQuAD | NewsQA | 3 |

Key hyperparameters (paper Table 7), `M = 4` KCEs per KCG in all cases:
- LoRA rank `r = 8`, alpha `16`, dropout `0.05`, top-k `2`.
- Applied to FFN `Up`, `Down`, `Gate` projections of every transformer block.
- AdamW, batch size 16, cutoff length 1024.
- LR: `3e-4` (Stage 1), `3e-5` (Stage 2).
- Plain-LoRA baseline uses rank `80` / alpha `160` to match HiMoLE's total trainable params.
- Training control: max 10,000 steps, eval every 50 steps, early-stop after 10 evals without
  improvement; pick best checkpoint by avg validation accuracy.

Metrics: NER → F1/P/R; SA → EM and REM (relaxed exact match); EQA → EM and ROUGE-2. Load balance →
`MaxVio_global = max_i (Load_i − L̄oad_i)/L̄oad_i`.

Hardware in the paper: RTX 4090 (24GB) for 7B models, A100 (40GB) for 13B. PEFT params are only
0.65–1.05% of total, so inference latency ≈ base model.

## Notes for implementation

- Build incrementally and verify against the paper's numbers where feasible. The two-stage training
  and the routing-dimension broadcast (above) are the two places most likely to be implemented wrong.
- `peft`'s stock LoRA does not support this routing; the HiMoLE FFN layer is custom. Plan to wrap or
  replace the base model's MLP module rather than relying on `peft` injection alone.
- Diversity-loss sampling (every 10 layers, subset of experts) is an efficiency detail — correctness
  first, then add the sampling.

## GPU / eval engineering rules (do not violate)

These were caught by static analysis. Treat them as non-negotiable for every model and eval written.

1. **Move every model to GPU before the first eval.**
   `load_base_model()` sets `torch_dtype=bfloat16` but does NOT call `.to("cuda")`. Always call
   `model.to(device)` (where `device = "cuda" if torch.cuda.is_available() else "cpu"`) immediately
   after loading, before any `evaluate()` call. Forgetting this silently runs the A100 at 0% GPU
   utilization — `generate_answers` infers device from `next(model.parameters()).device`, so if the
   model is on CPU, all generation runs on CPU.

2. **Never evaluate one prompt at a time.**
   Single-prompt generation (`for p in prompts: model.generate(...)`) wastes >95% of A100 capacity.
   Always batch: tokenize a list of prompts with `padding=True`, pass as a single tensor, decode each
   output by slicing off the padded input length. Use `tokenizer.padding_side = "left"` for causal LMs
   during batched generation (right-padded inputs corrupt attention for all but the last token).

3. **Cap eval samples — never run full validation every N steps.**
   SQuAD validation is ~10k examples. At 50-step eval intervals over 10k training steps that is 200
   full evals. Even at 1 s/batch this is hours of pure eval overhead. Use `cfg.max_eval_samples`
   (default 500) for periodic and baseline evals. Reserve full-set eval for final reporting only.
