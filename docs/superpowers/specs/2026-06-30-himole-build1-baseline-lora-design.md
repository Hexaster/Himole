# Build 1 — Plain-LoRA Baseline (EQA / SQuAD)

**Date:** 2026-06-30
**Status:** Approved design, pending implementation plan
**Part of:** HiMoLE study reimplementation (see `CLAUDE.md`)

## Context

This is the first of several build cycles toward reimplementing HiMoLE. The overall
project decomposes into independent pieces, each with its own design → plan → build cycle:

1. **Build 1 (this doc): plain-LoRA baseline** — learn LoRA fine-tuning end-to-end and
   produce a real paper baseline.
2. Build 2: the custom HiMoLE FFN layer (hierarchical routing), built and unit-tested in isolation.
3. Build 3: Stage-1 training (sentence clustering + per-KCG independent training).
4. Build 4: Stage-2 training (joint, with aux + diversity losses).
5. Build 5: full evaluation across the paper's tasks.

We start with the **EQA / SQuAD** task: Llama-2-7B base, DeBERTa-v3-large clustering
encoder (not needed until Build 3), N=3 groups × M=4 experts (not needed until Build 2).
Build 1 only needs the base model + standard `peft` LoRA.

## Goals / Non-goals

**Goal:** A complete, runnable-*shaped* framework that fine-tunes Llama-2-7B on SQuAD with
standard `peft` LoRA and evaluates EM + ROUGE-2 on SQuAD (in-distribution) and NewsQA
(out-of-distribution). It doubles as the paper's **plain-LoRA baseline** (rank 80 / alpha 160,
chosen to match HiMoLE's trainable-param count).

**Study-repo philosophy:**
- Scaffolding (structure, signatures, data flow, config) is written and complete.
- Pedagogically-critical lines are left as `# TODO` for the user to fill — this is where the
  learning lives.
- Deep explanation lives in **notebook markdown cells**, so `.py` comments stay light.

**Non-goals for Build 1:**
- Multi-GPU / FSDP across the 4 A100s — single-GPU first; scale in a later build.
- Any HiMoLE-specific code (routing, experts, two-stage training, clustering).
- Hitting the exact paper number — correctness of the *pipeline* first.

## Environment (given)

- Remote server, **4× A100** (compute is not a constraint; multi-GPU deferred to later builds).
- Hugging Face access for gated Llama-2-7B handled by the user.
- Assume internet access on the server for model/dataset downloads.

## Format: hybrid notebooks + thin modules

- **`notebooks/`** — the teaching surface. Rich markdown (headers, LaTeX for math), small
  runnable demos on a *tiny* non-gated model, inspect-the-shape cells. Where TODOs get
  explained. Imports from `himole/`.
- **`himole/` (`.py` modules)** — the thin, reusable framework that notebooks AND the server
  training script import. Lightly commented (deep explanation is in the notebook).
- **`scripts/`** — the no-frills entry point launched on the A100s for the real run.

## Repo structure

```
himole/
  config.py            # one dataclass of all hyperparams (paper Table 7); TODO notes where paper is ambiguous
  data/
    squad.py           # load SQuAD -> build QA prompt -> tokenize w/ prompt-masking  [TODO: template + loss mask]
    newsqa.py          # load NewsQA for OOD eval (same prompt format)
  model/
    baseline_lora.py   # load Llama-2-7B, attach peft LoraConfig to FFN up/down/gate   [TODO: target_modules, r/alpha]
  train/
    loop.py            # transparent PyTorch loop: forward->loss->backward->step + eval gate [TODO: core step]
  eval/
    metrics.py         # EM + ROUGE-2                                                   [TODO: normalize + rouge]
    generate.py        # batched generation, then score with metrics.py
  utils.py             # seed, device, simple logging
notebooks/
  01_baseline_lora.ipynb   # full walkthrough on a tiny model: data -> LoRA -> a few steps -> eval
scripts/
  train_baseline.py    # entry point: build config -> data -> model -> run loop (the real A100 run)
tests/
  test_metrics.py      # EM/ROUGE-2 on known inputs (written; pass once metrics TODO is filled)
  test_data_masking.py # prompt + loss-mask correctness (written; pass once data TODO is filled)
requirements.txt
README.md              # how to run + the "fill the TODOs in this order" study guide
```

## Data flow

SQuAD example → prompt template (`Context: ... Question: ... Answer: ...`) → tokenize and
**mask the prompt tokens so causal-LM loss is computed only on the answer span** (key learning
point, left as TODO) → custom loop trains LoRA adapters only (base model frozen) → eval runs
generation on SQuAD + NewsQA, decodes, scores EM / ROUGE-2.

## Key config values (paper Table 7)

- Plain-LoRA baseline: **rank 80 / alpha 160**, dropout 0.05.
- Target modules: FFN **up / down / gate** projections of every transformer block.
- Optimizer AdamW, batch size 16, cutoff length 1024.
- Training control: max 10,000 steps, eval every 50 steps, early-stop after 10 evals without
  improvement, keep best checkpoint by avg validation accuracy.
- **Baseline LR is not isolated in the paper** → config value with a `TODO` note to confirm
  (HiMoLE itself uses 3e-4 Stage 1 / 3e-5 Stage 2).

## What is written vs TODO

- **Written:** all file/module structure, the config dataclass, data-loading wiring, the loop
  skeleton (logging, eval cadence, early stopping, checkpointing), the generation harness, the
  entry script, and the two test files.
- **TODO (the learning):** the prompt template + loss-masking, the `LoraConfig` wiring, the core
  training step (loss / backward / optimizer), and the metric implementations.

## Verification

- **Smoke-test path:** run the whole pipeline on a handful of examples with a tiny non-gated
  model (e.g. a small GPT-2-class model) to confirm shapes/flow without the gated 7B or a long
  download.
- **Unit tests:** `tests/test_metrics.py` and `tests/test_data_masking.py` are written up front;
  the user fills the corresponding TODOs until the tests pass.

## Open questions to resolve during planning

- Exact baseline LR (see config note).
- Tiny model id to use for the smoke test / notebook demo.
- Whether `generate.py` evaluates "accuracy" the same way the paper's early-stopping does, vs
  EM as the early-stop signal.
