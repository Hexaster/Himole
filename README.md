# HiMoLE study reimplementation

A learning-focused reimplementation of **HiMoLE** (Hierarchical Mixture of LoRA Experts).
See [CLAUDE.md](CLAUDE.md) for the method overview and [docs/superpowers/](docs/superpowers/)
for the per-build design specs and plans.

This repo is built **bit by bit**. The plain-LoRA baseline and the HiMoLE two-stage
implementation are runnable; unit tests cover prompt masking, metrics, routing, losses,
patching, clustering, and one-step Stage-1/Stage-2 training.

## Build 1 — plain-LoRA baseline (EQA / SQuAD)

Fine-tune Qwen2.5-7B on SQuAD with standard `peft` LoRA; evaluate EM + ROUGE-2 on SQuAD (ID)
and NewsQA (OOD). This reproduces the paper's plain-LoRA baseline settings except for using
Qwen2.5-7B in place of Llama-2-7B (rank 80 / alpha 160).

### Install
```bash
pip install -r requirements.txt
```

### Run the tests
```bash
pytest -v
```

The notebook `notebooks/01_baseline_lora.ipynb` walks through the baseline components on a tiny model.

### Smoke test (tiny model, seconds, CPU-friendly)
```bash
python scripts/train_baseline.py --tiny --steps 5 --samples 8
```

### Real run (server)
Set `use_tiny=False` (the default) and run:
```bash
python scripts/train_baseline.py
```

## HiMoLE two-stage run

```bash
python scripts/train_himole.py --tiny --steps 5 --stage1-steps 5 --samples 8
```

The full run requires CUDA and uses DeBERTa-v3-large for Stage-1 clustering. The tiny mode uses
the tiny base model as the clustering encoder for a practical smoke test.

## Layout
```
himole/        core package (config, data, model, train, eval)
notebooks/     teaching walkthrough(s)
scripts/       entry point(s) for real runs
tests/         unit tests
docs/          design specs + implementation plans
paper/         the source paper + notes
```
