# HiMoLE study reimplementation

A learning-focused reimplementation of **HiMoLE** (Hierarchical Mixture of LoRA Experts).
See [CLAUDE.md](CLAUDE.md) for the method overview and [docs/superpowers/](docs/superpowers/)
for the per-build design specs and plans.

This repo is built **bit by bit**. The code is a *scaffold*: the framework is written and
commented, and the pedagogically-critical lines are left as `# TODO` (they raise
`NotImplementedError`) for you to fill in. Tests are written up front and **fail** until you
fill the matching TODO.

## Build 1 — plain-LoRA baseline (EQA / SQuAD)

Fine-tune Qwen2.5-7B on SQuAD with standard `peft` LoRA; evaluate EM + ROUGE-2 on SQuAD (ID)
and NewsQA (OOD). This reproduces the paper's plain-LoRA baseline settings except for using
Qwen2.5-7B in place of Llama-2-7B (rank 80 / alpha 160).

### Install
```bash
pip install -r requirements.txt
```

### Run the tests (they will fail until you fill the TODOs)
```bash
pytest -v
```

### Fill the TODOs in this order
1. `himole/data/squad.py` → `build_prompt`, `format_example` → `pytest tests/test_data_masking.py`
2. `himole/eval/metrics.py` → `normalize_answer` / `exact_match` / `rouge2` / `score_batch` → `pytest tests/test_metrics.py`
3. `himole/model/baseline_lora.py` → `attach_lora` (the `LoraConfig`)
4. `himole/train/loop.py` → the forward/backward/optimizer step
5. `himole/data/newsqa.py` → the OOD loader

The notebook `notebooks/01_baseline_lora.ipynb` walks through all of these on a tiny model.

### Smoke test (tiny model, seconds, CPU-friendly)
```bash
python scripts/train_baseline.py --tiny --steps 5 --samples 8
```

### Real run (server, after filling TODOs)
Set `use_tiny=False` (the default) and run:
```bash
python scripts/train_baseline.py
```

## Layout
```
himole/        core package (config, data, model, train, eval)
notebooks/     teaching walkthrough(s)
scripts/       entry point(s) for real runs
tests/         unit tests (fail until TODOs are filled)
docs/          design specs + implementation plans
paper/         the source paper + notes
```
