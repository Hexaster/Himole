# Build 1 — Plain-LoRA Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, runnable-*shaped*, heavily-explained scaffold that fine-tunes Llama-2-7B on SQuAD with standard `peft` LoRA and evaluates EM + ROUGE-2 on SQuAD (ID) and NewsQA (OOD), with the pedagogically-critical lines left as `# TODO` for the user to fill.

**Architecture:** A thin reusable Python package (`himole/`) imported by both a teaching notebook (`notebooks/01_baseline_lora.ipynb`) and a server entry script (`scripts/train_baseline.py`). The package is split by responsibility: config, data, model, train, eval. A transparent hand-written PyTorch training loop (no HF Trainer). Critical lines raise `NotImplementedError("TODO: ...")` so the scaffold runs up to the learning points; fully-written tests fail with those messages until the user fills them.

**Tech Stack:** Python 3.8+, PyTorch, HuggingFace `transformers` + `peft` + `datasets`, `pytest`. Single-GPU (or CPU for the tiny-model smoke test). Multi-GPU is deliberately out of scope.

## Global Constraints

- Study-repo style: scaffolding complete; critical lines are `# TODO` raising `NotImplementedError("TODO: <what to do>")`. Deep explanation lives in notebook markdown, `.py` comments stay light but present on every chunk.
- Base model (real run): `meta-llama/Llama-2-7b-hf` (gated; user handles access).
- Tiny model (notebook + smoke test): `sshleifer/tiny-gpt2` (no download gate, runs on CPU).
- LoRA baseline hyperparams (paper Table 7): rank `80`, alpha `160`, dropout `0.05`, target = FFN up/down/gate projections.
- Optimizer AdamW, batch size `16`, cutoff length `1024`. Training control: max `10000` steps, eval every `50` steps, early-stop after `10` stale evals, keep best checkpoint.
- Baseline LR not isolated in paper → default `3e-4` in config with a TODO note.
- Datasets: SQuAD = `rajpurkar/squad` (ID), NewsQA = OOD eval only.
- Tests live in `tests/`, run with `pytest`. Tests are written in full and are EXPECTED TO FAIL until the user fills the corresponding TODO. Task completion = scaffold present + imports clean + tests fail with the TODO message (not with import/collection errors).

---

### Task 1: Project scaffolding + config

**Files:**
- Create: `requirements.txt`
- Create: `himole/__init__.py`, `himole/data/__init__.py`, `himole/model/__init__.py`, `himole/train/__init__.py`, `himole/eval/__init__.py`, `tests/__init__.py`
- Create: `himole/config.py`
- Create: `himole/utils.py`

**Interfaces:**
- Produces: `BaselineConfig` dataclass with all hyperparams; `set_seed(seed:int)`, `get_device()->str`, `get_logger(name:str)` in `utils.py`.

- [ ] **Step 1: Write `requirements.txt`**

```
torch>=2.0
transformers>=4.38
peft>=0.9
datasets>=2.16
evaluate>=0.4
rouge-score>=0.1.2
accelerate>=0.27
pytest>=7
```

- [ ] **Step 2: Create empty `__init__.py` files** for `himole/`, `himole/data/`, `himole/model/`, `himole/train/`, `himole/eval/`, `tests/` (each a 1-line module docstring).

- [ ] **Step 3: Write `himole/config.py`** — a single fully-written dataclass (no TODO except the LR note):

```python
"""All Build-1 hyperparameters in one place. Values from the HiMoLE paper (Table 7)."""
from dataclasses import dataclass

@dataclass
class BaselineConfig:
    # --- models ---
    base_model: str = "meta-llama/Llama-2-7b-hf"   # real run (gated)
    tiny_model: str = "sshleifer/tiny-gpt2"        # smoke test / notebook
    use_tiny: bool = False                          # flip True for fast local runs

    # --- LoRA (plain-LoRA baseline matches HiMoLE param count) ---
    lora_r: int = 80
    lora_alpha: int = 160
    lora_dropout: float = 0.05
    # FFN projection names for Llama; overridden for tiny-gpt2 in model loader.
    target_modules: tuple = ("up_proj", "down_proj", "gate_proj")

    # --- data ---
    id_dataset: str = "rajpurkar/squad"
    cutoff_len: int = 1024
    max_train_samples: int | None = None  # None = full; set small for smoke test

    # --- optim / loop ---
    lr: float = 3e-4          # TODO: paper does not isolate the plain-LoRA LR; confirm.
    batch_size: int = 16
    max_steps: int = 10000
    eval_every: int = 50
    early_stop_patience: int = 10  # number of stale evals before stopping
    seed: int = 42

    # --- io ---
    output_dir: str = "outputs/baseline_lora"
```

- [ ] **Step 4: Write `himole/utils.py`** — fully written helpers:

```python
"""Small shared helpers: reproducibility, device, logging."""
import logging, random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Seed python/numpy/torch so runs are reproducible."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> str:
    """Return 'cuda' if a GPU is visible, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_logger(name: str) -> logging.Logger:
    """A logger that prints step/eval info to stdout."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(name)
```

- [ ] **Step 5: Verify imports**

Run: `python -c "from himole.config import BaselineConfig; from himole.utils import set_seed, get_device, get_logger; print(BaselineConfig().lora_r)"`
Expected: prints `80`, no errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt himole tests
git commit -m "feat(build1): project scaffold, config, utils"
```

---

### Task 2: Data — SQuAD/NewsQA loading, prompt + loss masking (TODO) + tests

**Files:**
- Create: `himole/data/squad.py`
- Create: `himole/data/newsqa.py`
- Test: `tests/test_data_masking.py`

**Interfaces:**
- Consumes: `BaselineConfig` (for `cutoff_len`, `id_dataset`, `max_train_samples`).
- Produces:
  - `build_prompt(context:str, question:str) -> str` (prompt WITHOUT the answer).
  - `format_example(example:dict, tokenizer, cutoff_len:int) -> dict` returning `{"input_ids", "attention_mask", "labels"}` where `labels` equals `input_ids` but with prompt positions set to `-100` (loss masked).
  - `load_squad(tokenizer, cfg) -> (train_ds, val_ds)` tokenized.
  - `load_newsqa(tokenizer, cfg) -> list[{"prompt","gold"}]` (OOD) — raw eval pairs, not loss-masked.

- [ ] **Step 1: Write the failing test `tests/test_data_masking.py`** (fully written):

```python
"""Tests for prompt construction and loss masking. Fail until the TODOs in data/squad.py are filled."""
from transformers import AutoTokenizer
from himole.data.squad import build_prompt, format_example

TOK = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
if TOK.pad_token is None:
    TOK.pad_token = TOK.eos_token

def test_prompt_excludes_answer():
    p = build_prompt("Paris is the capital of France.", "What is the capital of France?")
    assert "France" in p and "Paris" in p
    assert "Answer:" in p          # template marks where the answer goes
    assert p.rstrip().endswith("Answer:")  # prompt stops before the answer text

def test_labels_mask_prompt_tokens():
    ex = {"context": "Paris is the capital of France.",
          "question": "What is the capital of France?",
          "answers": {"text": ["Paris"]}}
    out = format_example(ex, TOK, cutoff_len=64)
    assert set(out.keys()) >= {"input_ids", "attention_mask", "labels"}
    assert len(out["input_ids"]) == len(out["labels"])
    # at least one masked (prompt) and one unmasked (answer) label
    assert any(l == -100 for l in out["labels"])
    assert any(l != -100 for l in out["labels"])
    # every unmasked label must equal the corresponding input id (next-token target)
    for i, l in enumerate(out["labels"]):
        if l != -100:
            assert l == out["input_ids"][i]
```

- [ ] **Step 2: Run test to verify it fails for the RIGHT reason**

Run: `pytest tests/test_data_masking.py -v`
Expected: FAIL with `NotImplementedError: TODO: ...` (NOT ImportError / collection error).

- [ ] **Step 3: Write `himole/data/squad.py` scaffold with TODOs**

```python
"""Load SQuAD, turn each QA example into a causal-LM training example.

The key learning point here is the LOSS MASK: we want the model to learn to
*produce the answer given the prompt*, so we compute loss only on answer tokens.
We do that by setting label = -100 on every prompt token (PyTorch CrossEntropy
ignores -100).
"""
from datasets import load_dataset

# Prompt template. The model sees everything up to and including "Answer:" and
# must generate the answer text after it.
PROMPT_TEMPLATE = (
    "Answer the question using the context.\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Answer:"
)

def build_prompt(context: str, question: str) -> str:
    """Return the prompt (no answer). Used for both training and generation."""
    # TODO: format PROMPT_TEMPLATE with context/question and return it.
    raise NotImplementedError("TODO: build the prompt from PROMPT_TEMPLATE")

def format_example(example: dict, tokenizer, cutoff_len: int) -> dict:
    """Tokenize one example into input_ids/attention_mask/labels with prompt masking."""
    prompt = build_prompt(example["context"], example["question"])
    answer = " " + example["answers"]["text"][0] + tokenizer.eos_token
    # TODO (the core learning point):
    #   1. tokenize `prompt` and `prompt + answer` (no special tokens) up to cutoff_len.
    #   2. input_ids = ids of (prompt + answer).
    #   3. labels = copy of input_ids, but set the FIRST len(prompt_ids) labels to -100.
    #   4. attention_mask = all 1s (we pad later in the collator).
    #   5. return {"input_ids", "attention_mask", "labels"}.
    raise NotImplementedError("TODO: tokenize and build the loss mask")

def load_squad(tokenizer, cfg):
    """Load SQuAD and map format_example over train/validation splits."""
    ds = load_dataset(cfg.id_dataset)
    train = ds["train"]; val = ds["validation"]
    if cfg.max_train_samples:                       # smoke-test shortcut
        train = train.select(range(cfg.max_train_samples))
        val = val.select(range(min(len(val), cfg.max_train_samples)))
    fn = lambda ex: format_example(ex, tokenizer, cfg.cutoff_len)
    cols = train.column_names
    return (train.map(fn, remove_columns=cols), val.map(fn, remove_columns=cols))
```

- [ ] **Step 4: Write `himole/data/newsqa.py` scaffold with TODO**

```python
"""Load NewsQA as an OUT-OF-DISTRIBUTION eval set.

We only need raw (prompt, gold_answer) pairs for generation-time scoring here,
not loss-masked training tensors.
"""
from himole.data.squad import build_prompt

def load_newsqa(tokenizer, cfg):
    """Return a list of {"prompt", "gold"} dicts for OOD evaluation."""
    # TODO: pick a NewsQA source on the Hub (e.g. "lucadiliello/newsqa" or a mirror),
    #       map each example to {"prompt": build_prompt(context, question),
    #       "gold": first answer text}. Respect cfg.max_train_samples for smoke tests.
    raise NotImplementedError("TODO: load NewsQA and return prompt/gold pairs")
```

- [ ] **Step 5: Re-run test — confirm it now fails inside `format_example`'s TODO**

Run: `pytest tests/test_data_masking.py -v`
Expected: still FAIL with `NotImplementedError: TODO: ...` (proves scaffold wiring is correct).

- [ ] **Step 6: Commit**

```bash
git add himole/data tests/test_data_masking.py
git commit -m "feat(build1): data scaffold (prompt+mask TODO) and masking tests"
```

---

### Task 3: Eval metrics — EM + ROUGE-2 (TODO) + tests

**Files:**
- Create: `himole/eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `normalize_answer(s:str) -> str` (lowercase, strip punctuation/articles/whitespace — SQuAD style).
  - `exact_match(pred:str, gold:str) -> float` (1.0/0.0 after normalization).
  - `rouge2(pred:str, gold:str) -> float` (ROUGE-2 F-measure).
  - `score_batch(preds:list[str], golds:list[str]) -> dict` → `{"em": float, "rouge2": float}` (means).

- [ ] **Step 1: Write the failing test `tests/test_metrics.py`** (fully written):

```python
"""Tests for EM and ROUGE-2. Fail until metrics.py TODOs are filled."""
from himole.eval.metrics import normalize_answer, exact_match, rouge2, score_batch

def test_normalize_strips_articles_punct_case():
    assert normalize_answer("The  Paris, ") == "paris"

def test_exact_match_after_normalization():
    assert exact_match("the Paris", "Paris.") == 1.0
    assert exact_match("London", "Paris") == 0.0

def test_rouge2_identical_is_one():
    assert rouge2("the quick brown fox", "the quick brown fox") == 1.0

def test_rouge2_disjoint_is_zero():
    assert rouge2("alpha beta gamma", "delta epsilon zeta") == 0.0

def test_score_batch_means():
    out = score_batch(["Paris", "London"], ["Paris", "Paris"])
    assert out["em"] == 0.5
    assert "rouge2" in out
```

- [ ] **Step 2: Run test to verify it fails for the right reason**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `NotImplementedError: TODO: ...`.

- [ ] **Step 3: Write `himole/eval/metrics.py` scaffold with TODOs**

```python
"""SQuAD-style EM and ROUGE-2 scoring for generated answers."""
from rouge_score import rouge_scorer

_SCORER = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles (a/an/the), and extra whitespace."""
    # TODO: implement SQuAD normalization:
    #   lower -> remove punctuation -> remove a/an/the -> collapse whitespace.
    raise NotImplementedError("TODO: implement SQuAD answer normalization")

def exact_match(pred: str, gold: str) -> float:
    """1.0 if normalized strings match exactly, else 0.0."""
    # TODO: return float(normalize_answer(pred) == normalize_answer(gold))
    raise NotImplementedError("TODO: implement exact match")

def rouge2(pred: str, gold: str) -> float:
    """ROUGE-2 F-measure between pred and gold."""
    # TODO: return _SCORER.score(gold, pred)["rouge2"].fmeasure
    raise NotImplementedError("TODO: implement rouge2")

def score_batch(preds, golds) -> dict:
    """Mean EM and ROUGE-2 over a list of predictions/golds."""
    # TODO: average exact_match and rouge2 over all pairs; return {"em":..,"rouge2":..}.
    raise NotImplementedError("TODO: implement batch scoring")
```

- [ ] **Step 4: Re-run test — confirm `NotImplementedError`**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `NotImplementedError: TODO: ...`.

- [ ] **Step 5: Commit**

```bash
git add himole/eval/metrics.py tests/test_metrics.py
git commit -m "feat(build1): metrics scaffold (EM/ROUGE-2 TODO) and tests"
```

---

### Task 4: Model — load base + attach LoRA (TODO)

**Files:**
- Create: `himole/model/baseline_lora.py`

**Interfaces:**
- Consumes: `BaselineConfig`.
- Produces:
  - `load_tokenizer(cfg) -> tokenizer` (sets pad_token if missing).
  - `load_base_model(cfg) -> nn.Module` (chooses tiny vs base, bf16 on GPU).
  - `attach_lora(model, cfg) -> peft_model` (LoRA on FFN proj; prints trainable %).

- [ ] **Step 1: Write `himole/model/baseline_lora.py` scaffold with TODOs**

```python
"""Load the base LLM (frozen) and attach trainable LoRA adapters with peft."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def load_tokenizer(cfg):
    """Load tokenizer; causal LMs often lack a pad token, so reuse eos."""
    name = cfg.tiny_model if cfg.use_tiny else cfg.base_model
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_base_model(cfg):
    """Load the frozen base model (bf16 on GPU, fp32 on CPU for the tiny model)."""
    name = cfg.tiny_model if cfg.use_tiny else cfg.base_model
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return model

def attach_lora(model, cfg):
    """Wrap the base model with LoRA adapters on the FFN projections."""
    # tiny-gpt2 has no up/down/gate_proj; its MLP uses 'c_fc'/'c_proj'.
    targets = ["c_fc", "c_proj"] if cfg.use_tiny else list(cfg.target_modules)
    # TODO (the core learning point): build a LoraConfig with
    #   r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
    #   target_modules=targets, task_type="CAUSAL_LM", bias="none".
    #   Then peft_model = get_peft_model(model, lora_config).
    #   Finally peft_model.print_trainable_parameters() and return peft_model.
    raise NotImplementedError("TODO: build LoraConfig and wrap with get_peft_model")
```

- [ ] **Step 2: Verify import + tokenizer path on the tiny model**

Run: `python -c "from himole.config import BaselineConfig; from himole.model.baseline_lora import load_tokenizer; c=BaselineConfig(); c.use_tiny=True; print(load_tokenizer(c).pad_token is not None)"`
Expected: prints `True`.

- [ ] **Step 3: Commit**

```bash
git add himole/model/baseline_lora.py
git commit -m "feat(build1): model loader scaffold (LoRA wiring TODO)"
```

---

### Task 5: Generation harness for eval (written)

**Files:**
- Create: `himole/eval/generate.py`

**Interfaces:**
- Consumes: `score_batch` (Task 3), a model + tokenizer.
- Produces: `generate_answers(model, tokenizer, prompts, max_new_tokens=32) -> list[str]`; `evaluate(model, tokenizer, eval_pairs) -> dict` where `eval_pairs` is a list of `{"prompt","gold"}`.

- [ ] **Step 1: Write `himole/eval/generate.py`** (fully written — this is harness, not a learning point):

```python
"""Run generation on prompts and score with EM/ROUGE-2."""
import torch
from himole.eval.metrics import score_batch

@torch.no_grad()
def generate_answers(model, tokenizer, prompts, max_new_tokens=32):
    """Greedy-decode an answer for each prompt; return decoded answer strings."""
    model.eval()
    device = next(model.parameters()).device
    outs = []
    for p in prompts:                                   # simple per-prompt loop (clear > fast)
        ids = tokenizer(p, return_tensors="pt", truncation=True).to(device)
        gen = model.generate(**ids, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id)
        # keep only the newly generated tokens (after the prompt)
        new_tokens = gen[0, ids["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        outs.append(text.strip().split("\n")[0])        # first line = the answer
    return outs

def evaluate(model, tokenizer, eval_pairs):
    """Generate answers for eval_pairs and return {'em','rouge2'}."""
    prompts = [e["prompt"] for e in eval_pairs]
    golds = [e["gold"] for e in eval_pairs]
    preds = generate_answers(model, tokenizer, prompts)
    return score_batch(preds, golds)
```

- [ ] **Step 2: Verify import**

Run: `python -c "from himole.eval.generate import generate_answers, evaluate; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add himole/eval/generate.py
git commit -m "feat(build1): generation+eval harness"
```

---

### Task 6: Transparent training loop (core step = TODO)

**Files:**
- Create: `himole/train/loop.py`

**Interfaces:**
- Consumes: `BaselineConfig`, a peft model, tokenizer, train dataset, ID eval pairs.
- Produces:
  - `collate(batch, tokenizer) -> dict` (pad input_ids/labels to batch max; labels pad = -100).
  - `train(model, tokenizer, train_ds, val_eval_pairs, cfg) -> dict` (runs the loop, returns best metrics).

- [ ] **Step 1: Write `himole/train/loop.py` scaffold** — skeleton written, the core optimization step is the TODO:

```python
"""A transparent, hand-written training loop so every step of LoRA fine-tuning
is visible: build batches -> forward -> loss -> backward -> optimizer step,
with periodic eval, early stopping, and best-checkpoint saving.
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from himole.eval.generate import evaluate
from himole.utils import get_logger

log = get_logger("train")

def collate(batch, tokenizer):
    """Pad a list of {input_ids,attention_mask,labels} to the batch's max length."""
    maxlen = max(len(b["input_ids"]) for b in batch)
    pad_id = tokenizer.pad_token_id
    def pad(seq, value): return seq + [value] * (maxlen - len(seq))
    input_ids = torch.tensor([pad(b["input_ids"], pad_id) for b in batch])
    attn = torch.tensor([pad(b["attention_mask"], 0) for b in batch])
    labels = torch.tensor([pad(b["labels"], -100) for b in batch])  # -100 = ignore in loss
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

def train(model, tokenizer, train_ds, val_eval_pairs, cfg):
    """Run the LoRA fine-tuning loop. Returns the best eval metrics seen."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer))
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr)

    best_em, stale, step = -1.0, 0, 0
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.train()
    while step < cfg.max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # TODO (the core learning point): one optimization step.
            #   1. outputs = model(**batch)        # HF returns outputs.loss (CE on labels)
            #   2. loss = outputs.loss
            #   3. loss.backward()
            #   4. optimizer.step()
            #   5. optimizer.zero_grad()
            raise NotImplementedError("TODO: implement the forward/backward/step")

            step += 1
            if step % cfg.eval_every == 0:                 # periodic eval + early stop
                metrics = evaluate(model, tokenizer, val_eval_pairs)
                model.train()
                log.info(f"step {step} loss {loss.item():.4f} eval {metrics}")
                if metrics["em"] > best_em:
                    best_em = metrics["em"]; stale = 0
                    model.save_pretrained(os.path.join(cfg.output_dir, "best"))
                else:
                    stale += 1
                if stale >= cfg.early_stop_patience or step >= cfg.max_steps:
                    return {"best_em": best_em, "last": metrics}
    return {"best_em": best_em}
```

- [ ] **Step 2: Verify import**

Run: `python -c "from himole.train.loop import train, collate; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add himole/train/loop.py
git commit -m "feat(build1): transparent training loop (core step TODO)"
```

---

### Task 7: Server entry script (written)

**Files:**
- Create: `scripts/train_baseline.py`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write `scripts/train_baseline.py`** (fully written):

```python
"""Entry point for the real A100 run: build config -> data -> model -> train -> eval OOD.

Usage:
    python scripts/train_baseline.py                 # full run on Llama-2-7B
    python scripts/train_baseline.py --tiny --steps 5 --samples 8   # smoke test
"""
import argparse
from datasets import load_dataset
from himole.config import BaselineConfig
from himole.utils import set_seed
from himole.model.baseline_lora import load_tokenizer, load_base_model, attach_lora
from himole.data.squad import load_squad, build_prompt
from himole.data.newsqa import load_newsqa
from himole.train.loop import train
from himole.eval.generate import evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="use tiny-gpt2 for a fast smoke test")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    args = ap.parse_args()

    cfg = BaselineConfig()
    if args.tiny: cfg.use_tiny = True
    if args.steps is not None: cfg.max_steps = args.steps
    if args.samples is not None: cfg.max_train_samples = args.samples
    set_seed(cfg.seed)

    tokenizer = load_tokenizer(cfg)
    model = attach_lora(load_base_model(cfg), cfg)

    train_ds, _ = load_squad(tokenizer, cfg)

    # Build (prompt, gold) eval pairs for ID validation from raw SQuAD val split.
    raw_val = load_dataset(cfg.id_dataset)["validation"]
    if cfg.max_train_samples:
        raw_val = raw_val.select(range(cfg.max_train_samples))
    id_pairs = [{"prompt": build_prompt(e["context"], e["question"]),
                 "gold": e["answers"]["text"][0]} for e in raw_val]

    result = train(model, tokenizer, train_ds, id_pairs, cfg)
    print("TRAIN RESULT:", result)

    ood_pairs = load_newsqa(tokenizer, cfg)        # OOD generalization check
    print("OOD (NewsQA):", evaluate(model, tokenizer, ood_pairs))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses args and fails only at the first TODO**

Run: `python scripts/train_baseline.py --tiny --steps 2 --samples 4`
Expected: FAIL with `NotImplementedError: TODO: build LoraConfig ...` (proves wiring is correct up to the first learning point).

- [ ] **Step 3: Commit**

```bash
git add scripts/train_baseline.py
git commit -m "feat(build1): server entry script"
```

---

### Task 8: Teaching notebook

**Files:**
- Create: `notebooks/01_baseline_lora.ipynb`

**Interfaces:**
- Consumes: the whole `himole/` package, using `cfg.use_tiny=True`.

- [ ] **Step 1: Create the notebook** with this cell sequence (markdown cells carry the explanation; code cells import from `himole/` and run on tiny-gpt2):

  1. **md** — "Build 1: LoRA fine-tuning, end to end" intro + what LoRA is (B·A low-rank update, base frozen), with the equation `W' = W + (α/r)·B·A` in LaTeX.
  2. **code** — `from himole.config import BaselineConfig; cfg = BaselineConfig(); cfg.use_tiny=True; cfg.max_train_samples=8`.
  3. **md** — "Step 1: Data & the loss mask" — explain why we mask prompt tokens (-100).
  4. **code** — load tokenizer, build one example with `format_example`, print `input_ids` vs `labels` side by side to SEE the mask. (Will raise until that TODO is filled — note this in the md above it.)
  5. **md** — "Step 2: Attach LoRA" — explain rank/alpha and target modules.
  6. **code** — `attach_lora(load_base_model(cfg), cfg)` and read the printed trainable %.
  7. **md** — "Step 3: One training step" — explain forward→loss→backward→step.
  8. **code** — call `train` with `cfg.max_steps=2` on the 8-sample subset.
  9. **md** — "Step 4: Evaluate EM / ROUGE-2".
  10. **code** — run `evaluate` on a few ID pairs; print metrics.
  11. **md** — "Your homework: fill the TODOs in this order" — list the 4 TODO sites and which test proves each.

- [ ] **Step 2: Commit**

```bash
git add notebooks/01_baseline_lora.ipynb
git commit -m "docs(build1): teaching notebook walkthrough"
```

---

### Task 9: README study guide

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`** covering: project purpose (link `CLAUDE.md`), install (`pip install -r requirements.txt`), the smoke test command (`python scripts/train_baseline.py --tiny --steps 5 --samples 8`), how to run tests (`pytest -v`), and the ordered TODO checklist:
  1. `himole/data/squad.py` → `build_prompt`, `format_example` → `pytest tests/test_data_masking.py`
  2. `himole/eval/metrics.py` → normalization/EM/ROUGE-2 → `pytest tests/test_metrics.py`
  3. `himole/model/baseline_lora.py` → `attach_lora` LoraConfig
  4. `himole/train/loop.py` → the forward/backward/step
  5. `himole/data/newsqa.py` → OOD loader
  Then: run the smoke test end-to-end on tiny-gpt2; finally flip `use_tiny=False` for the real Llama-2-7B run.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(build1): README study guide"
```

---

## Self-Review Notes

- **Spec coverage:** repo structure (Tasks 1–9), data flow + masking (Task 2), config values (Task 1), written-vs-TODO split (each task marks its TODO), smoke test (Tasks 7–8), unit tests (Tasks 2–3), multi-GPU excluded (no task adds it). All spec sections covered.
- **Open questions from spec:** baseline LR → defaulted `3e-4` with TODO note in config (Task 1); tiny model → `sshleifer/tiny-gpt2` (Global Constraints); early-stop signal → EM (Task 6 uses `metrics["em"]`).
- **Type consistency:** `format_example`/`build_prompt` (Task 2) consumed by Tasks 6–7; `score_batch` (Task 3) consumed by Task 5; `evaluate`/`generate_answers` (Task 5) consumed by Tasks 6–7; `attach_lora`/`load_*` (Task 4) consumed by Task 7. Names match across tasks.
