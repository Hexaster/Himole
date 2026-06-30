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
    train = ds["train"]
    val = ds["validation"]
    if cfg.max_train_samples:                       # smoke-test shortcut
        train = train.select(range(cfg.max_train_samples))
        val = val.select(range(min(len(val), cfg.max_train_samples)))
    fn = lambda ex: format_example(ex, tokenizer, cfg.cutoff_len)
    cols = train.column_names
    return (train.map(fn, remove_columns=cols), val.map(fn, remove_columns=cols))
