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
    return PROMPT_TEMPLATE.format(context=context, question=question)



def format_example(example: dict, tokenizer, cutoff_len: int) -> dict:
    """Tokenize one example into input_ids/attention_mask/labels with prompt masking."""
    prompt = build_prompt(example["context"], example["question"])
    answer = " " + example["answers"]["text"][0] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids[:cutoff_len]
    # Reserve room for the answer; otherwise a long context can mask every label.
    prompt_ids = prompt_ids[:max(0, cutoff_len - len(answer_ids))]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "clustering_text": f"{example['context']}\n{example['question']}",
    }


def load_squad(tokenizer, cfg):
    """Load SQuAD and map format_example over train/validation splits."""
    ds = load_dataset(cfg.id_dataset)
    train = ds["train"]
    val = ds["validation"]
    if cfg.max_train_samples:                       # smoke-test shortcut
        train = train.select(range(cfg.max_train_samples))
    if cfg.max_eval_samples:
        val = val.select(range(min(len(val), cfg.max_eval_samples)))
    fn = lambda ex: format_example(ex, tokenizer, cfg.cutoff_len)
    cols = train.column_names
    return (train.map(fn, remove_columns=cols), val.map(fn, remove_columns=cols))
