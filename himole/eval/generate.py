"""Run generation on prompts and score with EM/ROUGE-2."""
import torch
from tqdm.auto import tqdm

from himole.eval.metrics import score_batch


@torch.no_grad()
def generate_answers(model, tokenizer, prompts, max_new_tokens=32, batch_size=16, cutoff_len=1024):
    """Greedy-decode answers in batches; return decoded answer strings."""
    model.eval()
    device = next(model.parameters()).device
    orig_padding_side = tokenizer.padding_side
    orig_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"      # causal LMs require left-padding for batched generation
    tokenizer.truncation_side = "left"   # keep the question and "Answer:" suffix when capped
    # Training turns the KV cache off (it clashes with gradient checkpointing), but generation
    # is under eval()/no_grad so re-enabling it here is safe and ~10x faster. Guarded because
    # test doubles may not carry a .config.
    model_cfg = getattr(model, "config", None)
    orig_use_cache = getattr(model_cfg, "use_cache", None)
    if model_cfg is not None:
        model_cfg.use_cache = True
    outs = []
    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="generating", unit="batch", leave=False):
            batch = prompts[i:i + batch_size]
            ids = tokenizer(batch, return_tensors="pt", truncation=True,
                            max_length=cutoff_len, padding=True).to(device)
            gen = model.generate(**ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
            input_len = ids["input_ids"].shape[1]
            for seq in gen:
                text = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
                outs.append(text.strip().split("\n")[0])    # first line = the answer
    finally:
        tokenizer.padding_side = orig_padding_side
        tokenizer.truncation_side = orig_truncation_side
        if model_cfg is not None:
            model_cfg.use_cache = orig_use_cache
    return outs


def evaluate(model, tokenizer, eval_pairs, cutoff_len=1024):
    """Generate answers for eval_pairs and return {'em','rouge2'}."""
    prompts = [e["prompt"] for e in eval_pairs]
    golds = [e["gold"] for e in eval_pairs]
    preds = generate_answers(model, tokenizer, prompts, cutoff_len=cutoff_len)
    return score_batch(preds, golds)
