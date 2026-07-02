"""Run generation on prompts and score with EM/ROUGE-2."""
import torch
from tqdm.auto import tqdm


@torch.no_grad()
def generate_answers(model, tokenizer, prompts, max_new_tokens=32, batch_size=8):
    """Greedy-decode an answer for each prompt; return decoded answer strings."""
    model.eval()
    device = next(model.parameters()).device
    outs = []
    batch_size = max(1, batch_size)
    old_padding_side = getattr(tokenizer, "padding_side", None)
    if old_padding_side is not None:
        tokenizer.padding_side = "left"
    try:
        for start in tqdm(range(0, len(prompts), batch_size), desc="generating",
                          unit="batch", leave=False):
            batch_prompts = prompts[start:start + batch_size]
            ids = tokenizer(batch_prompts, return_tensors="pt", truncation=True,
                            padding=True).to(device)
            gen = model.generate(**ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
            # keep only the newly generated tokens (after the padded prompt)
            for row in gen[:, ids["input_ids"].shape[1]:]:
                text = tokenizer.decode(row, skip_special_tokens=True)
                outs.append(text.strip().split("\n")[0])        # first line = the answer
    finally:
        if old_padding_side is not None:
            tokenizer.padding_side = old_padding_side
    return outs


def evaluate(model, tokenizer, eval_pairs, batch_size=8):
    """Generate answers for eval_pairs and return {'em','rouge2'}."""
    from himole.eval.metrics import score_batch

    prompts = [e["prompt"] for e in eval_pairs]
    golds = [e["gold"] for e in eval_pairs]
    preds = generate_answers(model, tokenizer, prompts, batch_size=batch_size)
    return score_batch(preds, golds)
