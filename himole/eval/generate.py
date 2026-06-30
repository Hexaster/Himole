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
