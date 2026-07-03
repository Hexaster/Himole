"""Run generation on prompts and score with EM/ROUGE-2."""
import torch
from tqdm.auto import tqdm

from himole.eval.metrics import score_batch


@torch.no_grad()
def generate_answers(model, tokenizer, prompts, max_new_tokens=32, batch_size=16):
    """Greedy-decode answers in batches; return decoded answer strings."""
    model.eval()
    device = next(model.parameters()).device
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"     # causal LMs require left-padding for batched generation
    outs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="generating", unit="batch", leave=False):
        batch = prompts[i:i + batch_size]
        ids = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(device)
        gen = model.generate(**ids, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id)
        input_len = ids["input_ids"].shape[1]
        for seq in gen:
            text = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            outs.append(text.strip().split("\n")[0])    # first line = the answer
    tokenizer.padding_side = orig_padding_side
    return outs


def evaluate(model, tokenizer, eval_pairs):
    """Generate answers for eval_pairs and return {'em','rouge2'}."""
    prompts = [e["prompt"] for e in eval_pairs]
    golds = [e["gold"] for e in eval_pairs]
    preds = generate_answers(model, tokenizer, prompts)
    return score_batch(preds, golds)
