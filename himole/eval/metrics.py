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
