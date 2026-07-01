"""SQuAD-style EM and ROUGE-2 scoring for generated answers."""
from rouge_score import rouge_scorer

_SCORER = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles (a/an/the), and extra whitespace."""
    # TODO: implement SQuAD normalization:
    #   lower -> remove punctuation -> remove a/an/the -> collapse whitespace.
    #raise NotImplementedError("TODO: implement SQuAD answer normalization")
    s_lower = s.lower()
    s_no_punct = ''.join(c for c in s_lower if c.isalnum() or c.isspace())
    s_no_articles = ' '.join(word for word in s_no_punct.split() if word not in ('a', 'an', 'the'))
    s_normalized = ' '.join(s_no_articles.split())
    return s_normalized


def exact_match(pred: str, gold: str) -> float:
    """1.0 if normalized strings match exactly, else 0.0."""
    return float(normalize_answer(pred) == normalize_answer(gold))
    #raise NotImplementedError("TODO: implement exact match")


def rouge2(pred: str, gold: str) -> float:
    """ROUGE-2 F-measure between pred and gold."""
    return _SCORER.score(gold, pred)["rouge2"].fmeasure
    #raise NotImplementedError("TODO: implement rouge2")


def score_batch(preds, golds) -> dict:
    """Mean EM and ROUGE-2 over a list of predictions/golds."""
    em_scores = [exact_match(p, g) for p, g in zip(preds, golds)]
    rouge2_scores = [rouge2(p, g) for p, g in zip(preds, golds)]
    return {"em": sum(em_scores) / len(em_scores), "rouge2": sum(rouge2_scores) / len(rouge2_scores)}
