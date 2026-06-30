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
