from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_loop_uses_tqdm_progress_bar():
    source = (ROOT / "himole" / "train" / "loop.py").read_text()
    assert "from tqdm.auto import tqdm" in source
    assert "tqdm(" in source


def test_tqdm_is_declared_as_dependency():
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    assert any(line.strip().startswith("tqdm") for line in requirements)
