from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_loop_uses_tqdm_progress_bar():
    source = (ROOT / "himole" / "train" / "loop.py").read_text()
    assert "from tqdm.auto import tqdm" in source
    assert "tqdm(" in source


def test_stage1_reports_embedding_and_kcg_progress():
    clustering_source = (ROOT / "himole" / "data" / "clustering.py").read_text()
    stage1_source = (ROOT / "himole" / "train" / "stage1.py").read_text()
    assert 'desc="stage 1: embeddings"' in clustering_source
    assert 'f"stage 1: KCG {group_id + 1}/{cfg.num_kcgs}"' in stage1_source


def test_stage2_uses_progress_and_adapter_only_checkpoints():
    source = (ROOT / "himole" / "train" / "stage2.py").read_text()
    assert 'desc="stage 2"' in source
    assert "torch.save(best_state" in source
    assert "model.save_pretrained" not in source


def test_tqdm_is_declared_as_dependency():
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    assert any(line.strip().startswith("tqdm") for line in requirements)


def test_sentencepiece_is_declared_for_clustering_tokenizer():
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    assert any(line.strip().startswith("sentencepiece") for line in requirements)
