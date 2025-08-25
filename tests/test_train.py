
from pathlib import Path
from src.train import train_and_select_best

def test_train_and_artifacts(tmp_path, monkeypatch):
    # run training with artifacts written to temp dir
    from src import train as train_module
    monkeypatch.setattr(train_module, "ARTIFACTS", tmp_path)

    name, model, auc, meta = train_and_select_best("breast_cancer", test_size=0.3, seed=0)
    assert name in ("logreg", "rf", "svm")
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "scaler.joblib").exists()
    assert "best_model_name" in meta
