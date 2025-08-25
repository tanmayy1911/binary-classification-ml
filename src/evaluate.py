
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from .data import prepare_data

ARTIFACTS = Path("artifacts")
PLOTS = ARTIFACTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

def evaluate(dataset: str = "breast_cancer", test_size: float = 0.2, seed: int = 42):
    bundle = prepare_data(dataset=dataset, test_size=test_size, seed=seed)

    model_path = ARTIFACTS / "model.joblib"
    scaler_path = ARTIFACTS / "scaler.joblib"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Model or scaler not found. Train first: python -m src.train")

    model = load(model_path)

    y_prob = model.predict_proba(bundle.X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(bundle.y_test, y_pred, output_dict=True)
    cm = confusion_matrix(bundle.y_test, y_pred)
    fpr, tpr, _ = roc_curve(bundle.y_test, y_prob)
    auc = roc_auc_score(bundle.y_test, y_prob)

    # Save metrics
    metrics = {
        "roc_auc": float(auc),
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ROC plot
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS / "roc_curve.png", dpi=150)
    plt.close()

    # Confusion matrix plot (simple, no seaborn)
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar(im)
    # Add counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(PLOTS / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Saved metrics to {ARTIFACTS/'metrics.json'} and plots to {PLOTS}")

if __name__ == "__main__":
    evaluate()
