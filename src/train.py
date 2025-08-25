
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
from joblib import dump
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from .data import prepare_data
from .models import get_models, get_param_grids

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True, parents=True)

def train_and_select_best(dataset: str, test_size: float, seed: int) -> Tuple[str, Any, float, dict]:
    bundle = prepare_data(dataset=dataset, test_size=test_size, seed=seed)
    models = get_models()
    grids = get_param_grids()

    best_name, best_model, best_auc = None, None, -np.inf
    best_cv_results = {}

    for name, model in models.items():
        param_grid = grids.get(name, {})
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        gs = GridSearchCV(model, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        gs.fit(bundle.X_train, bundle.y_train)

        y_val_prob = gs.predict_proba(bundle.X_test)[:, 1]
        auc = roc_auc_score(bundle.y_test, y_val_prob)

        best_cv_results[name] = {
            "best_params": gs.best_params_,
            "test_auc": float(auc)
        }

        if auc > best_auc:
            best_name, best_model, best_auc = name, gs.best_estimator_, auc

    # Save artifacts
    dump(best_model, ARTIFACTS / "model.joblib")
    dump(bundle.scaler, ARTIFACTS / "scaler.joblib")

    # Persist metadata
    meta = {
        "best_model_name": best_name,
        "best_test_auc": float(best_auc),
        "cv_results": best_cv_results,
        "feature_names": bundle.feature_names,
        "target_names": bundle.target_names
    }
    (ARTIFACTS / "train_meta.json").write_text(json.dumps(meta, indent=2))

    return best_name, best_model, best_auc, meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="breast_cancer")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    best_name, _, best_auc, meta = train_and_select_best(args.dataset, args.test_size, args.seed)
    print(f"Best model: {best_name} | Test ROC-AUC: {best_auc:.4f}")
    print("Artifacts saved in ./artifacts")

if __name__ == "__main__":
    main()
