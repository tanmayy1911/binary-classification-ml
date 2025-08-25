
from __future__ import annotations
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_models() -> Dict[str, Any]:
    return {
        "logreg": LogisticRegression(max_iter=200, solver="liblinear"),
        "rf": RandomForestClassifier(random_state=42),
        "svm": SVC(probability=True)
    }

def get_param_grids() -> Dict[str, dict]:
    return {
        "logreg": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"]
        },
        "rf": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        },
        "svm": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"]
        }
    }
