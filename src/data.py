
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

@dataclass
class DatasetBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    scaler: StandardScaler

def load_dataset(name: str = "breast_cancer") -> Tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Load a dataset and return (X_df, y_series, feature_names, target_names).

    Supported: 'breast_cancer' (sklearn).

    To use your own CSV:
      - Replace this function with something like:
        df = pd.read_csv('path/to/your.csv')
        y = df['target']  # adjust column
        X = df.drop(columns=['target'])
        feature_names = list(X.columns)
        target_names = ['class_0', 'class_1']
        return X, y, feature_names, target_names
    """
    if name == "breast_cancer":
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        return X, y, list(data.feature_names), list(data.target_names)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

def prepare_data(
    dataset: str = "breast_cancer",
    test_size: float = 0.2,
    seed: int = 42
) -> DatasetBundle:
    X_df, y_series, feature_names, target_names = load_dataset(dataset)

    # Basic cleaning (no missing in sklearn data, but keep as example)
    X_df = X_df.copy()
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_series.values, test_size=test_size, random_state=seed, stratify=y_series.values
    )

    return DatasetBundle(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        target_names=target_names,
        scaler=scaler
    )
