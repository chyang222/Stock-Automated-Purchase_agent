import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from config.settings import PROCESSED_DIR, ARTIFACT_DIR

FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "return_1d", "return_5d",
    "ma5", "ma20", "ma60",
    "volatility_20", "rsi14",
]


def train_model():
    data_path = PROCESSED_DIR / "features.csv"
    df = pd.read_csv(data_path).sort_values(["stock_code", "date"]).reset_index(drop=True)

    X = df[FEATURE_COLUMNS]
    y = df["target"]
    split = int(len(df) * 0.8)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"
    )
    model.fit(X.iloc[:split], y.iloc[:split])

    pred = model.predict(X.iloc[split:])
    proba = model.predict_proba(X.iloc[split:])[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y.iloc[split:], pred)),
        "roc_auc": float(roc_auc_score(y.iloc[split:], proba))
        if len(set(y.iloc[split:])) > 1
        else None,
        "train_rows": split,
        "test_rows": len(df) - split,
    }

    model_path = ARTIFACT_DIR / "stock_updown_random_forest.joblib"
    metrics_path = ARTIFACT_DIR / "metrics.json"
    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"모델 저장: {model_path}")
    print(f"성능: {metrics}")
    return model, metrics


if __name__ == "__main__":
    train_model()
