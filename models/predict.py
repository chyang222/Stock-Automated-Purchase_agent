import joblib
import pandas as pd

from config.settings import PROCESSED_DIR, ARTIFACT_DIR

FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "return_1d", "return_5d",
    "ma5", "ma20", "ma60",
    "volatility_20", "rsi14",
]


def predict_latest() -> pd.DataFrame:
    model = joblib.load(ARTIFACT_DIR / "stock_updown_random_forest.joblib")
    df = pd.read_csv(PROCESSED_DIR / "features.csv")

    latest = (
        df.sort_values(["stock_code", "date"])
        .groupby("stock_code")
        .tail(1)
        .copy()
    )
    latest["up_probability"] = model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    result = latest[["date", "stock_code", "close", "up_probability"]]
    print(result)
    return result


if __name__ == "__main__":
    predict_latest()
