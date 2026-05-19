import numpy as np
import pandas as pd

from config.settings import MINUTE_DIR, PROCESSED_DIR


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["volatility_20"] = df["return_1d"].rolling(20).std()
    df["rsi14"] = calc_rsi(df["close"])
    # 내일 종가가 오늘 종가보다 오르면 1
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna().reset_index(drop=True)


def build_dataset() -> pd.DataFrame:
    frames = []
    for path in MINUTE_DIR.glob("*.csv"):
        df = pd.read_csv(path)
        if "close" not in df.columns:
            continue
        feat = make_features(df)
        frames.append(feat)

    if not frames:
        raise RuntimeError(f"분봉 CSV 파일이 없습니다: {MINUTE_DIR}")

    dataset = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED_DIR / "features.csv"
    dataset.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"features 저장: {out_path} ({len(dataset)}행)")
    return dataset


if __name__ == "__main__":
    build_dataset()
