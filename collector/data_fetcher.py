"""yfinance 기반 분봉 데이터 수집 (과거 최대 7일, 1분봉)"""

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config.settings import MINUTE_DIR


def _to_yf_ticker(stock_code: str) -> str:
    return f"{stock_code.zfill(6)}.KS"


def fetch_minute_data(stock_code: str, days: int = 5) -> pd.DataFrame:
    """yfinance로 분봉 데이터 수집 (최대 7일)"""
    ticker = yf.Ticker(_to_yf_ticker(stock_code))
    df = ticker.history(period=f"{min(days, 7)}d", interval="1m")
    if df.empty:
        raise RuntimeError(f"{stock_code} yfinance 분봉 데이터 없음")

    df = df.reset_index().rename(
        columns={
            "Datetime": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    df["date"] = df["datetime"].dt.strftime("%Y%m%d")
    df["time"] = df["datetime"].dt.strftime("%H%M%S")
    df["stock_code"] = stock_code.zfill(6)

    keep = ["date", "time", "open", "high", "low", "close", "volume", "datetime", "stock_code"]
    return (
        df[[c for c in keep if c in df.columns]]
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime", "stock_code"])
        .reset_index(drop=True)
    )


def save_minute_data(
    stock_code: str,
    stock_name: Optional[str] = None,
    days: int = 5,
) -> Path:
    """분봉 데이터를 CSV로 저장 (기존 파일에 병합)"""
    code = stock_code.zfill(6)
    name = stock_name or code
    path = MINUTE_DIR / f"{code}_{name}_1m.csv"

    new_df = fetch_minute_data(code, days=days)

    if path.exists():
        old_df = pd.read_csv(path, dtype={"stock_code": str})
        old_df["datetime"] = pd.to_datetime(old_df["datetime"], errors="coerce")
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["stock_code"] = code

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days + 2)
    df = df[df["datetime"] >= cutoff]
    df = (
        df.sort_values("datetime")
        .drop_duplicates(subset=["datetime", "stock_code"])
        .reset_index(drop=True)
    )
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"SAVE: {path} ({len(df)} rows, {df['date'].nunique()} 거래일)")
    return path
