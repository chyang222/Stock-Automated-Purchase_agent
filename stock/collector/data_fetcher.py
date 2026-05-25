"""yfinance 기반 분봉 데이터 수집

- 1분봉: 최대 7일  (MINUTE_DIR)    — 실시간 기술지표용
- 5분봉: 최대 60일 (MINUTE5_DIR)   — PatchTST 학습용
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config.settings import MINUTE_DIR, MINUTE5_DIR


def _to_yf_ticker(stock_code: str) -> str:
    return f"{stock_code.zfill(6)}.KS"


def _normalize_ohlcv(df: pd.DataFrame, stock_code: str, interval_label: str) -> pd.DataFrame:
    """yfinance 응답 → 공통 컬럼 정규화"""
    df = df.reset_index().rename(columns={
        "Datetime": "datetime", "Date": "datetime",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    if "datetime" not in df.columns:
        raise RuntimeError(f"{stock_code} datetime 컬럼 없음")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    df["date"] = df["datetime"].dt.strftime("%Y%m%d")
    df["time"] = df["datetime"].dt.strftime("%H%M%S")
    df["stock_code"] = stock_code.zfill(6)

    keep = ["date", "time", "open", "high", "low", "close", "volume", "datetime", "stock_code"]
    df = df[[c for c in keep if c in df.columns]]
    return (
        df.sort_values("datetime")
        .drop_duplicates(subset=["datetime", "stock_code"])
        .reset_index(drop=True)
    )


def fetch_minute_data(stock_code: str, days: int = 5) -> pd.DataFrame:
    """1분봉 수집 (yfinance, 최대 7일)"""
    ticker = yf.Ticker(_to_yf_ticker(stock_code))
    df = ticker.history(period=f"{min(days, 7)}d", interval="1m")
    if df.empty:
        raise RuntimeError(f"{stock_code} 1분봉 데이터 없음")
    return _normalize_ohlcv(df, stock_code, "1m")


def fetch_5m_data(stock_code: str, days: int = 60) -> pd.DataFrame:
    """5분봉 수집 (yfinance, 최대 60일) — PatchTST 학습용"""
    ticker = yf.Ticker(_to_yf_ticker(stock_code))
    df = ticker.history(period=f"{min(days, 59)}d", interval="5m")
    if df.empty:
        raise RuntimeError(f"{stock_code} 5분봉 데이터 없음")
    return _normalize_ohlcv(df, stock_code, "5m")


def _save_to_csv(
    new_df: pd.DataFrame,
    path: Path,
    code: str,
    keep_days: int,
) -> Path:
    """신규 데이터를 기존 CSV에 병합 후 저장 (공통 로직)"""
    if path.exists():
        old_df = pd.read_csv(path, dtype={"stock_code": str})
        old_df["datetime"] = pd.to_datetime(old_df["datetime"], errors="coerce")
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["stock_code"] = code
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=keep_days + 2)
    df = (
        df[df["datetime"] >= cutoff]
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime", "stock_code"])
        .reset_index(drop=True)
    )
    df.to_csv(path, index=False, encoding="utf-8-sig")
    days_count = df["date"].nunique() if "date" in df.columns else "?"
    print(f"SAVE: {path.name} ({len(df)} rows, {days_count} 거래일)")
    return path


def save_minute_data(
    stock_code: str,
    stock_name: Optional[str] = None,
    days: int = 5,
) -> Path:
    """1분봉 데이터를 CSV로 저장 (기존 파일에 병합)"""
    code = stock_code.zfill(6)
    path = MINUTE_DIR / f"{code}_{stock_name or code}_1m.csv"
    return _save_to_csv(fetch_minute_data(code, days=days), path, code, keep_days=days)


def save_5m_data(
    stock_code: str,
    stock_name: Optional[str] = None,
    days: int = 60,
) -> Path:
    """5분봉 데이터를 CSV로 저장 — PatchTST 학습용"""
    code = stock_code.zfill(6)
    path = MINUTE5_DIR / f"{code}_{stock_name or code}_5m.csv"
    return _save_to_csv(fetch_5m_data(code, days=days), path, code, keep_days=days)
