"""KOSPI 종목 스크리닝 및 추천 후보 도출

흐름:
  1. FinanceDataReader로 KOSPI 전체 종목 리스트 + 오늘 스냅샷 취득
  2. 가격/거래량/시총 기준 1차 필터링
  3. 상위 N개 종목의 일봉 이력 수집
  4. 기술적 지표(MA, RSI, 변동성) 계산
  5. ML 모델 상승 확률 스코어링
  6. 최종 후보 반환
"""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

from config.settings import ARTIFACT_DIR
from features.make_features import calc_rsi


# ---------------------------------------------------------- 1. 종목 리스트

def get_kospi_snapshot(
    min_price: int = 2_000,
    max_price: int = 500_000,
    min_volume: int = 200_000,
    min_marcap: int = 100_000_000_000,  # 시총 1천억 이상
) -> pd.DataFrame:
    """KOSPI 전체 스냅샷에서 1차 필터링한 종목 반환"""
    df = fdr.StockListing("KOSPI")

    df = df.rename(columns={
        "Code": "stock_code",
        "Name": "name",
        "Close": "close",
        "Volume": "volume",
        "Marcap": "marcap",
    })

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["marcap"] = pd.to_numeric(df["marcap"], errors="coerce")

    mask = (
        df["close"].between(min_price, max_price)
        & (df["volume"] >= min_volume)
        & (df["marcap"] >= min_marcap)
    )
    return df[mask][["stock_code", "name", "close", "volume", "marcap"]].reset_index(drop=True)


# ---------------------------------------------------------- 2. 이력 수집

def fetch_history(stock_code: str, days: int = 60) -> pd.DataFrame:
    """FinanceDataReader로 일봉 이력 수집"""
    start = (datetime.now() - timedelta(days=days + 20)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    df = fdr.DataReader(stock_code, start, end)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["stock_code"] = stock_code
    return df.tail(days).reset_index(drop=True)


# ---------------------------------------------------------- 3. 기술적 지표

def calc_indicators(df: pd.DataFrame) -> dict | None:
    """일봉 DataFrame에서 기술적 지표 계산"""
    if len(df) < 20:
        return None

    close = df["close"].astype(float)
    vol = df["volume"].astype(float)

    ma5 = float(close.rolling(5).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else None
    rsi = float(calc_rsi(close).iloc[-1])
    vol_ma20 = float(vol.rolling(20).mean().iloc[-1])
    vol_ratio = float(vol.iloc[-1] / vol_ma20) if vol_ma20 > 0 else 1.0
    ret_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0.0
    ret_20d = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else 0.0
    volatility = float(close.pct_change().rolling(20).std().iloc[-1] * 100)

    return {
        "ma5": round(ma5, 0),
        "ma20": round(ma20, 0),
        "ma60": round(ma60, 0) if ma60 else None,
        "rsi14": round(rsi, 2),
        "vol_ratio": round(vol_ratio, 2),   # 거래량/20일평균 비율
        "return_5d": round(ret_5d, 2),
        "return_20d": round(ret_20d, 2),
        "volatility": round(volatility, 2),
        "trend": "UP" if close.iloc[-1] > ma5 > ma20 else "FLAT",
    }


# ---------------------------------------------------------- 4. ML 스코어링

def score_candidates(candidates: list[dict]) -> list[dict]:
    """ML 모델로 상승 확률 계산 (모델 없으면 None)"""
    model_path = ARTIFACT_DIR / "stock_updown_random_forest.joblib"
    if not model_path.exists():
        for c in candidates:
            c["up_probability"] = None
        return candidates

    import joblib
    model = joblib.load(model_path)
    FEATURES = ["open", "high", "low", "close", "volume",
                "return_1d", "return_5d", "ma5", "ma20", "ma60",
                "volatility_20", "rsi14"]

    for c in candidates:
        df = c.pop("_history", None)
        if df is None or len(df) < 60:
            c["up_probability"] = None
            continue
        close = df["close"].astype(float)
        feat = {
            "open": float(df["open"].iloc[-1]),
            "high": float(df["high"].iloc[-1]),
            "low": float(df["low"].iloc[-1]),
            "close": float(close.iloc[-1]),
            "volume": float(df["volume"].iloc[-1]),
            "return_1d": float(close.pct_change().iloc[-1]),
            "return_5d": float(close.pct_change(5).iloc[-1]),
            "ma5": float(close.rolling(5).mean().iloc[-1]),
            "ma20": float(close.rolling(20).mean().iloc[-1]),
            "ma60": float(close.rolling(60).mean().iloc[-1]),
            "volatility_20": float(close.pct_change().rolling(20).std().iloc[-1]),
            "rsi14": float(calc_rsi(close).iloc[-1]),
        }
        X = pd.DataFrame([feat])
        c["up_probability"] = round(float(model.predict_proba(X)[:, 1][0]), 4)

    return candidates


# ---------------------------------------------------------- 5. 메인 함수

def get_recommendations(
    top_n: int = 15,
    prescreen_n: int = 80,
    days: int = 60,
) -> pd.DataFrame:
    """KOSPI 추천 종목 스크리닝

    Args:
        top_n: 최종 반환 종목 수
        prescreen_n: 기술적 분석 전 1차 필터 종목 수 (시총 기준 상위)
        days: 이력 수집 일수

    Returns:
        상승확률·기술지표 기준 정렬된 DataFrame
    """
    print("  [1/4] KOSPI 종목 스냅샷 수집 중...")
    snapshot = get_kospi_snapshot()

    # 시총 상위 N개만 정밀 분석 (API 호출 횟수 제한)
    top_by_marcap = snapshot.nlargest(prescreen_n, "marcap")
    print(f"  [2/4] 1차 필터 통과: {len(top_by_marcap)}개 종목 → 이력 수집 시작")

    candidates = []
    for i, (_, row) in enumerate(top_by_marcap.iterrows()):
        code = row["stock_code"]
        hist = fetch_history(code, days=days)
        if hist.empty or len(hist) < 20:
            continue

        ind = calc_indicators(hist)
        if ind is None:
            continue

        # 기술적 필터: RSI 과매수 제외, 최소 상승 추세
        if ind["rsi14"] > 75:
            continue

        entry = {
            "stock_code": code,
            "name": row["name"],
            "close": int(row["close"]),
            "marcap_억": int(row["marcap"] / 1e8),
            **ind,
            "_history": hist,  # ML 스코어링용 (출력 전 제거)
        }
        candidates.append(entry)

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(top_by_marcap)} 처리 중...")
        time.sleep(0.15)

    print(f"  [3/4] 기술적 필터 통과: {len(candidates)}개 → ML 스코어링")
    candidates = score_candidates(candidates)

    print(f"  [4/4] 스코어링 완료 → 상위 {top_n}개 선정")
    df = pd.DataFrame(candidates)
    if df.empty:
        return df

    # ML 확률 기준 정렬 (없으면 return_5d 기준)
    sort_col = "up_probability" if df["up_probability"].notna().any() else "return_5d"
    df = df.sort_values(sort_col, ascending=False).head(top_n).reset_index(drop=True)
    return df
