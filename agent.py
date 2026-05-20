"""Claude tool-use 기반 주식 자동매매 에이전트

흐름:
  1. Claude에게 감시 종목 목록과 함께 분석을 요청
  2. Claude가 도구를 호출하며 (현재가, 지표, ML예측, 잔고, 보유종목) 스스로 수집
  3. Claude가 판단하여 place_order 도구를 호출 → 실제 주문 실행
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import anthropic

from config.settings import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    WATCH_LIST,
    TRADE_QUANTITY,
    DRY_RUN,
    MINUTE_DIR,
    ARTIFACT_DIR,
)
from collector.kis_client import KisClient

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
_kis: KisClient | None = None


def _get_kis() -> KisClient:
    global _kis
    if _kis is None:
        _kis = KisClient()
    return _kis

# ------------------------------------------------------------------ 도구 구현

def _calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2) if not rsi.empty else float("nan")


def _load_minute_df(stock_code: str) -> pd.DataFrame | None:
    """저장된 분봉 CSV 로드 (없으면 None)"""
    matches = list(MINUTE_DIR.glob(f"{stock_code}_*.csv"))
    if not matches:
        return None
    df = pd.read_csv(matches[0])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df.sort_values("datetime").reset_index(drop=True)


def tool_get_current_price(stock_code: str) -> dict:
    try:
        return _get_kis().get_current_price(stock_code)
    except Exception as e:
        return {"error": str(e), "stock_code": stock_code}


def tool_get_technical_indicators(stock_code: str) -> dict:
    try:
        df = _load_minute_df(stock_code)
        if df is None or len(df) < 20:
            # 로컬 파일 없으면 FinanceDataReader 일봉으로 폴백
            try:
                import FinanceDataReader as fdr
                from datetime import timedelta
                start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
                fdr_df = fdr.DataReader(stock_code, start, datetime.now().strftime("%Y-%m-%d"))
                if not fdr_df.empty:
                    fdr_df = fdr_df.reset_index().rename(columns={
                        "Date": "date", "Open": "open", "High": "high",
                        "Low": "low", "Close": "close", "Volume": "volume",
                    })
                    df = fdr_df
            except Exception:
                pass
        if df is None or len(df) < 5:
            return {"error": "데이터 부족", "stock_code": stock_code}

        close = df["close"].astype(float)
        result = {
            "stock_code": stock_code,
            "data_points": len(df),
            "latest_close": float(close.iloc[-1]),
            "ma5": round(float(close.rolling(5).mean().iloc[-1]), 2),
            "ma20": round(float(close.rolling(20).mean().iloc[-1]), 2) if len(close) >= 20 else None,
            "ma60": round(float(close.rolling(60).mean().iloc[-1]), 2) if len(close) >= 60 else None,
            "rsi14": _calc_rsi(close),
            "volatility": round(float(close.pct_change().rolling(20).std().iloc[-1] * 100), 4)
            if len(close) >= 20
            else None,
            "return_1h": round(float((close.iloc[-1] / close.iloc[-60] - 1) * 100), 4)
            if len(close) >= 60
            else None,
        }
        return result
    except Exception as e:
        return {"error": str(e), "stock_code": stock_code}


def tool_get_ml_prediction(stock_code: str) -> dict:
    try:
        import joblib

        model_path = ARTIFACT_DIR / "stock_updown_random_forest.joblib"
        if not model_path.exists():
            return {
                "stock_code": stock_code,
                "up_probability": None,
                "note": "ML 모델 미학습 — 먼저 python app.py train 실행 필요",
            }

        df = _load_minute_df(stock_code)
        if df is None or len(df) < 60:
            # 로컬 파일 없으면 FinanceDataReader로 일봉 즉시 수집
            try:
                import FinanceDataReader as fdr
                from datetime import timedelta
                start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
                end = datetime.now().strftime("%Y-%m-%d")
                fdr_df = fdr.DataReader(stock_code, start, end)
                if fdr_df.empty or len(fdr_df) < 60:
                    return {"stock_code": stock_code, "up_probability": None, "note": f"데이터 부족 ({len(fdr_df) if not fdr_df.empty else 0}일)"}
                fdr_df = fdr_df.reset_index().rename(columns={
                    "Date": "date", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume",
                })
                df = fdr_df
            except Exception as fe:
                return {"stock_code": stock_code, "up_probability": None, "note": f"데이터 수집 실패: {fe}"}

        close = df["close"].astype(float)
        volume = df.get("volume", pd.Series(dtype=float)).astype(float)

        feat = {
            "open": float(df["open"].iloc[-1]),
            "high": float(df["high"].iloc[-1]),
            "low": float(df["low"].iloc[-1]),
            "close": float(close.iloc[-1]),
            "volume": float(volume.iloc[-1]) if not volume.empty else 0,
            "return_1d": float(close.pct_change().iloc[-1]),
            "return_5d": float(close.pct_change(5).iloc[-1]),
            "ma5": float(close.rolling(5).mean().iloc[-1]),
            "ma20": float(close.rolling(20).mean().iloc[-1]),
            "ma60": float(close.rolling(60).mean().iloc[-1]),
            "volatility_20": float(close.pct_change().rolling(20).std().iloc[-1]),
            "rsi14": _calc_rsi(close),
        }
        X = pd.DataFrame([feat])
        model = joblib.load(model_path)
        prob = float(model.predict_proba(X)[:, 1][0])
        return {
            "stock_code": stock_code,
            "up_probability": round(prob, 4),
            "signal": "UP" if prob > 0.55 else "DOWN" if prob < 0.45 else "NEUTRAL",
        }
    except Exception as e:
        return {"error": str(e), "stock_code": stock_code}


def tool_search_news_sentiment(stock_code: str, stock_name: str = "") -> dict:
    """Google News RSS로 종목 관련 최신 한국어 뉴스 헤드라인 수집"""
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET

    query = urllib.parse.quote(f"{stock_name or stock_code} 주식")
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (compatible; StockBot/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            root = ET.fromstring(resp.read())

        headlines = []
        for item in root.findall(".//item")[:10]:
            title_el = item.find("title")
            date_el = item.find("pubDate")
            if title_el is not None and title_el.text:
                headlines.append({
                    "title": title_el.text,
                    "date": date_el.text if date_el is not None else "",
                })

        return {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "headlines": headlines,
            "count": len(headlines),
            "note": "헤드라인을 읽고 긍정/부정/중립 감성을 직접 판단하세요.",
        }
    except Exception as e:
        return {"error": str(e), "stock_code": stock_code}


def tool_get_account_balance() -> dict:
    try:
        return _get_kis().get_account_balance()
    except Exception as e:
        return {"error": str(e)}


def tool_get_holdings() -> list:
    try:
        return _get_kis().get_holdings()
    except Exception as e:
        return [{"error": str(e)}]


def tool_place_order(
    stock_code: str,
    action: str,
    quantity: int,
    price: float,
) -> dict:
    try:
        result = _get_kis().place_order(
            stock_code=stock_code,
            action=action,
            quantity=quantity,
            price=int(price),
            dry_run=DRY_RUN,
        )
        _log_order(result)
        return result
    except Exception as e:
        return {"error": str(e), "stock_code": stock_code}


def _log_order(order: dict):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACT_DIR / "orders.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(order, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ 도구 정의 (Claude에게 전달)

TOOLS = [
    {
        "name": "get_current_price",
        "description": "KIS API를 통해 특정 종목의 현재가, 거래량, 전일 대비 등락률을 조회합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code": {
                    "type": "string",
                    "description": "종목 코드 (예: 005930은 삼성전자)",
                }
            },
            "required": ["stock_code"],
        },
    },
    {
        "name": "get_technical_indicators",
        "description": (
            "분봉 데이터로 기술적 지표를 계산합니다. "
            "MA5/MA20/MA60(이동평균), RSI14(과매수/과매도), 변동성, 수익률 등이 포함됩니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string", "description": "종목 코드"}
            },
            "required": ["stock_code"],
        },
    },
    {
        "name": "get_ml_prediction",
        "description": (
            "RandomForest 머신러닝 모델의 상승 확률 예측값을 조회합니다. "
            "0.55 이상이면 상승 신호(UP), 0.45 이하면 하락 신호(DOWN)입니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string", "description": "종목 코드"}
            },
            "required": ["stock_code"],
        },
    },
    {
        "name": "get_account_balance",
        "description": (
            "계좌 잔고를 조회합니다. 총 평가금액, 투자 가능 현금, 주식 평가금액 등을 반환합니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_holdings",
        "description": "현재 보유 중인 종목 목록과 평균 매수가, 현재가, 손익 정보를 조회합니다.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "search_news_sentiment",
        "description": (
            "종목 관련 최신 뉴스 헤드라인을 수집합니다. "
            "반환된 헤드라인을 읽고 시장 감성(긍정/부정/중립)을 직접 판단하여 "
            "매수/매도 의사결정에 반영하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string", "description": "종목 코드"},
                "stock_name": {"type": "string", "description": "종목명 (검색 정확도 향상, 선택)"},
            },
            "required": ["stock_code"],
        },
    },
    {
        "name": "place_order",
        "description": (
            "매수(BUY) 또는 매도(SELL) 주문을 실행합니다. "
            "반드시 사전에 계좌 잔고와 현재가를 확인한 뒤 호출하세요. "
            "price=0이면 시장가 주문입니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string", "description": "종목 코드"},
                "action": {
                    "type": "string",
                    "enum": ["BUY", "SELL"],
                    "description": "매수(BUY) 또는 매도(SELL)",
                },
                "quantity": {"type": "integer", "description": "주문 수량 (양의 정수)"},
                "price": {
                    "type": "number",
                    "description": "주문 가격. 0이면 시장가, 양수면 지정가.",
                },
            },
            "required": ["stock_code", "action", "quantity", "price"],
        },
    },
]

# ------------------------------------------------------------------ 도구 실행 디스패처

TOOL_DISPATCH: dict[str, Any] = {
    "get_current_price": tool_get_current_price,
    "get_technical_indicators": tool_get_technical_indicators,
    "get_ml_prediction": tool_get_ml_prediction,
    "search_news_sentiment": tool_search_news_sentiment,
    "get_account_balance": tool_get_account_balance,
    "get_holdings": tool_get_holdings,
    "place_order": tool_place_order,
}


def _execute_tool(name: str, tool_input: dict) -> str:
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"알 수 없는 도구: {name}"}, ensure_ascii=False)
    try:
        result = fn(**tool_input)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"error": traceback.format_exc()}, ensure_ascii=False)


# ------------------------------------------------------------------ 에이전트 실행

def _build_system_prompt(budget: int | None) -> str:
    if budget is not None:
        budget_section = f"""## 투자 예산
- 이번 실행의 총 투자 가능 금액: **{budget:,}원**
- 매수 수량 = budget ÷ 현재가 (소수점 버림), 단 {TRADE_QUANTITY}주를 초과할 수 없음
- 예산 초과 매수는 절대 금지 (get_account_balance 조회 불필요)
- 보유 종목 매도는 예산과 무관하게 조건 충족 시 실행"""
        balance_step = "1. get_holdings — 보유 종목 파악 (매도 대상 점검)"
    else:
        budget_section = "## 투자 예산\n- get_account_balance로 투자 가능 현금 확인 후 그 범위 내에서 매수"
        balance_step = "1. get_account_balance — 투자 가능 금액 파악\n2. get_holdings — 보유 종목 파악"

    return f"""당신은 한국 주식 자동매매 AI 에이전트입니다.
주어진 도구로 실시간 데이터를 수집하고 분석하여 매수/매도/보유 결정을 내린 뒤 주문을 실행합니다.

{budget_section}

## 분석 순서 (반드시 이 순서로)
{balance_step}
2. 보유 종목 매도 점검 (get_holdings로 가져온 각 종목):
   a. get_current_price → get_technical_indicators → get_ml_prediction
   b. search_news_sentiment → 헤드라인 읽고 감성 판단
   c. 매도 조건 충족 시 즉시 place_order(action="SELL")
3. 신규 매수 후보 분석 (종목 목록의 나머지):
   a. get_current_price → get_technical_indicators → get_ml_prediction
   b. search_news_sentiment → 헤드라인 읽고 감성 판단
   c. 매수 조건 충족 시 place_order(action="BUY")

## 감성 분석 반영 규칙
- **긍정(positive)**: 다른 매수 조건 충족 시 적극 매수. 보유 중이면 유지.
- **부정(negative)**: ML이 0.48 이상이어도 신규 매수 보류. 보유 중이면 손실률 무관하게 매도 우선 검토.
- **중립(neutral)**: ML/RSI 기준만으로 판단.
- 뉴스가 없거나 수집 실패 시 중립으로 처리하고 ML/RSI 기준 적용.

## 매도 조건 — 보유 종목에서 하나라도 해당하면 즉시 매도
- ML 상승 확률 < 0.45 (하락 신호)
- RSI14 > 75 (과매수, 고점 가능성)
- 손실률 ≤ -7% (손절)
- 수익률 ≥ +8% (익절)

## 매수 조건 — 아래 조건을 최대한 충족하는 종목 매수 (테스트 기간: 조건 완화)
- ML 상승 확률 ≥ 0.48 (NEUTRAL 포함 허용)
- RSI14 < 75 (극과매수만 제외)
- 예산(또는 잔고) ≥ 현재가 × 주문수량
- 위 조건을 모두 만족하는 종목 중 ML 확률이 가장 높은 종목 우선 매수

## 주의사항
- 종목당 최대 {TRADE_QUANTITY}주 (예산 내 최적 수량 자동 계산)
- 조건 충족 종목이 있으면 적극적으로 매수 실행 (테스트 기간이므로 소극적 보류 지양)
- DRY_RUN={DRY_RUN}: {'시뮬레이션만 실행, 실제 주문 없음' if DRY_RUN else '실제 주문 실행됩니다!'}
- 분석 근거와 최종 판단을 명확히 설명하세요
"""


def run(watch_list: list[str] | None = None, budget: int | None = None) -> list[dict]:
    """에이전트 실행

    Args:
        watch_list: 분석할 종목 코드 목록. None이면 settings의 WATCH_LIST 사용.
        budget: 이번 실행에 사용할 예산(원). None이면 계좌 잔고 기준.

    Returns:
        실행된 주문 목록
    """
    symbols = watch_list or WATCH_LIST
    print(f"\n{'='*60}")
    print(f"  주식 자동매매 에이전트 시작  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  모드: {'DRY-RUN (시뮬레이션)' if DRY_RUN else '실제 주문'}")
    print(f"  감시 종목: {symbols}")
    if budget is not None:
        print(f"  예산: {budget:,}원")
    print(f"{'='*60}\n")

    budget_desc = f"{budget:,}원 예산 내에서" if budget is not None else "계좌 잔고 범위 내에서"
    user_message = (
        f"다음 종목들을 분석하고 매매 결정을 내려주세요: {', '.join(symbols)}\n"
        f"각 종목의 현재가, 기술적 지표, ML 예측을 확인한 뒤 "
        f"{budget_desc} 최적의 매매 결정을 실행하세요."
    )

    messages = [{"role": "user", "content": user_message}]
    results = []

    max_iterations = 30
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = _client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=8192,
            system=_build_system_prompt(budget),
            tools=TOOLS,
            messages=messages,
            cache_control={"type": "ephemeral"},
        )

        # assistant 응답을 히스토리에 추가
        messages.append({"role": "assistant", "content": response.content})

        # 텍스트 출력
        for block in response.content:
            if block.type == "text" and block.text:
                print(block.text)

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                print(f"\n[도구 호출] {block.name}({json.dumps(block.input, ensure_ascii=False)})")

                result_str = _execute_tool(block.name, block.input)
                result_obj = json.loads(result_str)

                print(f"[도구 결과] {result_str[:300]}{'...' if len(result_str) > 300 else ''}")

                # place_order 결과 수집
                if block.name == "place_order" and "error" not in result_obj:
                    results.append(result_obj)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    }
                )

            messages.append({"role": "user", "content": tool_results})
        else:
            # 예상치 못한 stop_reason
            print(f"[경고] 예상치 못한 stop_reason: {response.stop_reason}")
            break

    if iteration >= max_iterations:
        print("[경고] 최대 반복 횟수에 도달했습니다.")

    print(f"\n{'='*60}")
    print(f"  에이전트 종료 — 주문 실행: {len(results)}건")
    for r in results:
        print(f"  {r.get('action')} {r.get('stock_code')} {r.get('quantity')}주 @ {r.get('price')}")
    print(f"{'='*60}\n")

    return results


# ------------------------------------------------------------------ 추천 분석

def recommend_with_claude(candidates_df) -> None:
    """스크리닝된 후보 종목을 Claude에게 전달해 최종 추천 의견 받기"""
    import pandas as pd

    display_cols = [
        "stock_code", "name", "close", "marcap_억",
        "ma5", "ma20", "rsi14", "return_5d", "return_20d",
        "vol_ratio", "volatility", "trend", "up_probability",
    ]
    cols = [c for c in display_cols if c in candidates_df.columns]
    summary = candidates_df[cols].to_string(index=False)

    print(summary)
    print()

    prompt = f"""다음은 오늘 기준 KOSPI 전체 종목을 스크리닝한 결과입니다.
시총 상위 80개 종목 중 가격·거래량·RSI 필터를 통과한 후보를 ML 상승확률 기준으로 정렬했습니다.

[스크리닝 결과]
{summary}

컬럼 설명:
- close: 현재가(원)  - marcap_억: 시가총액(억원)
- ma5/ma20: 5일/20일 이동평균  - rsi14: RSI (30 이하=과매도, 70 이상=과매수)
- return_5d/return_20d: 5일/20일 수익률(%)  - vol_ratio: 거래량/20일평균 비율
- trend: UP=현재가>MA5>MA20  - up_probability: ML 상승확률 (0~1)

위 데이터를 바탕으로:
1. **지금 당장 매수 고려할 만한 TOP 5** 종목을 선정하고 이유를 설명해주세요
2. 각 종목의 **투자 포인트**와 **주의할 리스크**를 간결하게 정리해주세요
3. 전반적인 **시장 분위기**도 데이터 기반으로 간단히 코멘트해주세요

단, 투자는 본인 책임임을 전제로, 근거 있는 분석을 제공해주세요."""

    print(f"{'='*60}")
    print("  Claude 추천 분석 중...")
    print(f"{'='*60}\n")

    response = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.content[0].text)


# ------------------------------------------------------------------ 자동매매 루프

_MARKET_OPEN = (9, 0)    # 09:00
_MARKET_CLOSE = (15, 20) # 15:20 (동시호가 직전 마감)


def _is_market_open(now: datetime | None = None) -> bool:
    t = now or datetime.now()
    if t.weekday() >= 5:  # 토·일
        return False
    h, m = t.hour, t.minute
    return (_MARKET_OPEN[0], _MARKET_OPEN[1]) <= (h, m) <= (_MARKET_CLOSE[0], _MARKET_CLOSE[1])


def _seconds_until_open(now: datetime | None = None) -> int:
    t = now or datetime.now()
    # 오늘 09:00
    target = t.replace(hour=9, minute=0, second=0, microsecond=0)
    if t >= target:
        # 이미 지났으면 내일 09:00
        target += timedelta(days=1)
    # 주말 건너뛰기
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return int((target - t).total_seconds())


def trading_loop(
    budget: int | None = None,
    interval_min: int = 30,
    screen_every_n: int = 4,
) -> None:
    """장중 자동매매 루프

    Args:
        budget: 회당 사용 예산(원). None이면 계좌 잔고 전액 기준.
        interval_min: 에이전트 실행 간격(분).
        screen_every_n: N회마다 KOSPI 재스크리닝.
    """
    from collector.recommender import get_recommendations

    print(f"\n{'='*60}")
    print("  자동매매 루프 시작 (Ctrl+C 로 종료)")
    print(f"  실행 간격: {interval_min}분 / KOSPI 재스크리닝: {screen_every_n}회마다")
    if budget:
        print(f"  예산: {budget:,}원")
    print(f"{'='*60}\n")

    watch_list: list[str] = []
    iteration = 0

    try:
        while True:
            now = datetime.now()

            if not _is_market_open(now):
                secs = _seconds_until_open(now)
                h, m = divmod(secs // 60, 60)
                print(f"[{now:%H:%M}] 장 마감 또는 휴장. {h}시간 {m}분 후 재개...")
                time.sleep(min(secs, 1800))  # 최대 30분씩 대기
                continue

            iteration += 1

            # KOSPI 스크리닝 (처음 or N회마다)
            if iteration == 1 or iteration % screen_every_n == 1:
                print(f"\n[{now:%H:%M}] KOSPI 스크리닝 중 (약 2~3분)...")
                try:
                    df = get_recommendations(top_n=10, prescreen_n=60, days=60)
                    if not df.empty:
                        watch_list = df["stock_code"].tolist()
                        names = df.set_index("stock_code")["name"].to_dict()
                        print(f"  후보 {len(watch_list)}개: " +
                              ", ".join(f"{c}({names.get(c,'')})" for c in watch_list))

                        # 로컬 데이터 없는 신규 종목 자동 수집
                        from collector.data_fetcher import save_minute_data
                        new_codes = [c for c in watch_list if not list(MINUTE_DIR.glob(f"{c}_*.csv"))]
                        if new_codes:
                            print(f"  신규 종목 데이터 수집: {new_codes}")
                            for code in new_codes:
                                try:
                                    save_minute_data(code, days=5)
                                except Exception as ce:
                                    print(f"    {code} 수집 실패: {ce}")
                except Exception as e:
                    print(f"  스크리닝 실패: {e}")
                    if not watch_list:
                        print("  후보 없음 — 다음 사이클에 재시도")
                        time.sleep(interval_min * 60)
                        continue

            if not watch_list:
                print(f"[{now:%H:%M}] 후보 종목 없음, 대기 중...")
                time.sleep(interval_min * 60)
                continue

            # 보유 종목을 watch_list에 합산 (매도 누락 방지)
            try:
                holdings = _get_kis().get_holdings()
                held_codes = [h["stock_code"] for h in holdings if "stock_code" in h]
                if held_codes:
                    added = [c for c in held_codes if c not in watch_list]
                    if added:
                        print(f"  보유 종목 추가 (매도 점검): {added}")
                    watch_list = list(dict.fromkeys(held_codes + watch_list))  # 보유 종목 우선
            except Exception as e:
                print(f"  보유 종목 조회 실패: {e}")

            # 에이전트 실행
            print(f"\n[{now:%H:%M}] 에이전트 실행 #{iteration} — {watch_list}")
            try:
                run(watch_list=watch_list, budget=budget)
            except Exception as e:
                print(f"  에이전트 오류: {e}")

            # 다음 실행까지 대기
            next_run = now + timedelta(minutes=interval_min)
            if not _is_market_open(next_run):
                secs = _seconds_until_open(next_run)
                h, m = divmod(secs // 60, 60)
                print(f"\n오늘 장 종료. 내일 개장까지 {h}시간 {m}분 대기...")
                time.sleep(secs)
            else:
                print(f"\n다음 실행: {next_run:%H:%M} ({interval_min}분 후)")
                time.sleep(interval_min * 60)

    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("  루프 종료 (사용자 중단)")
        print(f"{'='*60}\n")
