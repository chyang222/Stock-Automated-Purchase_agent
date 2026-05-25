import time
from pathlib import Path

import requests
import pandas as pd
import yfinance as yf


# =========================
# KIS API 직접 입력
# =========================
KIS_APP_KEY = "PSuNrHePzyDhZLzJTGH6KtNXu52HP3ufASD9"
KIS_APP_SECRET = "hwJDSzHukqYAjuf+PnqE73bnxuYBWv+l1zS4ITrx/butdYru0Tt2VudQrFXwO5UgvGtcDaEQa5nuJDplE0LuUxCnKKtLHvbLhm5t1eGLigB6AhYlH5fIcsOU9ByzAHEfoR+LhTSQ5TD9U5elx9TDxJapgV55YqNPk29o9j6G6PDT1hhUTcg="
KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"


# =========================
# 저장 경로
# =========================
BASE_DIR = Path("/home/dev/project/stock")
SAVE_DIR = BASE_DIR / "data" / "minute"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


STOCK_CODE = "000660"
STOCK_NAME = "SK_hynix"
YF_TICKER = "000660.KS"


def get_kis_token():
    url = f"{KIS_BASE_URL}/oauth2/tokenP"

    payload = {
        "grant_type": "client_credentials",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
    }

    res = requests.post(url, json=payload, timeout=20)
    res.raise_for_status()

    data = res.json()
    return data["access_token"]


def get_kis_headers(token: str):
    return {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": "FHKST03010200",
        "custtype": "P",
    }


def get_yfinance_3d_1m() -> pd.DataFrame:
    df = yf.download(
        YF_TICKER,
        period="3d",
        interval="1m",
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        raise RuntimeError("yfinance 데이터 없음")

    df = df.reset_index()

    df.columns = [
        col[0].lower() if isinstance(col, tuple) else str(col).lower()
        for col in df.columns
    ]

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"])
    else:
        raise RuntimeError(f"datetime 컬럼 없음: {df.columns}")

    # KST 변환
    if df["datetime"].dt.tz is not None:
        df["datetime"] = (
            df["datetime"]
            .dt.tz_convert("Asia/Seoul")
            .dt.tz_localize(None)
        )

    df["stock_code"] = STOCK_CODE
    df["stock_name"] = STOCK_NAME
    df["source"] = "yfinance"

    df = df[
        [
            "datetime",
            "stock_code",
            "stock_name",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source",
        ]
    ]

    return df


def get_kis_latest_1m() -> pd.DataFrame:
    token = get_kis_token()

    now = pd.Timestamp.now()
    input_hour = now.strftime("%H%M%S")

    url = (
        f"{KIS_BASE_URL}"
        "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
    )

    params = {
        "FID_ETC_CLS_CODE": "",
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": STOCK_CODE,
        "FID_INPUT_HOUR_1": input_hour,
        "FID_PW_DATA_INCU_YN": "Y",
    }

    res = requests.get(
        url,
        headers=get_kis_headers(token),
        params=params,
        timeout=20,
    )
    res.raise_for_status()

    data = res.json()

    print("KIS rt_cd:", data.get("rt_cd"))
    print("KIS msg:", data.get("msg1"))
    print("KIS rows:", len(data.get("output2", [])))

    if data.get("rt_cd") != "0":
        raise RuntimeError(data)

    rows = data.get("output2", [])
    if not rows:
        raise RuntimeError("KIS 분봉 데이터 없음")

    df = pd.DataFrame(rows)

    df = df.rename(
        columns={
            "stck_bsop_date": "date",
            "stck_cntg_hour": "time",
            "stck_oprc": "open",
            "stck_hgpr": "high",
            "stck_lwpr": "low",
            "stck_prpr": "close",
            "cntg_vol": "volume",
        }
    )

    df = df[["date", "time", "open", "high", "low", "close", "volume"]]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + df["time"].astype(str),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )

    df = df[df["datetime"] <= pd.Timestamp.now()]

    df["stock_code"] = STOCK_CODE
    df["stock_name"] = STOCK_NAME
    df["source"] = "kis"

    df = df[
        [
            "datetime",
            "stock_code",
            "stock_name",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source",
        ]
    ]

    return df


def build_realtime_dataset():
    yf_df = get_yfinance_3d_1m()
    kis_df = get_kis_latest_1m()

    df = pd.concat([yf_df, kis_df], ignore_index=True)

    # 같은 시간 데이터가 있으면 KIS 우선
    df["source_rank"] = df["source"].map({"yfinance": 1, "kis": 2})

    df = (
        df.sort_values(["datetime", "source_rank"])
        .drop_duplicates(subset=["datetime", "stock_code"], keep="last")
        .drop(columns=["source_rank"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    path = SAVE_DIR / "000660_SK_hynix_merged_3d_1m.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print("SAVE PATH:", path)
    print("ROWS:", len(df))
    print("LAST TIME:", df["datetime"].max())
    print(df.tail())

    return df


if __name__ == "__main__":
    build_realtime_dataset()