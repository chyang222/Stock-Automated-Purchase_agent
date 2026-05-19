import yfinance as yf
from pathlib import Path
import pandas as pd


BASE_DIR = Path("/home/dev/project/stock")
SAVE_DIR = BASE_DIR / "data" / "minute"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def get_hynix_1m_3d():
    ticker = "000660.KS"

    df = yf.download(
        ticker,
        period="3d",
        interval="1m",
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        raise RuntimeError("데이터가 없습니다.")

    df = df.reset_index()

    # 컬럼 정리
    df.columns = [
        str(col[0]).lower() if isinstance(col, tuple) else str(col).lower()
        for col in df.columns
    ]

    df = df.rename(
        columns={
            "datetime": "datetime",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )

    # 한국 시간 변환
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"])

    # UTC -> KST 변환
    if df["datetime"].dt.tz is not None:
        df["datetime"] = (
            df["datetime"]
            .dt.tz_convert("Asia/Seoul")
            .dt.tz_localize(None)
    )


    df["stock_code"] = "000660"
    df["stock_name"] = "SK_hynix"

    keep_cols = [
        "datetime",
        "stock_code",
        "stock_name",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    df = df[[c for c in keep_cols if c in df.columns]]

    path = SAVE_DIR / "000660_SK_hynix_3d_1m.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print("SAVE PATH:", path)
    print("ROWS:", len(df))
    print(df.tail())

    return df


if __name__ == "__main__":
    get_hynix_1m_3d()