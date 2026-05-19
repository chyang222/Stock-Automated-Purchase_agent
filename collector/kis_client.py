import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

from config.settings import (
    KIS_APP_KEY,
    KIS_APP_SECRET,
    KIS_ACCOUNT_CANO,
    KIS_ACCOUNT_PRDT,
    KIS_IS_PAPER,
    KIS_BASE_URL,
    MINUTE_DIR,
)


class KisClient:
    def __init__(self):
        self.access_token, self.token_expires = self._issue_token()

    # ------------------------------------------------------------------ auth

    def _issue_token(self) -> tuple[str, float]:
        url = f"{KIS_BASE_URL}/oauth2/tokenP"
        res = requests.post(
            url,
            json={
                "grant_type": "client_credentials",
                "appkey": KIS_APP_KEY,
                "appsecret": KIS_APP_SECRET,
            },
            timeout=20,
        )
        res.raise_for_status()
        data = res.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"KIS 토큰 발급 실패: {data}")
        expires_at = time.time() + int(data.get("expires_in", 86400)) - 300
        return token, expires_at

    def _ensure_token(self):
        if time.time() > self.token_expires:
            self.access_token, self.token_expires = self._issue_token()

    def _headers(self, tr_id: str) -> dict:
        self._ensure_token()
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": KIS_APP_KEY,
            "appsecret": KIS_APP_SECRET,
            "tr_id": tr_id,
            "custtype": "P",
        }

    def _get(self, path: str, tr_id: str, params: dict) -> dict:
        res = requests.get(
            f"{KIS_BASE_URL}{path}",
            headers=self._headers(tr_id),
            params=params,
            timeout=20,
        )
        res.raise_for_status()
        data = res.json()
        if data.get("rt_cd") not in ("0", None):
            raise RuntimeError(f"KIS API 오류 [{tr_id}]: {data.get('msg1')} / {data}")
        return data

    def _post(self, path: str, tr_id: str, body: dict) -> dict:
        res = requests.post(
            f"{KIS_BASE_URL}{path}",
            headers=self._headers(tr_id),
            json=body,
            timeout=20,
        )
        res.raise_for_status()
        data = res.json()
        if data.get("rt_cd") not in ("0", None):
            raise RuntimeError(f"KIS API 오류 [{tr_id}]: {data.get('msg1')} / {data}")
        return data

    # ------------------------------------------------------ 시세 조회

    def get_current_price(self, stock_code: str) -> dict:
        """현재가 조회"""
        data = self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            "FHKST01010100",
            {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": stock_code},
        )
        o = data.get("output", {})
        return {
            "stock_code": stock_code,
            "price": int(o.get("stck_prpr", 0)),
            "open": int(o.get("stck_oprc", 0)),
            "high": int(o.get("stck_hgpr", 0)),
            "low": int(o.get("stck_lwpr", 0)),
            "volume": int(o.get("acml_vol", 0)),
            "change": int(o.get("prdy_vrss", 0)),
            "change_rate": float(o.get("prdy_ctrt", 0)),
            "market_cap": int(o.get("hts_avls", 0)),
        }

    def get_today_minute_price(self, stock_code: str) -> pd.DataFrame:
        """당일 전체 분봉 조회 (30개씩 반복 호출)"""
        all_rows: list[dict] = []
        seen: set[str] = set()
        input_hour = datetime.now().strftime("%H%M%S")

        for _ in range(30):  # 최대 900분 커버 (30개 × 30회)
            data = self._get(
                "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
                "FHKST03010200",
                {
                    "FID_ETC_CLS_CODE": "",
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": stock_code,
                    "FID_INPUT_HOUR_1": input_hour,
                    "FID_PW_DATA_INCU_YN": "Y",
                },
            )
            rows = data.get("output2", [])
            if not rows:
                break

            new_rows = [r for r in rows if r.get("stck_cntg_hour") not in seen]
            if not new_rows:
                break
            for r in new_rows:
                seen.add(r["stck_cntg_hour"])
            all_rows.extend(new_rows)

            last_time = rows[-1].get("stck_cntg_hour", "")
            if not last_time or last_time <= "090100":
                break
            input_hour = last_time
            time.sleep(0.3)

        if not all_rows:
            raise RuntimeError(f"{stock_code} 분봉 데이터 없음")

        df = pd.DataFrame(all_rows).rename(
            columns={
                "stck_bsop_date": "date",
                "stck_cntg_hour": "time",
                "stck_oprc": "open",
                "stck_hgpr": "high",
                "stck_lwpr": "low",
                "stck_prpr": "close",
                "cntg_vol": "volume",
                "acml_tr_pbmn": "trading_value",
            }
        )
        keep = ["date", "time", "open", "high", "low", "close", "volume", "trading_value"]
        df = df[[c for c in keep if c in df.columns]]
        for col in ["open", "high", "low", "close", "volume", "trading_value"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + df["time"].astype(str),
            format="%Y%m%d%H%M%S",
            errors="coerce",
        )
        df["stock_code"] = stock_code
        df = df[df["datetime"] <= pd.Timestamp.now()]
        return (
            df.sort_values("datetime")
            .drop_duplicates(subset=["datetime", "stock_code"])
            .reset_index(drop=True)
        )

    def get_daily_price(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """일봉 데이터 조회 (YYYYMMDD 형식)"""
        data = self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            "FHKST03010100",
            {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": stock_code,
                "FID_INPUT_DATE_1": start_date,
                "FID_INPUT_DATE_2": end_date,
                "FID_PERIOD_DIV_CODE": "D",
                "FID_ORG_ADJ_PRC": "0",
            },
        )
        rows = data.get("output2", [])
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).rename(
            columns={
                "stck_bsop_date": "date",
                "stck_oprc": "open",
                "stck_hgpr": "high",
                "stck_lwpr": "low",
                "stck_clpr": "close",
                "acml_vol": "volume",
                "acml_tr_pbmn": "trading_value",
            }
        )
        keep = ["date", "open", "high", "low", "close", "volume", "trading_value"]
        df = df[[c for c in keep if c in df.columns]]
        for col in ["open", "high", "low", "close", "volume", "trading_value"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["datetime"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        df["time"] = "153000"
        df["stock_code"] = stock_code
        return (
            df.dropna(subset=["datetime"])
            .sort_values("datetime")
            .reset_index(drop=True)
        )

    @staticmethod
    def _norm_code(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """stock_code 컬럼을 6자리 문자열로 정규화"""
        df = df.copy()
        df["stock_code"] = stock_code
        return df

    def save_ohlcv(
        self,
        stock_code: str,
        stock_name: Optional[str] = None,
        weeks: int = 1,
    ) -> Path:
        """분봉(당일) + 일봉(과거 N주) 합산 저장"""
        name = stock_name or stock_code
        path = MINUTE_DIR / f"{stock_code}_{name}_1m.csv"
        today = pd.Timestamp.now().normalize()
        cutoff = today - pd.Timedelta(weeks=weeks, days=5)

        frames: list[pd.DataFrame] = []

        # 기존 파일 로드 (stock_code 정규화 포함)
        if path.exists():
            old = pd.read_csv(path, dtype={"stock_code": str})
            old["datetime"] = pd.to_datetime(old["datetime"], errors="coerce")
            old = self._norm_code(old, stock_code)
            frames.append(old)

        # 기존 데이터에 없는 과거 날짜 → 일봉으로 보충
        existing_dates: set = set()
        if frames:
            existing_dates = set(frames[0]["datetime"].dt.date.dropna())

        end_dt = today - pd.Timedelta(days=1)  # 어제까지 (당일은 분봉으로)
        end_str = end_dt.strftime("%Y%m%d")
        start_str = cutoff.strftime("%Y%m%d")

        # 보충이 필요한 날짜가 있는지 확인
        need_daily = any(
            d.date() not in existing_dates
            for d in pd.date_range(cutoff, end_dt, freq="B")  # 영업일 기준
        )
        if need_daily:
            print(f"  일봉 수집: {start_str} ~ {end_str}")
            daily = self.get_daily_price(stock_code, start_str, end_str)
            if not daily.empty:
                daily = self._norm_code(daily, stock_code)
                frames.append(daily)
            time.sleep(0.5)

        # 당일 분봉 추가 (장중/장 직후에만 유효)
        try:
            print(f"  분봉 수집 중: {stock_code}")
            minute = self.get_today_minute_price(stock_code)
            minute = self._norm_code(minute, stock_code)
            frames.append(minute)
        except RuntimeError as e:
            print(f"  분봉 수집 실패 (장 마감 후일 수 있음): {e}")

        if not frames:
            raise RuntimeError(f"{stock_code} 수집 데이터 없음")

        df = pd.concat(frames, ignore_index=True)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        df["stock_code"] = stock_code  # 전체 정규화

        df = df[df["datetime"] >= cutoff]
        df = (
            df.sort_values("datetime")
            .drop_duplicates(subset=["datetime", "stock_code"])
            .reset_index(drop=True)
        )
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"SAVE: {path} ({len(df)} rows, {df['datetime'].dt.date.nunique()} 거래일)")
        return path

    # 하위 호환
    def save_minute_price(
        self,
        stock_code: str,
        stock_name: Optional[str] = None,
        keep_business_days: int = 5,
    ) -> Path:
        return self.save_ohlcv(stock_code, stock_name, weeks=1)

    # ------------------------------------------------------ 계좌 조회

    def get_account_balance(self) -> dict:
        """계좌 잔고 및 투자 가능 금액 조회"""
        tr_id = "VTTC8434R" if KIS_IS_PAPER else "TTTC8434R"
        data = self._get(
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id,
            {
                "CANO": KIS_ACCOUNT_CANO,
                "ACNT_PRDT_CD": KIS_ACCOUNT_PRDT,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "01",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            },
        )
        o2 = data.get("output2", [{}])
        summary = o2[0] if o2 else {}
        return {
            "total_eval": int(summary.get("tot_evlu_amt", 0)),
            "cash_balance": int(summary.get("dnca_tot_amt", 0)),
            "investable_cash": int(summary.get("prvs_rcdl_excc_amt", 0)),
            "stock_eval": int(summary.get("scts_evlu_amt", 0)),
            "profit_loss": int(summary.get("evlu_pfls_smtl_amt", 0)),
        }

    def get_holdings(self) -> list[dict]:
        """보유 종목 조회"""
        tr_id = "VTTC8434R" if KIS_IS_PAPER else "TTTC8434R"
        data = self._get(
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id,
            {
                "CANO": KIS_ACCOUNT_CANO,
                "ACNT_PRDT_CD": KIS_ACCOUNT_PRDT,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "01",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            },
        )
        items = data.get("output1", [])
        result = []
        for item in items:
            qty = int(item.get("hldg_qty", 0))
            if qty == 0:
                continue
            result.append(
                {
                    "stock_code": item.get("pdno", ""),
                    "stock_name": item.get("prdt_name", ""),
                    "quantity": qty,
                    "avg_price": int(item.get("pchs_avg_pric", 0)),
                    "current_price": int(item.get("prpr", 0)),
                    "eval_amount": int(item.get("evlu_amt", 0)),
                    "profit_loss": int(item.get("evlu_pfls_amt", 0)),
                    "profit_loss_rate": float(item.get("evlu_pfls_rt", 0)),
                }
            )
        return result

    # ------------------------------------------------------ 주문

    def place_order(
        self,
        stock_code: str,
        action: str,
        quantity: int,
        price: int = 0,
        dry_run: bool = True,
    ) -> dict:
        """매수/매도 주문 실행

        action: "BUY" | "SELL"
        price: 0이면 시장가, 양수면 지정가
        """
        if action.upper() == "BUY":
            tr_id = "VTTC0802U" if KIS_IS_PAPER else "TTTC0802U"
        elif action.upper() == "SELL":
            tr_id = "VTTC0801U" if KIS_IS_PAPER else "TTTC0801U"
        else:
            raise ValueError(f"action은 BUY 또는 SELL이어야 합니다: {action}")

        ord_dvsn = "01" if price == 0 else "00"  # 01: 시장가, 00: 지정가

        order = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stock_code": stock_code,
            "action": action.upper(),
            "quantity": quantity,
            "price": price,
            "ord_dvsn": "시장가" if ord_dvsn == "01" else "지정가",
            "paper_trade": KIS_IS_PAPER,
            "dry_run": dry_run,
        }

        if dry_run:
            print(f"[DRY-RUN] 주문 시뮬레이션: {order}")
            order["status"] = "dry_run"
            return order

        body = {
            "CANO": KIS_ACCOUNT_CANO,
            "ACNT_PRDT_CD": KIS_ACCOUNT_PRDT,
            "PDNO": stock_code,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
        }

        data = self._post(
            "/uapi/domestic-stock/v1/trading/order-cash",
            tr_id,
            body,
        )
        out = data.get("output", {})
        order["order_no"] = out.get("ODNO", "")
        order["order_time"] = out.get("ORD_TMD", "")
        order["status"] = "submitted"
        print(f"[ORDER] 주문 실행: {order}")
        return order
