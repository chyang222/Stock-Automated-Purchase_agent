"""코스피 시총 상위 N개 종목 5분봉 60일치 일괄 수집

사용법:
  python app.py collect-all               # 시총 상위 300개
  python app.py collect-all --top 100    # 상위 100개만
  python app.py collect-all --no-skip    # 기존 파일도 재수집
"""

from __future__ import annotations

import time
from pathlib import Path

from config.settings import MINUTE5_DIR


def collect_all_kospi(
    top_n: int = 300,
    days: int = 60,
    skip_existing: bool = True,
    sleep_sec: float = 0.8,
) -> list[str]:
    """코스피 시총 상위 top_n개 종목 5분봉 수집

    Args:
        top_n: 수집할 종목 수 (시총 순)
        days: 수집 기간 (최대 59)
        skip_existing: 이미 파일이 있으면 건너뜀
        sleep_sec: 종목 간 대기 시간 (API 레이트 리밋 방지)

    Returns:
        실패한 종목 코드 목록
    """
    import FinanceDataReader as fdr
    from collector.data_fetcher import save_5m_data

    print("KOSPI 종목 목록 조회 중...")
    listing = fdr.StockListing("KOSPI")

    # 시총 컬럼 찾기
    marcap_col = next(
        (c for c in listing.columns if c.lower() in ("marcap", "시가총액", "market_cap")),
        None,
    )
    if marcap_col:
        listing = listing.sort_values(marcap_col, ascending=False)

    listing = listing.dropna(subset=["Code"]).head(top_n)
    codes = listing["Code"].astype(str).str.zfill(6).tolist()
    names = (
        listing.set_index(listing["Code"].astype(str).str.zfill(6))["Name"].to_dict()
        if "Name" in listing.columns
        else {}
    )

    total = len(codes)
    print(f"수집 대상: {total}개 종목 ({days}일치 5분봉, 예상 소요: ~{total * (sleep_sec + 2):.0f}초)\n")

    failed: list[str] = []

    for i, code in enumerate(codes, 1):
        name = names.get(code, code)

        if skip_existing and list(MINUTE5_DIR.glob(f"{code}_*.csv")):
            print(f"  [{i:3d}/{total}] {code} ({name}) — 건너뜀 (기존 파일 있음)")
            continue

        print(f"  [{i:3d}/{total}] {code} ({name}) 수집 중...", end=" ", flush=True)
        try:
            save_5m_data(code, stock_name=name, days=days)
            time.sleep(sleep_sec)
        except Exception as e:
            print(f"실패: {e}")
            failed.append(code)
            time.sleep(sleep_sec * 2)

    print(f"\n{'='*50}")
    print(f"완료: {total - len(failed)}개 성공 / {len(failed)}개 실패")
    if failed:
        print(f"실패 목록: {failed}")
    return failed
