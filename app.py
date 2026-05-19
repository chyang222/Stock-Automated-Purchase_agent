"""주식 자동매매 에이전트 진입점

사용법:
  python app.py agent                    # 에이전트 1회 실행
  python app.py loop                     # 장중 자동매매 루프 (계속 반복)
  python app.py loop --budget 30000      # 예산 3만원으로 루프
  python app.py collect                  # 분봉 데이터 수집
  python app.py features                 # 피처 생성
  python app.py train                    # ML 모델 학습
  python app.py recommend                # KOSPI 추천 종목 분석
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="주식 자동매매 에이전트")
    parser.add_argument(
        "command",
        choices=["agent", "loop", "collect", "features", "train", "recommend"],
        help="실행할 명령",
    )
    parser.add_argument("--symbols", type=str, help="콤마 구분 종목 코드 (예: 005930,000660)")
    parser.add_argument("--dry-run", action="store_true", help="시뮬레이션 모드 (실제 주문 없음)")
    parser.add_argument("--budget", type=int, help="이번 실행에 사용할 예산(원). 예: --budget 30000")
    args = parser.parse_args()

    if args.symbols:
        os.environ["WATCH_LIST"] = args.symbols
    if args.dry_run:
        os.environ["DRY_RUN"] = "true"

    if args.command == "agent":
        from agent import run
        from config.settings import WATCH_LIST

        symbols = args.symbols.split(",") if args.symbols else WATCH_LIST
        run(watch_list=symbols, budget=args.budget)

    elif args.command == "collect":
        from collector.data_fetcher import save_minute_data
        from config.settings import WATCH_LIST

        symbols = args.symbols.split(",") if args.symbols else WATCH_LIST
        days = int(os.environ.get("COLLECT_DAYS", "5"))
        for code in symbols:
            print(f"수집 중: {code}")
            save_minute_data(code, days=days)

    elif args.command == "features":
        from features.make_features import build_dataset

        build_dataset()

    elif args.command == "train":
        from models.train_model import train_model

        train_model()

    elif args.command == "loop":
        from agent import trading_loop

        interval = int(os.environ.get("LOOP_INTERVAL_MIN", "30"))
        screen_every = int(os.environ.get("SCREEN_EVERY_N", "4"))
        print(f"자동매매 루프 시작 (간격: {interval}분, KOSPI 재스크리닝: {screen_every}회마다)")
        trading_loop(budget=args.budget, interval_min=interval, screen_every_n=screen_every)

    elif args.command == "recommend":
        from collector.recommender import get_recommendations
        from agent import recommend_with_claude

        top_n = int(os.environ.get("RECOMMEND_TOP_N", "15"))
        print(f"\nKOSPI 추천 스크리닝 시작 (상위 {top_n}개, 약 2~3분 소요)\n")
        df = get_recommendations(top_n=top_n)
        if df.empty:
            print("추천 종목을 찾지 못했습니다.")
        else:
            recommend_with_claude(df)


if __name__ == "__main__":
    main()
