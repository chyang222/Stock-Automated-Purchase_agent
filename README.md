# 주식 자동매매 에이전트

Claude API(Tool Use) + 한국투자증권(KIS) OpenAPI 기반 KOSPI 자동매매 시스템.
PatchTST 시계열 모델 + 실시간 뉴스 감성 분석을 결합해 매수/매도를 자율 결정합니다.

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        app.py (CLI 진입점)                       │
│  agent │ loop │ collect │ collect-all │ train │ train-patchtst   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     agent.py (핵심 에이전트)                      │
│                                                                  │
│              Claude API — claude-sonnet-4-6                      │
│         Claude가 아래 7개 도구를 자율 호출하며 매매 판단           │
│                                                                  │
│  ┌──────────────────┐  ┌─────────────────────────────────────┐  │
│  │ get_current_price│  │ get_technical_indicators            │  │
│  │ KIS 실시간 시세  │  │ MA5/20/60, RSI14, 변동성            │  │
│  └──────────────────┘  └─────────────────────────────────────┘  │
│  ┌──────────────────┐  ┌─────────────────────────────────────┐  │
│  │get_ml_prediction │  │ search_news_sentiment               │  │
│  │ ① PatchTST 우선  │  │ Google News RSS → 헤드라인 수집     │  │
│  │ ② RandomForest   │  │ Claude가 직접 감성 판단             │  │
│  │ ③ FDR 일봉 즉시  │  │ 긍정 / 부정 / 중립                 │  │
│  └──────────────────┘  └─────────────────────────────────────┘  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │get_account_      │  │ get_holdings     │  │ place_order  │  │
│  │balance           │  │ 보유종목·손익     │  │ KIS 실제주문 │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
          │                         │
          ▼                         ▼
┌──────────────────┐     ┌──────────────────────────────┐
│  collector/      │     │  models/                      │
│  kis_client.py   │     │  patchtst_model.py  ← 메인    │
│  data_fetcher.py │     │  train_model.py     ← fallback│
│  recommender.py  │     └──────────────────────────────┘
│  collect_all.py  │
└──────────────────┘
```

---

## ML 모델 (PatchTST)

```
입력: 5분봉 최근 128봉 × 7채널
      [수익률, 거래량(log), RSI, MA5비율, MA20비율, 변동성, 60봉수익률]
         │
         ▼
    Patch 분할 (16봉씩, stride=8) → 15개 패치
         │
         ▼
    Transformer (2 layer, d_model=64, 4 head, ~69K 파라미터)
         │
         ▼
출력: 60분 후(12봉) 수익률 > 수수료(0.21%) 확률 [0~1]
      0.55 이상 → UP  (매수 신호)
      0.45 이하 → DOWN (매도 신호)
      그 사이   → NEUTRAL

학습 데이터: 코스피 상위 300종목 × 5분봉 60일치
             → 약 70,000 샘플
```

### 모델 선택 우선순위 (자동)

| 순위 | 조건 | 사용 모델 |
|------|------|----------|
| 1 | `patchtst_stock.pt` + 해당 종목 5분봉 데이터 존재 | **PatchTST** |
| 2 | `stock_updown_random_forest.joblib` 존재 | RandomForest |
| 3 | 둘 다 없음 | FDR 일봉 즉시 계산 |

---

## 수수료 반영

```
매수 수수료: 0.015%
매도 수수료: 0.015% + 증권거래세 0.18% = 0.195%
왕복 총 비용: 0.21%

→ 매수 시 잔고 체크: 현재가 × 수량 × 1.00015 이상 필요
→ 익절 기준: 실질 수익률 +5.0% 이상 (수수료 차감 후 +4.79% 순수익)
→ 손절 기준: 실질 손실률 -3% 이하
```

---

## 워크플로우

### `python app.py loop` 동작 구조

```
시작
 │
 ├─ 장 시간 확인 (평일 09:00 ~ 15:20)
 │   └─ 장외 → 다음 개장까지 대기
 │
 ├─ [4회마다] KOSPI 스크리닝
 │   ├─ FinanceDataReader → 시총 상위 60개
 │   ├─ 가격 · 거래량 · RSI 1차 필터
 │   ├─ ML 스코어링 → 상위 10개 선정
 │   └─ 신규 종목 5분봉 자동 수집
 │
 ├─ 보유 종목을 watch_list 앞에 추가 (매도 누락 방지)
 │
 ├─ Claude 에이전트 실행
 │   ├─ [보유 종목] 현재가 → 기술지표 → PatchTST → 뉴스감성 → 매도?
 │   └─ [후보 종목] 현재가 → 기술지표 → PatchTST → 뉴스감성 → 매수?
 │
 └─ 20분 대기 → 반복
```

### 매매 조건

**매수** — 아래 조건 모두 충족 시

| 조건 | 기준 |
|------|------|
| PatchTST 상승확률 | ≥ 0.48 |
| RSI14 | < 75 |
| 잔고 | ≥ 현재가 × 수량 × 1.00015 |
| 뉴스 감성 | 긍정 또는 중립 |

**매도** — 아래 조건 하나라도 해당 시

| 조건 | 기준 |
|------|------|
| PatchTST 상승확률 | < 0.45 |
| RSI14 | > 75 (과매수) |
| 실질 손실률 | ≤ -3% (손절) |
| 실질 수익률 | ≥ +5.0% (익절) |
| 뉴스 부정 + 수익권 | 즉시 익절 |

---

## 초기 설정 순서

```bash
# 1. 패키지 설치
pip install anthropic python-dotenv requests pandas scikit-learn joblib \
            yfinance finance-datareader torch transformers

# 2. 환경변수 설정
cp .env.example .env   # 값 채워넣기

# 3. 코스피 전체 5분봉 수집 (~20분)
python app.py collect-all

# 4. PatchTST 모델 학습 (백그라운드, CPU ~1시간)
nohup python -u app.py train-patchtst > train_patchtst.log 2>&1 &
tail -f train_patchtst.log   # 진행 확인

# 5. 학습 완료 확인
ls -lh data/artifacts/patchtst_stock.pt
cat data/artifacts/patchtst_metrics.json

# 6. 모의투자로 먼저 테스트 (.env: KIS_IS_PAPER=true, DRY_RUN=true)
python app.py loop --budget 100000

# 7. 실전 전환 (.env: KIS_IS_PAPER=false, DRY_RUN=false)
python app.py loop --budget 100000
```

---

## 전체 명령어

```bash
# ── 데이터 수집
python app.py collect                          # 1분봉 7일 (WATCH_LIST 종목)
python app.py collect --symbols 005930,000660  # 특정 종목만
python app.py collect-all                      # 코스피 상위 300개 5분봉 60일

# ── 모델 학습
nohup python -u app.py train-patchtst > train_patchtst.log 2>&1 &  # PatchTST (메인)
python app.py features && python app.py train                       # RandomForest (fallback)

# ── 에이전트 실행
python app.py agent                            # 1회 실행
python app.py agent --symbols 005930 --budget 30000
python app.py loop                             # 장중 반복 (20분 간격)
python app.py loop --budget 100000

# ── 추천
python app.py recommend                        # Claude가 TOP 5 추천

# ── 간격 조정
LOOP_INTERVAL_MIN=10 python app.py loop
SCREEN_EVERY_N=2 python app.py loop
COLLECT_TOP_N=100 python app.py collect-all    # 상위 100개만 수집
PATCHTST_EPOCHS=20 python app.py train-patchtst
```

---

## crontab 설정 (매일 자동 실행)

이 서버의 crond는 UTC 기준으로 동작합니다. 시간은 모두 UTC로 입력해야 합니다.

```bash
crontab -e
```

```
# 주식 자동매매 (UTC 기준)
0 0 * * 1-5 /home/dev/project/stock/start.sh >> /home/dev/project/stock/logs/cron.log 2>&1
30 6 * * 1-5 /home/dev/project/stock/stop.sh >> /home/dev/project/stock/logs/cron.log 2>&1
# PatchTST 재학습 (23:00 KST = 14:00 UTC)
0 14 * * 1-5 /home/dev/miniconda3/envs/stock/bin/python /home/dev/project/stock/app.py train-patchtst >> /home/dev/project/stock/logs/train_patchtst.log 2>&1
```

| 목적 | KST | UTC (cron) |
|------|-----|------------|
| 장 시작 | 09:00 | `0 0 * * 1-5` |
| 장 마감 | 15:30 | `30 6 * * 1-5` |
| PatchTST 재학습 | 23:00 | `0 14 * * 1-5` |

```bash
# 로그 확인
tail -f logs/trade_$(date +%Y%m%d).log
```

---

## 디렉토리 구조

```
stock/
├── app.py                        # CLI 진입점
├── agent.py                      # Claude tool-use 에이전트 (핵심)
├── start.sh / stop.sh            # crontab용 시작/종료 스크립트
├── train_patchtst.log            # PatchTST 학습 로그
├── .env                          # API 키 등 환경변수
├── config/
│   └── settings.py               # 환경변수 로드 및 경로 설정
├── collector/
│   ├── kis_client.py             # KIS API (현재가 · 잔고 · 주문)
│   ├── data_fetcher.py           # yfinance 1분봉(7일) / 5분봉(60일) 수집
│   ├── collect_all.py            # 코스피 전체 5분봉 일괄 수집
│   └── recommender.py            # KOSPI 종목 스크리닝
├── features/
│   └── make_features.py          # RandomForest용 피처 생성
├── models/
│   ├── patchtst_model.py         # PatchTST 학습 · 추론 (메인 모델)
│   ├── train_model.py            # RandomForest 학습 (fallback)
│   └── predict.py                # RF 예측
├── data/
│   ├── minute/                   # 1분봉 CSV (최대 7일, RF용)
│   ├── minute5/                  # 5분봉 CSV (최대 60일, PatchTST용)
│   ├── processed/                # RF 피처 CSV
│   └── artifacts/
│       ├── patchtst_stock.pt     # PatchTST 모델 가중치 (best val_acc 자동 저장)
│       ├── patchtst_metrics.json # 학습 성능 지표
│       ├── stock_updown_random_forest.joblib  # RF 모델 (fallback)
│       └── orders.log            # 주문 이력
└── logs/
    ├── trade_YYYYMMDD.log        # 일별 매매 로그
    └── cron.log                  # crontab 실행 로그
```

---

## 환경변수 (.env)

| 변수 | 설명 |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic 콘솔에서 발급 |
| `KIS_APP_KEY` | KIS 개발자 포털 앱키 |
| `KIS_APP_SECRET` | KIS 개발자 포털 앱시크릿 |
| `KIS_ACCOUNT_NO` | 계좌번호 형식: `12345678-01` |
| `KIS_IS_PAPER` | `true` = 모의투자, `false` = 실전 |
| `DRY_RUN` | `true` = 시뮬레이션만, `false` = 실제 주문 |
| `WATCH_LIST` | 감시 종목 코드 (쉼표 구분, 예: `005930,000660`) |
| `TRADE_QUANTITY` | 종목당 최대 주문 수량 |

---

## 주의사항

- 이 프로그램은 투자 참고용이며 수익을 보장하지 않습니다.
- KIS API 키는 모의투자용과 실전투자용이 별도입니다.
- 실전 전 반드시 `KIS_IS_PAPER=true`, `DRY_RUN=true`로 충분히 테스트하세요.
- `DRY_RUN=false`로 변경하면 실제 주문이 체결됩니다.
