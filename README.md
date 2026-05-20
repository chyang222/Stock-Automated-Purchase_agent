# 주식 자동매매 에이전트

Claude API(Tool Use) + 한국투자증권(KIS) OpenAPI 기반 KOSPI 자동매매 시스템.
ML 시계열 예측과 실시간 뉴스 감성 분석을 결합하여 매수/매도를 자율 결정합니다.

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        app.py (CLI 진입점)                       │
│     agent | loop | collect | features | train | recommend        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     agent.py (핵심 에이전트)                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Claude API (Tool Use)                    │   │
│  │                  model: claude-sonnet-4-6                 │   │
│  │                                                          │   │
│  │   Claude가 스스로 판단하며 아래 도구를 자율 호출           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  도구 (Tools)                                                    │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐   │
│  │ get_current_price   │  │ get_technical_indicators        │   │
│  │ KIS API 실시간 시세 │  │ MA5/20/60, RSI14, 변동성        │   │
│  └─────────────────────┘  └─────────────────────────────────┘   │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐   │
│  │ get_ml_prediction   │  │ search_news_sentiment           │   │
│  │ RandomForest 상승   │  │ Google News RSS → 감성 판단     │   │
│  │ 확률 (0~1)          │  │ 긍정 / 부정 / 중립              │   │
│  └─────────────────────┘  └─────────────────────────────────┘   │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐   │
│  │ get_account_balance │  │ get_holdings                    │   │
│  │ 투자 가능 현금 조회  │  │ 보유 종목 · 손익 조회           │   │
│  └─────────────────────┘  └─────────────────────────────────┘   │
│  ┌─────────────────────┐                                         │
│  │ place_order         │                                         │
│  │ KIS API 실제 주문   │                                         │
│  └─────────────────────┘                                         │
└─────────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌──────────────────┐          ┌──────────────────────┐
│  collector/      │          │  models/              │
│  kis_client.py   │          │  train_model.py       │
│  data_fetcher.py │          │  predict.py           │
│  recommender.py  │          └──────────────────────┘
└──────────────────┘
          │
          ▼
┌──────────────────┐
│  features/       │
│  make_features.py│
│  MA, RSI, 수익률 │
└──────────────────┘
```

---

## 워크플로우

### 자동매매 루프 (`python app.py loop`)

```
시작
 │
 ├─ 장 시간 확인 (09:00 ~ 15:20 평일)
 │   └─ 장외 → 다음 개장까지 대기
 │
 ├─ [4회마다] KOSPI 스크리닝 ──────────────────────────────────┐
 │   │                                                         │
 │   ├─ FinanceDataReader → 시총 상위 60개 1차 필터            │
 │   │   (가격 2,000~500,000 / 거래량 ≥ 20만 / 시총 ≥ 1,000억) │
 │   │                                                         │
 │   ├─ 기술적 필터 (MA, RSI, 추세)                            │
 │   │                                                         │
 │   ├─ ML 스코어링 → 상위 10개 선정                           │
 │   │                                                         │
 │   └─ 신규 종목 분봉 데이터 자동 수집 (yfinance 5일치)        │
 │                                                             │
 ├─ 보유 종목을 watch_list 앞에 추가 (매도 누락 방지) ◄────────┘
 │
 └─ Claude 에이전트 실행 (아래 에이전트 워크플로우)
     │
     └─ 3분 대기 → 반복


### 에이전트 분석 워크플로우 (매 실행)

 잔고/보유 종목 조회
        │
        ▼
 [보유 종목 매도 점검] ─────────────────────────────────────────┐
        │                                                        │
        ├─ get_current_price                                     │
        ├─ get_technical_indicators                              │
        ├─ get_ml_prediction                                     │
        ├─ search_news_sentiment ← 뉴스 감성 확인                │
        │                                                        │
        └─ 매도 조건 판단                                        │
            ├─ ML 확률 < 0.45                                    │
            ├─ RSI > 75                                          │
            ├─ 손실률 ≤ -7%                                      │
            ├─ 수익률 ≥ +8%                                      │
            └─ 뉴스 부정 + 보유 중 → place_order(SELL) ──────────┘
        │
        ▼
 [신규 매수 후보 분석]
        │
        ├─ get_current_price
        ├─ get_technical_indicators
        ├─ get_ml_prediction
        ├─ search_news_sentiment ← 뉴스 감성 확인
        │
        └─ 매수 조건 판단
            ├─ ML 확률 ≥ 0.48
            ├─ RSI < 75
            ├─ 잔고 ≥ 현재가 × 수량
            └─ 뉴스 긍정 → place_order(BUY)
               뉴스 부정 → 매수 보류 (ML 무관)
               뉴스 중립 → ML/RSI 기준만 적용
```

### 감성 분석 흐름

```
search_news_sentiment(stock_code, stock_name)
        │
        ▼
Google News RSS (한국어)
"삼성전자 005930 주식" 검색
        │
        ▼
최근 헤드라인 10개 수집
        │
        ▼
Claude가 헤드라인 읽고 직접 판단
  → 긍정 / 부정 / 중립
        │
        ▼
매수/매도 최종 결정에 반영
```

---

## 디렉토리 구조

```
stock/
├── app.py                        # CLI 진입점
├── agent.py                      # Claude tool-use 에이전트 (핵심)
├── start.sh                      # 자동매매 시작 스크립트 (crontab용)
├── stop.sh                       # 자동매매 종료 스크립트 (crontab용)
├── .env                          # 환경변수 (API 키 등)
├── config/
│   └── settings.py               # 환경변수 로드 및 공통 설정
├── collector/
│   ├── kis_client.py             # KIS API (현재가 · 잔고 · 주문)
│   ├── data_fetcher.py           # yfinance 분봉 수집 (5거래일)
│   └── recommender.py            # KOSPI 종목 스크리닝
├── features/
│   └── make_features.py          # 기술적 지표 피처 생성
├── models/
│   ├── train_model.py            # RandomForest 모델 학습
│   └── predict.py                # 예측
├── data/
│   ├── minute/                   # 분봉 CSV (yfinance)
│   └── processed/                # 피처 CSV
├── artifacts/
│   ├── stock_updown_random_forest.joblib  # 학습된 ML 모델
│   └── orders.log                # 주문 이력
└── logs/
    ├── trade_YYYYMMDD.log        # 일별 매매 로그
    └── cron.log                  # crontab 실행 로그
```

---

## 설치

```bash
pip install anthropic python-dotenv requests pandas scikit-learn joblib \
            yfinance finance-datareader
```

---

## 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 아래 항목 입력
```

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

> KIS API 키는 모의투자용과 실전투자용이 별도입니다.
> 실전 주문 전 반드시 `KIS_IS_PAPER=true`, `DRY_RUN=true`로 먼저 테스트하세요.

---

## 명령어

### 데이터 수집 및 모델 학습

```bash
# 분봉 데이터 수집 (yfinance, 최근 5거래일)
python app.py collect

# 특정 종목만 수집
python app.py collect --symbols 005930,000660

# 피처 생성 (MA, RSI, 변동성 등)
python app.py features

# RandomForest 모델 학습
python app.py train
```

### 에이전트 1회 실행

```bash
# .env의 WATCH_LIST 종목 분석 후 매매
python app.py agent

# 특정 종목만
python app.py agent --symbols 005930,000660

# 예산 지정
python app.py agent --budget 30000

# 시뮬레이션 강제
python app.py agent --dry-run
```

### 자동매매 루프 (장중 반복 실행)

```bash
# 기본 실행 — KOSPI 스크리닝 → 3분마다 에이전트 반복
python app.py loop

# 예산 지정
python app.py loop --budget 100000

# 실행 간격 변경
LOOP_INTERVAL_MIN=10 python app.py loop --budget 100000

# 종료: Ctrl+C
```

루프 동작:
- 장 시작(09:00) ~ 장 마감(15:20) 사이에만 실행
- 매 3분마다 에이전트 실행 (`LOOP_INTERVAL_MIN`으로 조정)
- 4회마다 KOSPI 전체 재스크리닝 (`SCREEN_EVERY_N`으로 조정)
- 장 마감 후 다음날 09:00까지 자동 대기

### KOSPI 추천

```bash
# KOSPI 스크리닝 → Claude가 TOP 5 추천
python app.py recommend
```

### 특정 종목 전체 파이프라인

```bash
python app.py collect --symbols 203650 && \
python app.py features && \
python app.py train && \
python app.py agent --symbols 203650 --budget 30000
```

---

## 매매 조건

### 매수 (모두 만족 시)

| 조건 | 기준 |
|---|---|
| ML 상승확률 | ≥ 0.48 |
| RSI14 | < 75 |
| 잔고 | ≥ 현재가 × 주문수량 |
| 뉴스 감성 | 긍정 또는 중립 (부정이면 보류) |

### 매도 (하나라도 해당 시)

| 조건 | 기준 |
|---|---|
| ML 상승확률 | < 0.45 |
| RSI14 | > 75 (과매수) |
| 손실률 | ≤ -7% (손절) |
| 수익률 | ≥ +8% (익절) |
| 뉴스 감성 | 부정 (보유 종목에 적용) |

---

## crontab 설정

매일 09:00 자동 시작, 15:30 자동 종료:

```bash
crontab -e
```

```
0 9 * * 1-5 /home/dev/project/stock/start.sh >> /home/dev/project/stock/logs/cron.log 2>&1
30 15 * * 1-5 /home/dev/project/stock/stop.sh >> /home/dev/project/stock/logs/cron.log 2>&1
```

로그 확인:

```bash
tail -f /home/dev/project/stock/logs/trade_$(date +%Y%m%d).log
```

---

## 주의사항

- 이 프로그램은 투자 참고용이며 수익을 보장하지 않습니다.
- 실전 투자 전 반드시 모의투자(`KIS_IS_PAPER=true`)로 충분히 테스트하세요.
- `DRY_RUN=false`로 변경하면 실제 주문이 체결됩니다.
