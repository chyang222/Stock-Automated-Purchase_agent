# 주식 자동매매 에이전트

Claude API(Tool Use)와 한국투자증권(KIS) OpenAPI를 연동한 KOSPI 자동매매 시스템입니다.

## 구조

```
stock/
├── app.py                        # CLI 진입점
├── agent.py                      # Claude tool-use 에이전트 (핵심)
├── .env.example                  # 환경변수 템플릿
├── config/
│   └── settings.py               # 환경변수 로드 및 공통 설정
├── collector/
│   ├── kis_client.py             # KIS API (현재가·잔고·주문)
│   ├── data_fetcher.py           # yfinance 분봉 수집
│   └── recommender.py            # KOSPI 종목 스크리닝
├── features/
│   └── make_features.py          # 기술적 지표 피처 생성
└── models/
    ├── train_model.py            # RandomForest 모델 학습
    └── predict.py                # 예측
```

## 동작 방식

1. **KOSPI 스크리닝** — FinanceDataReader로 전체 종목 수집 → 가격·거래량·RSI 필터 → ML 상승확률 순위
2. **에이전트 분석** — Claude가 6개 도구(현재가·기술지표·ML예측·잔고·보유종목·주문)를 자율 호출
3. **매수/매도 실행** — 조건 충족 시 KIS API로 주문 (DRY_RUN=true면 시뮬레이션)
4. **루프** — 장중 N분마다 자동 반복, 장 마감 후 다음날 개장까지 대기

## 설치

```bash
pip install anthropic python-dotenv requests pandas scikit-learn joblib \
            yfinance finance-datareader
```

## 환경변수 설정

```bash
cp .env.example .env
```

`.env` 필수 항목:

| 변수 | 설명 |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic 콘솔에서 발급 |
| `KIS_APP_KEY` | KIS 개발자 포털 앱키 |
| `KIS_APP_SECRET` | KIS 개발자 포털 앱시크릿 |
| `KIS_ACCOUNT_NO` | 계좌번호 (형식: `12345678-01`) |
| `KIS_IS_PAPER` | `true` = 모의투자, `false` = 실전 |
| `DRY_RUN` | `true` = 시뮬레이션만, `false` = 실제 주문 |
| `WATCH_LIST` | 감시 종목 코드 (쉼표 구분, 예: `005930,000660`) |
| `TRADE_QUANTITY` | 종목당 최대 주문 수량 |

> KIS API 키는 모의투자용과 실전투자용이 별도입니다.  
> 실전 주문 전 반드시 `KIS_IS_PAPER=true`, `DRY_RUN=true`로 먼저 테스트하세요.

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

### 에이전트 실행 (1회)

```bash
# .env의 WATCH_LIST 종목 분석 후 매매
python app.py agent

# 특정 종목만
python app.py agent --symbols 005930,000660

# 예산 지정 (계좌 잔고 무관하게 이 금액 내에서만 매수)
python app.py agent --budget 30000

# 시뮬레이션 강제
python app.py agent --dry-run
```

### 자동매매 루프 (장중 반복 실행)

```bash
# 기본 실행 — KOSPI 스크리닝 → 30분마다 에이전트 반복
python app.py loop

# 예산 지정
python app.py loop --budget 100000

# 실행 간격 변경 (환경변수)
LOOP_INTERVAL_MIN=10 python app.py loop --budget 100000

# 종료: Ctrl+C
```

루프 동작:
- 장 시작(09:00) ~ 장 마감(15:20) 사이에만 실행
- 매 30분마다 에이전트 실행 (LOOP_INTERVAL_MIN으로 조정)
- 4회마다 KOSPI 전체 재스크리닝 (SCREEN_EVERY_N으로 조정)
- 장 마감 후 다음날 09:00까지 자동 대기

### KOSPI 추천

```bash
# KOSPI 시총 상위 80개 분석 → Claude가 TOP 5 추천
python app.py recommend
```

### 특정 종목 전체 파이프라인

```bash
python app.py collect --symbols 203650 && \
python app.py features && \
python app.py train && \
python app.py agent --symbols 203650 --budget 30000
```

## 매매 조건

**매수** (모두 만족 시)
- ML 상승확률 ≥ 0.55
- RSI14 < 65
- 현재가 > MA5 (단기 상승 추세)
- 예산(또는 잔고) ≥ 현재가 × 주문수량

**매도** (하나라도 만족 시)
- ML 상승확률 < 0.40
- RSI14 > 70 (과매수)
- 손실률 ≤ −5%

## 주의사항

- 이 프로그램은 투자 참고용이며 수익을 보장하지 않습니다.
- 실전 투자 전 반드시 모의투자(`KIS_IS_PAPER=true`)로 충분히 테스트하세요.
- `DRY_RUN=false`로 변경하면 실제 주문이 체결됩니다.
