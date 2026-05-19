# Stock Auto Trading Agent

이 저장소는 한국 주식 데이터를 수집하고, 특성(feature) 기반 예측과 AWS Bedrock 기반 의사결정 보조를 연결하는 자동매매 에이전트 초기 구조입니다.

## 주요 구조
- `collector/`: KIS API로 분봉 데이터를 수집
- `features/`: 기술적 지표를 계산하는 feature 엔지니어링
- `models/`: 학습 및 예측
- `bedrock_client.py`: Bedrock 모델 호출 인터페이스
- `broker_client.py`: 주문 기록 및 모의주문 스텁
- `trade_agent.py`: 전체 파이프라인 제어
- `app.py`: CLI 진입점

## 사용법
### 1) 데이터 수집
```bash
python app.py collect
```

### 2) 특징 생성
```bash
python app.py features
```

### 3) 모델 학습
```bash
python app.py train
```

### 4) 예측
```bash
python app.py predict
```

### 5) Bedrock 보조 자동매매 실행 (모의)
```bash
python app.py agent --dry-run
```

## 환경 변수
- `WATCH_LIST`: 콤마 구분 종목 코드 목록, 기본값: `005930,000660,035420,051910,207940`
- `TRADE_QUANTITY`: 모의 주문 수량, 기본값: `1`
- `AWS_DEFAULT_REGION`: Bedrock 리전, 기본값: `ap-northeast-2`
- `BEDROCK_MODEL_ID`: 사용할 Bedrock 모델 ID
- `BROKER_ORDER_LOG`: 주문 로그 파일 경로

## 주의사항
- `bedrock_client.py`는 `boto3`가 설치되어 있어야 합니다.
- 실제 주문 연동 시 `broker_client.py`를 증권사 API에 맞게 수정하세요.
- 실거래 전에는 충분한 백테스트와 시뮬레이션을 권장합니다.
