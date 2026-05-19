# 1. 의존성 설치
pip install anthropic python-dotenv requests pandas scikit-learn joblib

# 2. 환경변수 설정
cp .env.example .env
# .env 파일에 실제 키 값 입력

# 3. 데이터 수집 → 피처 생성 → 모델 학습 (선택사항)
python app.py collect
python app.py features
python app.py train

# 4. 에이전트 실행 (DRY_RUN=true로 시작 권장)
python app.py agent


#---------------------------------------------------------------------

# 1. 해당 종목만 분봉 수집
python app.py collect --symbols 203650

# 2. 피처 재생성 (전체 종목 대상이지만 203650 데이터도 포함됨)
python app.py features

# 3. 모델 재학습
python app.py train

# 4. 해당 종목만 에이전트 실행
python app.py agent --symbols 203650
--symbols 플래그를 쓰면 .env의 WATCH_LIST를 무시하고 지정한 종목만 처리합니다.

한 번에 붙여서 실행하려면:


python app.py collect --symbols 203650 && \
python app.py features && \
python app.py train && \
python app.py agent --symbols 203650


#--------------------------------------------------------------------------
python app.py agent --symbols 203650 --budget 30000

#--------------------------------------------------------------------------
python app.py recommend


#--------------------------------------------------------------------------
# 예산 없이 계좌 잔고 전액 기준으로 30분마다 자동매매
python app.py loop

# 예산 3만원으로 30분마다 자동매매
python app.py loop --budget 30000

# 10분마다 더 자주 실행
LOOP_INTERVAL_MIN=10 python app.py loop --budget 30000
