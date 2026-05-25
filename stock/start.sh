#!/bin/bash
# 자동매매 루프 시작 스크립트
# crontab: 0 9 * * 1-5 /home/dev/project/stock/start.sh

STOCK_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/dev/miniconda3/envs/stock/bin/python"
LOG_DIR="$STOCK_DIR/logs"
TODAY="$(date +%Y%m%d)"
LOG_FILE="$LOG_DIR/trade_${TODAY}.log"
PID_FILE="$LOG_DIR/.pid"

mkdir -p "$LOG_DIR"

# 이미 실행 중이면 중복 실행 방지
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] 이미 실행 중 (PID: $PID). 종료 후 재시작합니다." | tee -a "$LOG_FILE"
        kill "$PID"
        sleep 2
    fi
fi

cd "$STOCK_DIR"

echo "" >> "$LOG_FILE"
echo "======================================================" >> "$LOG_FILE"
echo "  자동매매 시작: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "  예산: ${BUDGET:-(계좌 잔고 기준)}" >> "$LOG_FILE"
echo "======================================================" >> "$LOG_FILE"

# BUDGET 환경변수가 있으면 --budget 옵션 추가
if [ -n "$BUDGET" ]; then
    BUDGET_OPT="--budget $BUDGET"
else
    BUDGET_OPT=""
fi

# -u: Python 출력 버퍼링 해제, 타임스탬프 붙여서 로그에 기록
LOOP_INTERVAL_MIN=60 "$PYTHON" -u app.py loop $BUDGET_OPT 2>&1 | \
    awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }' >> "$LOG_FILE" &

echo $! > "$PID_FILE"
echo "[$(date '+%H:%M:%S')] 시작됨 — PID: $!, 로그: $LOG_FILE"
