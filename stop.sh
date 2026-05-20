#!/bin/bash
# 자동매매 루프 중단 스크립트
# crontab: 30 15 * * 1-5 /home/dev/project/stock/stop.sh

STOCK_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$STOCK_DIR/logs"
TODAY="$(date +%Y%m%d)"
LOG_FILE="$LOG_DIR/trade_${TODAY}.log"
PID_FILE="$LOG_DIR/.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "[$(date '+%H:%M:%S')] PID 파일 없음 — 실행 중이 아닌 것 같습니다."
    exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    sleep 1
    # 그래도 살아있으면 강제 종료
    if kill -0 "$PID" 2>/dev/null; then
        kill -9 "$PID"
    fi
    MSG="[$(date '+%H:%M:%S')] 자동매매 종료 (PID: $PID)"
    echo "$MSG"
    echo "" >> "$LOG_FILE"
    echo "======================================================" >> "$LOG_FILE"
    echo "  자동매매 종료: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
    echo "======================================================" >> "$LOG_FILE"
else
    echo "[$(date '+%H:%M:%S')] 프로세스 $PID 는 이미 종료되어 있습니다."
fi

rm -f "$PID_FILE"
