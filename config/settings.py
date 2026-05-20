import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")


def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"환경변수 {key} 가 설정되지 않았습니다. .env 파일을 확인하세요.")
    return val


# Anthropic
ANTHROPIC_API_KEY = _require("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# KIS
KIS_APP_KEY = _require("KIS_APP_KEY")
KIS_APP_SECRET = _require("KIS_APP_SECRET")

_acct = _require("KIS_ACCOUNT_NO")
_parts = _acct.replace("-", "").split("-") if "-" in _acct else [_acct[:8], _acct[8:] or "01"]
KIS_ACCOUNT_CANO = _acct.replace("-", "")[:8]
KIS_ACCOUNT_PRDT = _acct.replace("-", "")[8:] or "01"
if "-" in _acct:
    _split = _acct.split("-")
    KIS_ACCOUNT_CANO = _split[0]
    KIS_ACCOUNT_PRDT = _split[1] if len(_split) > 1 else "01"

KIS_IS_PAPER = os.environ.get("KIS_IS_PAPER", "true").lower() == "true"
KIS_BASE_URL = (
    "https://openapivts.koreainvestment.com:29443"
    if KIS_IS_PAPER
    else "https://openapi.koreainvestment.com:9443"
)

# Paths
DATA_DIR = BASE_DIR / "data"
MINUTE_DIR  = DATA_DIR / "minute"    # 1분봉 (yfinance, 최대 7일)
MINUTE5_DIR = DATA_DIR / "minute5"   # 5분봉 (yfinance, 최대 60일) — PatchTST용
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = DATA_DIR / "artifacts"
ORDER_LOG = ARTIFACT_DIR / "orders.log"

for _d in (MINUTE_DIR, MINUTE5_DIR, RAW_DIR, PROCESSED_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Trading
WATCH_LIST = [
    s.strip()
    for s in os.environ.get("WATCH_LIST", "005930,000660").split(",")
    if s.strip()
]
TRADE_QUANTITY = int(os.environ.get("TRADE_QUANTITY", "1"))
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() == "true"
