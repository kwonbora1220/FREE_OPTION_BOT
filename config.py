"""
FREE OPTION BOT
Configuration

무료 데이터 기반 옵션 분석 봇의 기본 설정.
비밀값은 GitHub Secrets / 환경변수에서 가져온다.
"""

import os


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ============================================================
# OPTION DATA
# ============================================================

# 첫 번째 데이터 검증 종목
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "NVDA").upper()

# 1일 옵션 분석에서 사용할 최대 DTE
ONE_DAY_MAX_DTE = int(os.getenv("ONE_DAY_MAX_DTE", "1"))

# Yahoo Finance 요청 간 최소 대기시간
REQUEST_DELAY_SECONDS = float(
    os.getenv("REQUEST_DELAY_SECONDS", "0.5")
)


# ============================================================
# DATA QUALITY
# ============================================================

# 옵션 체인에서 반드시 필요한 컬럼
REQUIRED_OPTION_COLUMNS = [
    "contractSymbol",
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "volume",
    "openInterest",
    "impliedVolatility",
]


# ============================================================
# SUPPORTED SYMBOLS
# ============================================================

DEFAULT_SYMBOLS = [
    "NVDA",
    "MRVL",
    "RKLB",
    "IREN",
    "ASTS",
    "TSLA",
    "AMD",
    "MSFT",
    "AAPL",
    "SPY",
    "QQQ",
]


# ============================================================
# APPLICATION
# ============================================================

BOT_NAME = "FREE OPTION BOT"

VERSION = "0.1.0"

DATA_SOURCE = "Yahoo Finance / yfinance"

DATA_SOURCE_POLICY = {
    "paid_api": False,
    "unusual_whales": False,
    "realtime_options": False,
    "free_data": True,
}
