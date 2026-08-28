"""
FREE OPTION BOT
CONFIGURATION

무료 데이터 기반 옵션 분석 봇 설정.

Data:
    Yahoo Finance / yfinance

Policy:
    Paid API              = False
    Unusual Whales        = False
    Realtime Options      = False
    Free Data             = True
"""

from __future__ import annotations

import os


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "",
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    "",
)


# ============================================================
# OPTION DATA
# ============================================================

DEFAULT_SYMBOL = os.getenv(
    "DEFAULT_SYMBOL",
    "NVDA",
).upper()


ONE_DAY_MAX_DTE = int(
    os.getenv(
        "ONE_DAY_MAX_DTE",
        "1",
    )
)


REQUEST_DELAY_SECONDS = float(
    os.getenv(
        "REQUEST_DELAY_SECONDS",
        "0.5",
    )
)


# ============================================================
# GEX
# ============================================================

GEX_RISK_FREE_RATE = float(
    os.getenv(
        "GEX_RISK_FREE_RATE",
        "0.04",
    )
)


GEX_PERCENT_MOVE = float(
    os.getenv(
        "GEX_PERCENT_MOVE",
        "0.01",
    )
)


CONTRACT_MULTIPLIER = 100


# ============================================================
# DATA QUALITY
# ============================================================

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

BOT_NAME = (
    "FREE OPTION BOT"
)


VERSION = (
    "0.2.0"
)


DATA_SOURCE = (
    "Yahoo Finance / yfinance"
)


DATA_SOURCE_POLICY = {

    "paid_api":
        False,

    "unusual_whales":
        False,

    "realtime_options":
        False,

    "free_data":
        True,

}


# ============================================================
# MENU
# ============================================================

MENU_ITEMS = [

    "📊 1일 옵션",

    "🐋 기관/고래 옵션",

    "🎯 MaxPain",

    "🦅 ARK 포지션",

    "📈 종합분석",

    "⚙️ 시스템",

]
