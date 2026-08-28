"""
FREE OPTION BOT
Telegram Bot - STEP 1

현재 단계:
- Telegram 연결 테스트
- /start
- /status
"""

from __future__ import annotations

import os
import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from config import (
    TELEGRAM_BOT_TOKEN,
    BOT_NAME,
    VERSION,
    DATA_SOURCE,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
    level=logging.INFO,
)

logger = logging.getLogger(
    "FREE_OPTION_BOT"
)


# ============================================================
# COMMANDS
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    message = (
        "🤖 <b>FREE OPTION BOT</b>\n"
        "━━━━━━━━━━━━━━━━\n\n"
        "현재 STEP 1 테스트 버전입니다.\n\n"
        "데이터 소스:\n"
        f"• {DATA_SOURCE}\n\n"
        "현재 구현:\n"
        "🟢 Telegram 연결\n"
        "🟡 Option Collector 테스트\n"
        "⚪ MaxPain\n"
        "⚪ GEX\n"
        "⚪ Whale Estimate\n"
        "⚪ ARK\n"
        "⚪ 종합분석\n\n"
        f"Version: {VERSION}"
    )

    await update.message.reply_text(
        message,
        parse_mode="HTML",
    )


async def status(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    token_status = (
        "🟢 OK"
        if TELEGRAM_BOT_TOKEN
        else "🔴 MISSING"
    )

    message = (
        "⚙️ <b>SYSTEM STATUS</b>\n"
        "━━━━━━━━━━━━━━━━\n\n"
        f"BOT TOKEN: {token_status}\n"
        "DATA SOURCE: 🟢 Yahoo Finance\n"
        "PAID API: 🔴 NO\n"
        "UNUSUAL WHALES: 🔴 NO\n"
        "REALTIME OPTIONS: 🔴 NO\n"
        f"VERSION: {VERSION}"
    )

    await update.message.reply_text(
        message,
        parse_mode="HTML",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set."
        )

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    application.add_handler(
        CommandHandler(
            "status",
            status,
        )
    )

    logger.info(
        "%s %s starting...",
        BOT_NAME,
        VERSION,
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
