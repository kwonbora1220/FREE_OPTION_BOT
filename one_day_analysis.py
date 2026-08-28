"""
FREE OPTION BOT
1-DAY OPTION ANALYSIS

Collector
    ↓
Normalizer
    ↓
1-Day Analysis

현재 구현:
- CALL / PUT Volume
- CALL / PUT OI
- CALL / PUT Premium
- Volume Ratio
- OI Ratio
- Premium Ratio
- Put/Call Volume Ratio
- Put/Call OI Ratio
- ATM Strike
- ATM IV
- 가장 큰 CALL / PUT Volume Strike
- 가장 큰 CALL / PUT OI Strike
- 가장 큰 CALL / PUT Premium Strike
- 기본 Direction Score

주의:
이 단계의 Direction은 투자 신호가 아니라
옵션 구조를 요약한 지표다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import DEFAULT_SYMBOL
from option_collector import OptionCollector
from normalizer import normalize_options


# ============================================================
# HELPERS
# ============================================================

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    0으로 나누는 것을 방지한다.
    """

    if denominator == 0:
        return 0.0

    return numerator / denominator


def money(
    value: float,
) -> str:
    """
    Premium 표시.
    """

    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"

    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"

    if value >= 1_000:
        return f"${value / 1_000:.2f}K"

    return f"${value:.0f}"


def number(
    value: float,
) -> str:

    return f"{value:,.0f}"


def price(
    value: float | None,
) -> str:

    if value is None:
        return "N/A"

    return f"${value:,.2f}"


def percent(
    value: float,
) -> str:

    return f"{value:.1f}%"


# ============================================================
# ANALYZER
# ============================================================

class OneDayAnalyzer:

    def __init__(
        self,
        df: pd.DataFrame,
    ):

        self.df = df.copy()

    # ========================================================
    # BASIC
    # ========================================================

    def _side(
        self,
        option_type: str,
    ) -> pd.DataFrame:

        return self.df[
            self.df["option_type"]
            == option_type
        ].copy()

    # ========================================================
    # VOLUME
    # ========================================================

    def volume_summary(
        self,
    ) -> dict[str, float]:

        calls = self._side("CALL")
        puts = self._side("PUT")

        call_volume = float(
            calls["volume"].sum()
        )

        put_volume = float(
            puts["volume"].sum()
        )

        total = (
            call_volume
            + put_volume
        )

        return {
            "call": call_volume,
            "put": put_volume,
            "total": total,
            "call_ratio": safe_divide(
                call_volume,
                total,
            ),
            "put_ratio": safe_divide(
                put_volume,
                total,
            ),
            "put_call_ratio": safe_divide(
                put_volume,
                call_volume,
            ),
        }

    # ========================================================
    # OPEN INTEREST
    # ========================================================

    def oi_summary(
        self,
    ) -> dict[str, float]:

        calls = self._side("CALL")
        puts = self._side("PUT")

        call_oi = float(
            calls["openInterest"].sum()
        )

        put_oi = float(
            puts["openInterest"].sum()
        )

        total = (
            call_oi
            + put_oi
        )

        return {
            "call": call_oi,
            "put": put_oi,
            "total": total,
            "call_ratio": safe_divide(
                call_oi,
                total,
            ),
            "put_ratio": safe_divide(
                put_oi,
                total,
            ),
            "put_call_ratio": safe_divide(
                put_oi,
                call_oi,
            ),
        }

    # ========================================================
    # PREMIUM
    # ========================================================

    def premium_summary(
        self,
    ) -> dict[str, float]:

        calls = self._side("CALL")
        puts = self._side("PUT")

        call_premium = float(
            calls["premium"].sum()
        )

        put_premium = float(
            puts["premium"].sum()
        )

        total = (
            call_premium
            + put_premium
        )

        return {
            "call": call_premium,
            "put": put_premium,
            "total": total,
            "call_ratio": safe_divide(
                call_premium,
                total,
            ),
            "put_ratio": safe_divide(
                put_premium,
                total,
            ),
            "put_call_ratio": safe_divide(
                put_premium,
                call_premium,
            ),
        }

    # ========================================================
    # ATM
    # ========================================================

    def atm_row(
        self,
    ) -> pd.Series | None:

        if self.df.empty:
            return None

        valid = self.df[
            self.df["underlying_price"].notna()
            & self.df["strike"].notna()
        ].copy()

        if valid.empty:
            return None

        valid["distance"] = (
            (
                valid["strike"]
                - valid["underlying_price"]
            )
            .abs()
        )

        index = (
            valid["distance"]
            .idxmin()
        )

        return valid.loc[index]

    def atm_summary(
        self,
    ) -> dict[str, Any]:

        row = self.atm_row()

        if row is None:

            return {
                "strike": None,
                "iv": None,
                "underlying_price": None,
            }

        return {
            "strike": float(
                row["strike"]
            ),
            "iv": float(
                row["impliedVolatility"]
            )
            if pd.notna(
                row["impliedVolatility"]
            )
            else None,
            "underlying_price": float(
                row["underlying_price"]
            )
            if pd.notna(
                row["underlying_price"]
            )
            else None,
        }

    # ========================================================
    # TOP STRIKES
    # ========================================================

    def top_strikes(
        self,
    ) -> dict[str, Any]:

        result = {}

        for option_type in [
            "CALL",
            "PUT",
        ]:

            side = self._side(
                option_type
            )

            if side.empty:
                continue

            # --------------------------------------------
            # Volume
            # --------------------------------------------

            volume_row = side.loc[
                side["volume"].idxmax()
            ]

            # --------------------------------------------
            # OI
            # --------------------------------------------

            oi_row = side.loc[
                side["openInterest"].idxmax()
            ]

            # --------------------------------------------
            # Premium
            # --------------------------------------------

            premium_row = side.loc[
                side["premium"].idxmax()
            ]

            result[
                option_type.lower()
            ] = {
                "volume_strike": float(
                    volume_row["strike"]
                ),
                "volume": float(
                    volume_row["volume"]
                ),
                "oi_strike": float(
                    oi_row["strike"]
                ),
                "oi": float(
                    oi_row["openInterest"]
                ),
                "premium_strike": float(
                    premium_row["strike"]
                ),
                "premium": float(
                    premium_row["premium"]
                ),
            }

        return result

    # ========================================================
    # DIRECTION
    # ========================================================

    def direction(
        self,
        volume: dict[str, float],
        oi: dict[str, float],
        premium: dict[str, float],
    ) -> dict[str, Any]:
        """
        기본 구조 점수.

        Volume 40%
        OI     25%
        Premium 35%

        CALL 비중이 높을수록 bullish.
        PUT 비중이 높을수록 bearish.

        주의:
        실제 거래 방향(BTO/STO/BTC/STC)을
        확인하는 것이 아니다.
        """

        volume_call = (
            volume["call_ratio"]
        )

        oi_call = (
            oi["call_ratio"]
        )

        premium_call = (
            premium["call_ratio"]
        )

        score = (
            volume_call * 40
            + oi_call * 25
            + premium_call * 35
        )

        score = max(
            0,
            min(
                100,
                score,
            ),
        )

        if score >= 60:

            signal = "BULLISH"

        elif score <= 40:

            signal = "BEARISH"

        else:

            signal = "MIXED"

        return {
            "score": score,
            "signal": signal,
        }

    # ========================================================
    # FULL ANALYSIS
    # ========================================================

    def analyze(
        self,
        symbol: str,
        expiration: str | None,
        dte: int | None,
        current_price: float | None,
        quality: dict[str, Any],
    ) -> dict[str, Any]:

        volume = (
            self.volume_summary()
        )

        oi = (
            self.oi_summary()
        )

        premium = (
            self.premium_summary()
        )

        atm = (
            self.atm_summary()
        )

        tops = (
            self.top_strikes()
        )

        direction = (
            self.direction(
                volume,
                oi,
                premium,
            )
        )

        return {
            "symbol": symbol,
            "expiration": expiration,
            "DTE": dte,
            "current_price": current_price,
            "quality": quality,
            "volume": volume,
            "open_interest": oi,
            "premium": premium,
            "atm": atm,
            "top_strikes": tops,
            "direction": direction,
        }


# ============================================================
# TEXT REPORT
# ============================================================

def print_report(
    result: dict[str, Any],
) -> None:

    symbol = result[
        "symbol"
    ]

    volume = result[
        "volume"
    ]

    oi = result[
        "open_interest"
    ]

    premium = result[
        "premium"
    ]

    atm = result[
        "atm"
    ]

    direction = result[
        "direction"
    ]

    quality = result[
        "quality"
    ]

    tops = result[
        "top_strikes"
    ]

    print()
    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        f"🔥 {symbol} 1-DAY OPTION"
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        f"💰 Current Price: "
        f"{price(result['current_price'])}"
    )

    print(
        f"📅 Expiration: "
        f"{result['expiration']}"
    )

    print(
        f"⏳ DTE: "
        f"{result['DTE']}"
    )

    print()

    print(
        "📊 OPTION FLOW"
    )

    print(
        f"CALL Volume : "
        f"{number(volume['call'])}"
    )

    print(
        f"PUT Volume  : "
        f"{number(volume['put'])}"
    )

    print(
        f"CALL Ratio  : "
        f"{percent(volume['call_ratio'] * 100)}"
    )

    print(
        f"PUT Ratio   : "
        f"{percent(volume['put_ratio'] * 100)}"
    )

    print(
        f"PUT/CALL    : "
        f"{volume['put_call_ratio']:.2f}"
    )

    print()

    print(
        "📈 OPEN INTEREST"
    )

    print(
        f"CALL OI     : "
        f"{number(oi['call'])}"
    )

    print(
        f"PUT OI      : "
        f"{number(oi['put'])}"
    )

    print(
        f"PUT/CALL OI : "
        f"{oi['put_call_ratio']:.2f}"
    )

    print()

    print(
        "💰 PREMIUM"
    )

    print(
        f"CALL Premium: "
        f"{money(premium['call'])}"
    )

    print(
        f"PUT Premium : "
        f"{money(premium['put'])}"
    )

    print(
        f"PUT/CALL    : "
        f"{premium['put_call_ratio']:.2f}"
    )

    print()

    print(
        "🎯 ATM"
    )

    print(
        f"ATM Strike  : "
        f"{price(atm['strike'])}"
    )

    if atm["iv"] is not None:

        print(
            f"ATM IV      : "
            f"{atm['iv'] * 100:.1f}%"
        )

    else:

        print(
            "ATM IV      : N/A"
        )

    print()

    print(
        "🔥 TOP STRIKES"
    )

    for option_type in [
        "call",
        "put",
    ]:

        if option_type not in tops:
            continue

        item = tops[
            option_type
        ]

        print(
            f"{option_type.upper()}"
        )

        print(
            f"  Volume Strike : "
            f"{item['volume_strike']:.2f} "
            f"({number(item['volume'])})"
        )

        print(
            f"  OI Strike     : "
            f"{item['oi_strike']:.2f} "
            f"({number(item['oi'])})"
        )

        print(
            f"  Premium Strike: "
            f"{item['premium_strike']:.2f} "
            f"({money(item['premium'])})"
        )

    print()

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        "🧠 STRUCTURE SIGNAL"
    )

    print(
        f"Score : "
        f"{direction['score']:.1f}/100"
    )

    print(
        f"Signal: "
        f"{direction['signal']}"
    )

    print()

    print(
        "📋 DATA QUALITY"
    )

    print(
        f"Quality: "
        f"{quality['quality']}"
    )

    print(
        f"Score  : "
        f"{quality['score']}/100"
    )

    print(
        f"Rows   : "
        f"{quality['rows']}"
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        "⚠️ Direction is an option-structure "
        "estimate, not confirmed trade direction."
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    symbol = DEFAULT_SYMBOL

    print(
        f"Collecting {symbol}..."
    )

    collector = OptionCollector(
        symbol
    )

    raw_result = (
        collector.collect_one_day()
    )

    if not raw_result["success"]:

        print(
            "❌ Collector failed:"
        )

        print(
            raw_result["error"]
        )

        return

    raw_df = pd.DataFrame(
        raw_result["data"]
    )

    normalized_df, quality = (
        normalize_options(
            raw_df
        )
    )

    result = (
        OneDayAnalyzer(
            normalized_df
        ).analyze(
            symbol=symbol,
            expiration=raw_result[
                "expiration"
            ],
            dte=raw_result[
                "DTE"
            ],
            current_price=raw_result[
                "current_price"
            ],
            quality=quality,
        )
    )

    print_report(
        result
    )


if __name__ == "__main__":

    main()
