"""
FREE OPTION BOT
MAX PAIN ANALYSIS

Collector
    ↓
Normalizer
    ↓
MaxPain

무료 Yahoo Finance 옵션체인의
Strike / CALL OI / PUT OI를 이용해서
이론적 Max Pain을 계산한다.

정의:
각 만기에서 옵션 매수자가 만기 시점에
가장 적은 총 손실을 보는 Strike를 Max Pain으로 계산한다.

중요:
Max Pain은 만기 가격을 예측하는 확정 신호가 아니다.
옵션 OI 기반의 구조적 참고 지표다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import DEFAULT_SYMBOL
from option_collector import OptionCollector
from normalizer import normalize_options


# ============================================================
# CONSTANTS
# ============================================================

CONTRACT_MULTIPLIER = 100


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


def format_price(
    value: float | None,
) -> str:

    if value is None:
        return "N/A"

    return f"${value:,.2f}"


def format_number(
    value: float,
) -> str:

    return f"{value:,.0f}"


def format_money(
    value: float,
) -> str:

    if value >= 1_000_000_000:

        return (
            f"${value / 1_000_000_000:.2f}B"
        )

    if value >= 1_000_000:

        return (
            f"${value / 1_000_000:.2f}M"
        )

    if value >= 1_000:

        return (
            f"${value / 1_000:.2f}K"
        )

    return f"${value:.0f}"


# ============================================================
# MAX PAIN CALCULATOR
# ============================================================

class MaxPainCalculator:
    """
    옵션체인 DataFrame으로 Max Pain을 계산한다.
    """

    def __init__(
        self,
        df: pd.DataFrame,
    ):

        self.df = df.copy()

        self._prepare()

    # ========================================================
    # PREPARE
    # ========================================================

    def _prepare(
        self,
    ) -> None:

        if self.df.empty:
            return

        required_columns = [
            "option_type",
            "strike",
            "openInterest",
        ]

        for column in required_columns:

            if column not in self.df.columns:

                self.df[column] = 0

        # --------------------------------------------
        # Numeric
        # --------------------------------------------

        self.df["strike"] = pd.to_numeric(
            self.df["strike"],
            errors="coerce",
        )

        self.df["openInterest"] = pd.to_numeric(
            self.df["openInterest"],
            errors="coerce",
        )

        # --------------------------------------------
        # Clean
        # --------------------------------------------

        self.df = self.df[
            self.df["strike"].notna()
        ].copy()

        self.df["openInterest"] = (
            self.df["openInterest"]
            .fillna(0)
            .clip(lower=0)
        )

        self.df["option_type"] = (
            self.df["option_type"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        self.df = self.df[
            self.df["option_type"].isin(
                [
                    "CALL",
                    "PUT",
                ]
            )
        ].copy()

    # ========================================================
    # STRIKES
    # ========================================================

    def get_strikes(
        self,
    ) -> list[float]:

        if self.df.empty:

            return []

        strikes = sorted(
            self.df["strike"]
            .dropna()
            .unique()
            .tolist()
        )

        return [
            float(strike)
            for strike in strikes
        ]

    # ========================================================
    # OI BY STRIKE
    # ========================================================

    def aggregate_oi(
        self,
    ) -> pd.DataFrame:

        if self.df.empty:

            return pd.DataFrame(
                columns=[
                    "strike",
                    "call_oi",
                    "put_oi",
                ]
            )

        calls = (
            self.df[
                self.df["option_type"]
                == "CALL"
            ]
            .groupby("strike")[
                "openInterest"
            ]
            .sum()
            .rename("call_oi")
        )

        puts = (
            self.df[
                self.df["option_type"]
                == "PUT"
            ]
            .groupby("strike")[
                "openInterest"
            ]
            .sum()
            .rename("put_oi")
        )

        result = pd.concat(
            [
                calls,
                puts,
            ],
            axis=1,
        ).fillna(0)

        result = result.reset_index()

        result["strike"] = pd.to_numeric(
            result["strike"],
            errors="coerce",
        )

        result = result.sort_values(
            "strike"
        )

        result = result.reset_index(
            drop=True
        )

        return result

    # ========================================================
    # PAYOFF AT ONE PRICE
    # ========================================================

    def calculate_loss_at_price(
        self,
        settlement_price: float,
    ) -> dict[str, float]:
        """
        특정 만기 가격에서의
        CALL / PUT 이론적 손실을 계산한다.

        CALL:
            max(0, settlement - strike)

        PUT:
            max(0, strike - settlement)

        OI × 100 계약 승수 적용.
        """

        if self.df.empty:

            return {
                "call_loss": 0.0,
                "put_loss": 0.0,
                "total_loss": 0.0,
            }

        calls = self.df[
            self.df["option_type"]
            == "CALL"
        ]

        puts = self.df[
            self.df["option_type"]
            == "PUT"
        ]

        # ----------------------------------------------------
        # CALL LOSS
        # ----------------------------------------------------

        call_intrinsic = (
            settlement_price
            - calls["strike"]
        ).clip(lower=0)

        call_loss = (
            call_intrinsic
            * calls["openInterest"]
            * CONTRACT_MULTIPLIER
        ).sum()

        # ----------------------------------------------------
        # PUT LOSS
        # ----------------------------------------------------

        put_intrinsic = (
            puts["strike"]
            - settlement_price
        ).clip(lower=0)

        put_loss = (
            put_intrinsic
            * puts["openInterest"]
            * CONTRACT_MULTIPLIER
        ).sum()

        call_loss = float(
            call_loss
        )

        put_loss = float(
            put_loss
        )

        return {
            "call_loss": call_loss,
            "put_loss": put_loss,
            "total_loss": (
                call_loss
                + put_loss
            ),
        }

    # ========================================================
    # FULL CALCULATION
    # ========================================================

    def calculate(
        self,
        current_price: float | None = None,
    ) -> dict[str, Any]:

        if self.df.empty:

            return {
                "success": False,
                "max_pain": None,
                "current_price": current_price,
                "distance": None,
                "distance_percent": None,
                "call_oi": 0,
                "put_oi": 0,
                "total_oi": 0,
                "table": [],
                "error": "Empty option data.",
            }

        strikes = self.get_strikes()

        if not strikes:

            return {
                "success": False,
                "max_pain": None,
                "current_price": current_price,
                "distance": None,
                "distance_percent": None,
                "call_oi": 0,
                "put_oi": 0,
                "total_oi": 0,
                "table": [],
                "error": "No valid strikes.",
            }

        # ----------------------------------------------------
        # Total OI
        # ----------------------------------------------------

        call_oi = float(
            self.df[
                self.df["option_type"]
                == "CALL"
            ]["openInterest"].sum()
        )

        put_oi = float(
            self.df[
                self.df["option_type"]
                == "PUT"
            ]["openInterest"].sum()
        )

        total_oi = (
            call_oi
            + put_oi
        )

        # ----------------------------------------------------
        # Evaluate every strike
        # ----------------------------------------------------

        rows = []

        for settlement_price in strikes:

            loss = (
                self.calculate_loss_at_price(
                    settlement_price
                )
            )

            rows.append(
                {
                    "strike": settlement_price,
                    "call_loss": loss[
                        "call_loss"
                    ],
                    "put_loss": loss[
                        "put_loss"
                    ],
                    "total_loss": loss[
                        "total_loss"
                    ],
                }
            )

        table = pd.DataFrame(
            rows
        )

        # ----------------------------------------------------
        # Minimum total loss
        # ----------------------------------------------------

        min_index = (
            table["total_loss"]
            .idxmin()
        )

        max_pain = float(
            table.loc[
                min_index,
                "strike",
            ]
        )

        # ----------------------------------------------------
        # Current distance
        # ----------------------------------------------------

        distance = None
        distance_percent = None

        if current_price is not None:

            distance = (
                max_pain
                - current_price
            )

            distance_percent = (
                safe_divide(
                    distance,
                    current_price,
                )
                * 100
            )

        # ----------------------------------------------------
        # Add distance from MaxPain
        # ----------------------------------------------------

        table["distance_from_max_pain"] = (
            table["strike"]
            - max_pain
        )

        table["distance_percent"] = (
            table[
                "distance_from_max_pain"
            ]
            / max_pain
            * 100
        )

        # ----------------------------------------------------
        # Sort by total loss
        # ----------------------------------------------------

        ranking = (
            table.sort_values(
                "total_loss"
            )
            .reset_index(
                drop=True
            )
        )

        # ----------------------------------------------------
        # Convert to records
        # ----------------------------------------------------

        records = (
            ranking.to_dict(
                orient="records"
            )
        )

        return {
            "success": True,
            "max_pain": max_pain,
            "current_price": current_price,
            "distance": distance,
            "distance_percent": distance_percent,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "total_oi": total_oi,
            "table": records,
            "error": None,
        }

    # ========================================================
    # NEARBY LEVELS
    # ========================================================

    def nearby_levels(
        self,
        result: dict[str, Any],
        count: int = 5,
    ) -> list[dict[str, Any]]:

        table = result.get(
            "table",
            [],
        )

        if not table:

            return []

        current_price = result.get(
            "current_price"
        )

        if current_price is None:

            return table[
                :count
            ]

        ranked = sorted(
            table,
            key=lambda row:
            abs(
                row["strike"]
                - current_price
            ),
        )

        return ranked[
            :count
        ]


# ============================================================
# REPORT
# ============================================================

def print_report(
    result: dict[str, Any],
    calculator: MaxPainCalculator,
) -> None:

    print()
    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        "🎯 MAX PAIN ANALYSIS"
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    if not result["success"]:

        print(
            f"❌ Failed: "
            f"{result['error']}"
        )

        print(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        return

    current_price = result[
        "current_price"
    ]

    max_pain = result[
        "max_pain"
    ]

    distance = result[
        "distance"
    ]

    distance_percent = result[
        "distance_percent"
    ]

    print(
        f"💰 Current Price : "
        f"{format_price(current_price)}"
    )

    print(
        f"🎯 Max Pain      : "
        f"{format_price(max_pain)}"
    )

    if distance is not None:

        print(
            f"📏 Difference    : "
            f"{distance:+.2f}"
        )

    if distance_percent is not None:

        print(
            f"📊 Distance      : "
            f"{distance_percent:+.2f}%"
        )

    print()

    print(
        "📈 OPEN INTEREST"
    )

    print(
        f"CALL OI : "
        f"{format_number(result['call_oi'])}"
    )

    print(
        f"PUT OI  : "
        f"{format_number(result['put_oi'])}"
    )

    print(
        f"TOTAL OI: "
        f"{format_number(result['total_oi'])}"
    )

    print()

    print(
        "🎯 MAX PAIN RANKING"
    )

    print(
        "-" * 70
    )

    for index, row in enumerate(
        result["table"][:10],
        start=1,
    ):

        marker = (
            "🔥"
            if row["strike"]
            == max_pain
            else "  "
        )

        print(
            f"{marker} "
            f"{index:02d}. "
            f"${row['strike']:,.2f}"
            f" | "
            f"CALL LOSS "
            f"{format_money(row['call_loss'])}"
            f" | "
            f"PUT LOSS "
            f"{format_money(row['put_loss'])}"
            f" | "
            f"TOTAL "
            f"{format_money(row['total_loss'])}"
        )

    print()

    print(
        "📍 NEAR CURRENT PRICE"
    )

    print(
        "-" * 70
    )

    nearby = (
        calculator.nearby_levels(
            result,
            count=7,
        )
    )

    for row in nearby:

        marker = ""

        if (
            row["strike"]
            == max_pain
        ):

            marker = " 🎯 MAX PAIN"

        print(
            f"{format_price(row['strike'])}"
            f" | Total Loss "
            f"{format_money(row['total_loss'])}"
            f"{marker}"
        )

    print()

    print(
        "⚠️ Max Pain is an OI-based structural "
        "indicator, not a guaranteed settlement-price prediction."
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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

    # --------------------------------------------------------
    # Collector
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Raw DataFrame
    # --------------------------------------------------------

    raw_df = pd.DataFrame(
        raw_result["data"]
    )

    # --------------------------------------------------------
    # Normalizer
    # --------------------------------------------------------

    normalized_df, quality = (
        normalize_options(
            raw_df
        )
    )

    # --------------------------------------------------------
    # MaxPain
    # --------------------------------------------------------

    calculator = (
        MaxPainCalculator(
            normalized_df
        )
    )

    result = (
        calculator.calculate(
            current_price=raw_result[
                "current_price"
            ]
        )
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    result[
        "symbol"
    ] = symbol

    result[
        "expiration"
    ] = raw_result[
        "expiration"
    ]

    result[
        "DTE"
    ] = raw_result[
        "DTE"
    ]

    result[
        "quality"
    ] = quality

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_report(
        result,
        calculator,
    )


if __name__ == "__main__":

    main()
