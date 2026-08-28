"""
FREE OPTION BOT
GEX ANALYSIS

무료 옵션체인의 Gamma + Open Interest 기반
구조적 GEX를 계산한다.

GEX는 실제 딜러 포지션을 직접 관측한 값이 아니다.

모델:

CALL GEX
= Gamma × OI × 100 × Spot² × 0.01

PUT GEX
= -Gamma × OI × 100 × Spot² × 0.01

목적:
- Strike별 GEX
- Net GEX
- GEX Flip
- Positive / Negative GEX
- 주요 GEX Strike
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import DEFAULT_SYMBOL
from option_collector import OptionCollector
from normalizer import (
    normalize_options,
    print_normalizer_debug,
)


# ============================================================
# CONSTANTS
# ============================================================

CONTRACT_MULTIPLIER = 100

GEX_PERCENT_MOVE = 0.01


# ============================================================
# FORMAT
# ============================================================

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


def format_gex(
    value: float,
) -> str:

    sign = (
        "+"
        if value > 0
        else ""
    )

    absolute = abs(value)

    if absolute >= 1_000_000_000:

        return (
            f"{sign}"
            f"{value / 1_000_000_000:.2f}B"
        )

    if absolute >= 1_000_000:

        return (
            f"{sign}"
            f"{value / 1_000_000:.2f}M"
        )

    if absolute >= 1_000:

        return (
            f"{sign}"
            f"{value / 1_000:.2f}K"
        )

    return (
        f"{sign}"
        f"{value:.0f}"
    )


# ============================================================
# GEX CALCULATOR
# ============================================================

class GEXCalculator:

    def __init__(
        self,
        df: pd.DataFrame,
        current_price: float | None = None,
    ):

        self.df = df.copy()

        self.current_price = (
            current_price
        )

        self._prepare()

    # ========================================================
    # PREPARE
    # ========================================================

    def _prepare(self) -> None:

        if self.df.empty:

            return

        required = [
            "option_type",
            "strike",
            "openInterest",
            "gamma",
        ]

        for column in required:

            if column not in self.df.columns:

                self.df[column] = 0

        self.df["strike"] = pd.to_numeric(
            self.df["strike"],
            errors="coerce",
        )

        self.df["openInterest"] = pd.to_numeric(
            self.df["openInterest"],
            errors="coerce",
        )

        self.df["gamma"] = pd.to_numeric(
            self.df["gamma"],
            errors="coerce",
        )

        self.df["openInterest"] = (
            self.df["openInterest"]
            .fillna(0)
            .clip(lower=0)
        )

        self.df["gamma"] = (
            self.df["gamma"]
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
            self.df["strike"].notna()
        ].copy()

        self.df = self.df[
            self.df["option_type"].isin(
                [
                    "CALL",
                    "PUT",
                ]
            )
        ].copy()

    # ========================================================
    # SPOT
    # ========================================================

    def _get_spot(
        self,
    ) -> float | None:

        if (
            self.current_price
            is not None
            and self.current_price > 0
        ):

            return float(
                self.current_price
            )

        if (
            "underlying_price"
            in self.df.columns
        ):

            prices = pd.to_numeric(
                self.df[
                    "underlying_price"
                ],
                errors="coerce",
            ).dropna()

            prices = prices[
                prices > 0
            ]

            if not prices.empty:

                return float(
                    prices.iloc[0]
                )

        return None

    # ========================================================
    # CALCULATE
    # ========================================================

    def calculate(
        self,
    ) -> dict[str, Any]:

        spot = self._get_spot()

        if self.df.empty:

            return {
                "success": False,
                "error":
                    "Empty option data.",
                "current_price":
                    spot,
                "table": [],
            }

        if spot is None:

            return {
                "success": False,
                "error":
                    "Current price unavailable.",
                "current_price":
                    None,
                "table": [],
            }

        # ====================================================
        # VALIDATE GAMMA
        # ====================================================

        gamma_nonzero = int(
            (
                self.df["gamma"]
                > 0
            ).sum()
        )

        if gamma_nonzero == 0:

            return {
                "success": False,
                "error":
                    "Gamma data is unavailable or all gamma values are zero.",
                "current_price":
                    spot,
                "table": [],
                "gamma_available":
                    False,
            }

        # ====================================================
        # BASE GEX
        # ====================================================

        self.df["gex_base"] = (
            self.df["gamma"]
            * self.df["openInterest"]
            * CONTRACT_MULTIPLIER
            * (spot ** 2)
            * GEX_PERCENT_MOVE
        )

        # ====================================================
        # CALL / PUT
        # ====================================================

        self.df["call_gex"] = 0.0

        self.df["put_gex"] = 0.0

        call_mask = (
            self.df["option_type"]
            == "CALL"
        )

        put_mask = (
            self.df["option_type"]
            == "PUT"
        )

        self.df.loc[
            call_mask,
            "call_gex",
        ] = self.df.loc[
            call_mask,
            "gex_base",
        ]

        self.df.loc[
            put_mask,
            "put_gex",
        ] = (
            -self.df.loc[
                put_mask,
                "gex_base",
            ]
        )

        # ====================================================
        # STRIKE AGGREGATION
        # ====================================================

        result = (
            self.df
            .groupby(
                "strike",
                as_index=False,
            )[
                [
                    "call_gex",
                    "put_gex",
                ]
            ]
            .sum()
        )

        result["net_gex"] = (
            result["call_gex"]
            + result["put_gex"]
        )

        result = (
            result
            .sort_values(
                "strike"
            )
            .reset_index(
                drop=True
            )
        )

        # ====================================================
        # DISTANCE
        # ====================================================

        result["distance"] = (
            result["strike"]
            - spot
        )

        result["distance_percent"] = (
            result["distance"]
            / spot
            * 100
        )

        # ====================================================
        # SIGN
        # ====================================================

        result["gex_sign"] = (
            result["net_gex"]
            .apply(
                lambda value:
                "POSITIVE"
                if value > 0
                else (
                    "NEGATIVE"
                    if value < 0
                    else "ZERO"
                )
            )
        )

        # ====================================================
        # TOTAL
        # ====================================================

        total_call_gex = float(
            result[
                "call_gex"
            ].sum()
        )

        total_put_gex = float(
            result[
                "put_gex"
            ].sum()
        )

        total_net_gex = float(
            result[
                "net_gex"
            ].sum()
        )

        # ====================================================
        # EXTREMES
        # ====================================================

        max_positive = None

        max_negative = None

        positive = result[
            result["net_gex"] > 0
        ]

        negative = result[
            result["net_gex"] < 0
        ]

        if not positive.empty:

            row = positive.loc[
                positive[
                    "net_gex"
                ].idxmax()
            ]

            max_positive = {
                "strike":
                    float(
                        row["strike"]
                    ),
                "net_gex":
                    float(
                        row["net_gex"]
                    ),
            }

        if not negative.empty:

            row = negative.loc[
                negative[
                    "net_gex"
                ].idxmin()
            ]

            max_negative = {
                "strike":
                    float(
                        row["strike"]
                    ),
                "net_gex":
                    float(
                        row["net_gex"]
                    ),
            }

        # ====================================================
        # GEX FLIP
        # ====================================================

        flip = self._find_gex_flip(
            result
        )

        # ====================================================
        # TOP ABSOLUTE
        # ====================================================

        top_absolute = (
            result.assign(
                abs_gex=
                result[
                    "net_gex"
                ].abs()
            )
            .sort_values(
                "abs_gex",
                ascending=False,
            )
            .head(10)
            .drop(
                columns=[
                    "abs_gex"
                ]
            )
        )

        # ====================================================
        # TOP POSITIVE
        # ====================================================

        top_positive = (
            result[
                result["net_gex"] > 0
            ]
            .sort_values(
                "net_gex",
                ascending=False,
            )
            .head(5)
        )

        # ====================================================
        # TOP NEGATIVE
        # ====================================================

        top_negative = (
            result[
                result["net_gex"] < 0
            ]
            .sort_values(
                "net_gex",
                ascending=True,
            )
            .head(5)
        )

        return {
            "success": True,

            "current_price":
                spot,

            "gamma_available":
                True,

            "gamma_nonzero_rows":
                gamma_nonzero,

            "total_call_gex":
                total_call_gex,

            "total_put_gex":
                total_put_gex,

            "total_net_gex":
                total_net_gex,

            "max_positive":
                max_positive,

            "max_negative":
                max_negative,

            "gex_flip":
                flip,

            "table":
                result.to_dict(
                    orient="records"
                ),

            "top_absolute":
                top_absolute.to_dict(
                    orient="records"
                ),

            "top_positive":
                top_positive.to_dict(
                    orient="records"
                ),

            "top_negative":
                top_negative.to_dict(
                    orient="records"
                ),

            "error":
                None,
        }

    # ========================================================
    # FLIP
    # ========================================================

    def _find_gex_flip(
        self,
        result: pd.DataFrame,
    ) -> dict[str, Any] | None:

        if result.empty:

            return None

        # ----------------------------------------------------
        # IMPORTANT
        # Ignore zero GEX rows.
        #
        # We only want a genuine sign change:
        #
        # negative → positive
        # positive → negative
        # ----------------------------------------------------

        nonzero = result[
            result["net_gex"] != 0
        ].copy()

        if len(nonzero) < 2:

            return None

        previous_row = None

        for _, row in nonzero.iterrows():

            current_gex = float(
                row["net_gex"]
            )

            if previous_row is None:

                previous_row = row

                continue

            previous_gex = float(
                previous_row["net_gex"]
            )

            # ------------------------------------------------
            # SIGN CHANGE
            # ------------------------------------------------

            if (
                previous_gex < 0
                and current_gex > 0
            ) or (
                previous_gex > 0
                and current_gex < 0
            ):

                lower_strike = float(
                    previous_row["strike"]
                )

                upper_strike = float(
                    row["strike"]
                )

                denominator = (
                    current_gex
                    - previous_gex
                )

                if denominator != 0:

                    flip_strike = (
                        lower_strike
                        + (
                            -previous_gex
                            / denominator
                        )
                        * (
                            upper_strike
                            - lower_strike
                        )
                    )

                else:

                    flip_strike = (
                        lower_strike
                        + upper_strike
                    ) / 2

                return {
                    "strike":
                        float(
                            flip_strike
                        ),

                    "lower_strike":
                        lower_strike,

                    "upper_strike":
                        upper_strike,

                    "method":
                        "INTERPOLATED",
                }

            previous_row = row

        return None

    # ========================================================
    # NEAR SPOT
    # ========================================================

    def near_spot(
        self,
        result: dict[str, Any],
        count: int = 9,
    ) -> list[dict[str, Any]]:

        table = result.get(
            "table",
            []
        )

        spot = result.get(
            "current_price"
        )

        if (
            not table
            or spot is None
        ):

            return []

        rows = sorted(
            table,
            key=lambda row:
            abs(
                row["strike"]
                - spot
            ),
        )

        return rows[
            :count
        ]


# ============================================================
# REPORT
# ============================================================

def print_report(
    result: dict[str, Any],
    calculator: GEXCalculator,
) -> None:

    print()

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        "⚡ GEX ANALYSIS"
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

    spot = result[
        "current_price"
    ]

    print(
        f"💰 Current Price : "
        f"{format_price(spot)}"
    )

    print()

    print(
        "⚡ TOTAL GEX"
    )

    print(
        f"CALL GEX : "
        f"{format_gex(result['total_call_gex'])}"
    )

    print(
        f"PUT GEX  : "
        f"{format_gex(result['total_put_gex'])}"
    )

    print(
        f"NET GEX  : "
        f"{format_gex(result['total_net_gex'])}"
    )

    if result["total_net_gex"] > 0:

        print(
            "STRUCTURE: 🟢 POSITIVE GEX"
        )

    elif result["total_net_gex"] < 0:

        print(
            "STRUCTURE: 🔴 NEGATIVE GEX"
        )

    else:

        print(
            "STRUCTURE: 🟡 NEUTRAL GEX"
        )

    # ========================================================
    # FLIP
    # ========================================================

    print()

    print(
        "🔄 GEX FLIP"
    )

    flip = result[
        "gex_flip"
    ]

    if flip is None:

        print(
            "GEX Flip : N/A"
        )

    else:

        print(
            f"GEX Flip : "
            f"{format_price(flip['strike'])}"
        )

        print(
            f"Range    : "
            f"{format_price(flip['lower_strike'])}"
            f" ~ "
            f"{format_price(flip['upper_strike'])}"
        )

    # ========================================================
    # EXTREMES
    # ========================================================

    print()

    print(
        "🔥 GEX EXTREMES"
    )

    positive = result[
        "max_positive"
    ]

    negative = result[
        "max_negative"
    ]

    if positive:

        print(
            f"🟢 Max Positive : "
            f"{format_price(positive['strike'])}"
            f" "
            f"({format_gex(positive['net_gex'])})"
        )

    else:

        print(
            "🟢 Max Positive : N/A"
        )

    if negative:

        print(
            f"🔴 Max Negative : "
            f"{format_price(negative['strike'])}"
            f" "
            f"({format_gex(negative['net_gex'])})"
        )

    else:

        print(
            "🔴 Max Negative : N/A"
        )

    # ========================================================
    # TOP GEX
    # ========================================================

    print()

    print(
        "🎯 TOP GEX STRIKES"
    )

    print(
        "-" * 70
    )

    for index, row in enumerate(
        result[
            "top_absolute"
        ],
        start=1,
    ):

        net = float(
            row["net_gex"]
        )

        if net > 0:

            marker = "🟢"

        elif net < 0:

            marker = "🔴"

        else:

            marker = "🟡"

        print(
            f"{marker} "
            f"{index:02d}. "
            f"${row['strike']:,.2f}"
            f" | CALL "
            f"{format_gex(row['call_gex'])}"
            f" | PUT "
            f"{format_gex(row['put_gex'])}"
            f" | NET "
            f"{format_gex(net)}"
        )

    # ========================================================
    # NEAR SPOT
    # ========================================================

    print()

    print(
        "📍 NEAR CURRENT PRICE"
    )

    print(
        "-" * 70
    )

    near_rows = (
        calculator.near_spot(
            result,
            count=9,
        )
    )

    for row in sorted(
        near_rows,
        key=lambda x:
        x["strike"],
    ):

        net = float(
            row["net_gex"]
        )

        if net > 0:

            marker = "🟢"

        elif net < 0:

            marker = "🔴"

        else:

            marker = "🟡"

        print(
            f"{marker} "
            f"{format_price(row['strike'])}"
            f" | Net GEX "
            f"{format_gex(net)}"
        )

    print()

    print(
        "⚠️ GEX is a structural estimate based on "
        "Gamma × Open Interest."
    )

    print(
        "⚠️ It does not directly reveal dealer or "
        "institutional positions."
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

    # ========================================================
    # COLLECT
    # ========================================================

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

    # ========================================================
    # DATAFRAME
    # ========================================================

    raw_df = pd.DataFrame(
        raw_result["data"]
    )

    # ========================================================
    # NORMALIZE
    # ========================================================

    normalized_df, quality = (
        normalize_options(
            raw_df
        )
    )

    # ========================================================
    # DEBUG
    # ========================================================

    print_normalizer_debug(
        normalized_df,
        quality,
    )

    # ========================================================
    # GEX
    # ========================================================

    calculator = GEXCalculator(
        normalized_df,
        current_price=
            raw_result[
                "current_price"
            ],
    )

    result = (
        calculator.calculate()
    )

    # ========================================================
    # METADATA
    # ========================================================

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

    # ========================================================
    # REPORT
    # ========================================================

    print_report(
        result,
        calculator,
    )


if __name__ == "__main__":

    main()
