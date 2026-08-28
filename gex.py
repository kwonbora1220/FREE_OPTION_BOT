"""
FREE OPTION BOT
GEX ANALYSIS

Gamma × Open Interest 기반 구조적 GEX.

CALL:
    Gamma × OI × 100 × Spot² × 0.01

PUT:
    -Gamma × OI × 100 × Spot² × 0.01

주의:
GEX는 실제 딜러 포지션을 직접 관측하는 값이 아니다.
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

    def _prepare(self):

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

                self.df[column] = 0.0

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

    def _get_spot(self):

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
                    (
                        "Gamma data is "
                        "unavailable or all "
                        "gamma values are zero."
                    ),
                "current_price":
                    spot,
                "table": [],
                "gamma_available":
                    False,
            }

        # ----------------------------------------------------
        # GEX
        # ----------------------------------------------------

        self.df["gex_base"] = (
            self.df["gamma"]
            * self.df["openInterest"]
            * CONTRACT_MULTIPLIER
            * (
                spot ** 2
            )
            * GEX_PERCENT_MOVE
        )

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
        ] = -self.df.loc[
            put_mask,
            "gex_base",
        ]

        # ----------------------------------------------------
        # STRIKE AGGREGATION
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # DISTANCE
        # ----------------------------------------------------

        result["distance"] = (
            result["strike"]
            - spot
        )

        result["distance_percent"] = (
            result["distance"]
            / spot
            * 100.0
        )

        # ----------------------------------------------------
        # SIGN
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # EXTREMES
        # ----------------------------------------------------

        positive = result[
            result["net_gex"] > 0
        ]

        negative = result[
            result["net_gex"] < 0
        ]

        max_positive = None
        max_negative = None

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

        # ----------------------------------------------------
        # FLIP
        # ----------------------------------------------------

        flip = (
            self._find_gex_flip(
                result
            )
        )

        # ----------------------------------------------------
        # TOP
        # ----------------------------------------------------

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

            "error": None,
        }

    # ========================================================
    # FLIP
    # ========================================================

    def _find_gex_flip(
        self,
        result: pd.DataFrame,
    ):

        nonzero = result[
            result["net_gex"] != 0
        ].copy()

        if len(nonzero) < 2:

            return None

        previous = None

        for _, row in nonzero.iterrows():

            if previous is None:

                previous = row

                continue

            prev_gex = float(
                previous["net_gex"]
            )

            curr_gex = float(
                row["net_gex"]
            )

            if (
                prev_gex < 0
                and curr_gex > 0
            ) or (
                prev_gex > 0
                and curr_gex < 0
            ):

                lower = float(
                    previous["strike"]
                )

                upper = float(
                    row["strike"]
                )

                denominator = (
                    curr_gex
                    - prev_gex
                )

                if denominator != 0:

                    flip = (
                        lower
                        + (
                            -prev_gex
                            / denominator
                        )
                        * (
                            upper
                            - lower
                        )
                    )

                else:

                    flip = (
                        lower
                        + upper
                    ) / 2.0

                return {
                    "strike":
                        float(flip),

                    "lower_strike":
                        lower,

                    "upper_strike":
                        upper,

                    "method":
                        "INTERPOLATED",
                }

            previous = row

        return None

    # ========================================================
    # NEAR SPOT
    # ========================================================

    def near_spot(
        self,
        result: dict[str, Any],
        count: int = 9,
    ):

        table = result.get(
            "table",
            [],
        )

        spot = result.get(
            "current_price"
        )

        if (
            not table
            or spot is None
        ):

            return []

        return sorted(
            table,
            key=lambda row:
                abs(
                    row["strike"]
                    - spot
                ),
        )[:count]


# ============================================================
# REPORT
# ============================================================

def print_report(
    result: dict[str, Any],
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

    if result[
        "total_net_gex"
    ] > 0:

        print(
            "STRUCTURE: 🟢 POSITIVE GEX"
        )

    elif result[
        "total_net_gex"
    ] < 0:

        print(
            "STRUCTURE: 🔴 NEGATIVE GEX"
        )

    else:

        print(
            "STRUCTURE: 🟡 NEUTRAL GEX"
        )

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
            f"{format_price(positive['strike'])} "
            f"({format_gex(positive['net_gex'])})"
        )

    else:

        print(
            "🟢 Max Positive : N/A"
        )

    if negative:

        print(
            f"🔴 Max Negative : "
            f"{format_price(negative['strike'])} "
            f"({format_gex(negative['net_gex'])})"
        )

    else:

        print(
            "🔴 Max Negative : N/A"
        )

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

        marker = (
            "🟢"
            if net > 0
            else (
                "🔴"
                if net < 0
                else "🟡"
            )
        )

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

    print()

    print(
        "📍 NEAR CURRENT PRICE"
    )

    print(
        "-" * 70
    )

    near = GEXCalculator(
        pd.DataFrame(
            result["table"]
        ),
        spot,
    ).near_spot(
        result,
        9,
    )

    for row in near:

        net = float(
            row["net_gex"]
        )

        marker = (
            "🟢"
            if net > 0
            else (
                "🔴"
                if net < 0
                else "🟡"
            )
        )

        print(
            f"{marker} "
            f"${row['strike']:,.2f}"
            f" | Net GEX "
            f"{format_gex(net)}"
        )

    print()

    print(
        "⚠️ GEX is a structural estimate "
        "based on model-derived Gamma × Open Interest."
    )

    print(
        "⚠️ It does not directly reveal "
        "dealer or institutional positions."
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Collecting "
        f"{DEFAULT_SYMBOL}..."
    )

    collector = OptionCollector(
        DEFAULT_SYMBOL
    )

    current_price = (
        collector.get_current_price()
    )

    expiration = (
        collector.get_nearest_expiration(
            1
        )
    )

    if expiration is None:

        print(
            "❌ No suitable expiration found."
        )

        return

    raw = (
        collector.fetch_chain(
            expiration
        )
    )

    df = normalize_options(
        raw,
        current_price,
    )

    print_normalizer_debug(
        df
    )

    print()

    calculator = GEXCalculator(
        df,
        current_price,
    )

    result = (
        calculator.calculate()
    )

    print_report(
        result
    )


if __name__ == "__main__":

    main()
