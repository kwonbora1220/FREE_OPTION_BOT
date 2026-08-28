"""
FREE OPTION BOT
GEX ANALYSIS

무료 옵션체인 데이터만 사용한다.

INPUT
    symbol
    strike
    option_type
    openInterest
    gamma
    underlying_price

OUTPUT
    Call GEX
    Put GEX
    Net GEX
    GEX Flip
    Positive / Negative GEX zones
    Top GEX strikes

주의:
GEX는 딜러 포지션을 직접 관측한 값이 아니다.
무료 옵션체인의 Gamma/OI를 이용한 구조적 추정치다.

따라서:
    GEX != 실제 기관 포지션

특히 PUT GEX의 부호는 시장에서 사용하는
dealer-position convention에 따라 달라질 수 있다.

본 모듈에서는:
    CALL GEX = +Gamma × OI × multiplier × Spot² × 0.01
    PUT GEX  = -Gamma × OI × multiplier × Spot² × 0.01

을 사용한다.

이 방식은 Strike별 상대적인 GEX 구조를 보기 위한
단순하고 일관된 모델이다.
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

# Spot gamma를 dollar-gamma 형태로 변환할 때 사용하는
# 1% move normalization.
GEX_PERCENT_MOVE = 0.01


# ============================================================
# FORMAT HELPERS
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

    sign = "+" if value > 0 else ""

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

    return f"{sign}{value:.0f}"


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

        required_columns = [
            "option_type",
            "strike",
            "openInterest",
            "gamma",
        ]

        for column in required_columns:

            if column not in self.df.columns:

                self.df[column] = 0

        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Clean
        # ----------------------------------------------------

        self.df = self.df[
            self.df["strike"].notna()
        ].copy()

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
            self.df["option_type"].isin(
                [
                    "CALL",
                    "PUT",
                ]
            )
        ].copy()

    # ========================================================
    # DETERMINE SPOT
    # ========================================================

    def _get_spot(self) -> float | None:

        if (
            self.current_price is not None
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
    # CALCULATE GEX
    # ========================================================

    def calculate(
        self,
    ) -> dict[str, Any]:

        spot = self._get_spot()

        if self.df.empty:

            return {
                "success": False,
                "error": "Empty option data.",
                "current_price": spot,
                "table": [],
            }

        if spot is None:

            return {
                "success": False,
                "error": "Current price unavailable.",
                "current_price": None,
                "table": [],
            }

        # ----------------------------------------------------
        # Base GEX
        # ----------------------------------------------------

        self.df["gex_base"] = (
            self.df["gamma"]
            * self.df["openInterest"]
            * CONTRACT_MULTIPLIER
            * (spot ** 2)
            * GEX_PERCENT_MOVE
        )

        # ----------------------------------------------------
        # Call / Put GEX
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Aggregate by Strike
        # ----------------------------------------------------

        result = (
            self.df.groupby(
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

        result = result.sort_values(
            "strike"
        ).reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # Distance from spot
        # ----------------------------------------------------

        result["distance"] = (
            result["strike"]
            - spot
        )

        result["distance_percent"] = (
            result["distance"]
            / spot
            * 100
        )

        # ----------------------------------------------------
        # Cumulative GEX
        # ----------------------------------------------------

        result["cumulative_gex"] = (
            result["net_gex"].cumsum()
        )

        # ----------------------------------------------------
        # Positive / Negative
        # ----------------------------------------------------

        result["gex_sign"] = (
            result["net_gex"]
            .apply(
                lambda x:
                "POSITIVE"
                if x > 0
                else (
                    "NEGATIVE"
                    if x < 0
                    else "ZERO"
                )
            )
        )

        # ----------------------------------------------------
        # Total
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
        # Max Positive / Negative
        # ----------------------------------------------------

        max_positive = None
        max_negative = None

        positive_rows = result[
            result["net_gex"] > 0
        ]

        negative_rows = result[
            result["net_gex"] < 0
        ]

        if not positive_rows.empty:

            row = positive_rows.loc[
                positive_rows[
                    "net_gex"
                ].idxmax()
            ]

            max_positive = {
                "strike": float(
                    row["strike"]
                ),
                "net_gex": float(
                    row["net_gex"]
                ),
            }

        if not negative_rows.empty:

            row = negative_rows.loc[
                negative_rows[
                    "net_gex"
                ].idxmin()
            ]

            max_negative = {
                "strike": float(
                    row["strike"]
                ),
                "net_gex": float(
                    row["net_gex"]
                ),
            }

        # ----------------------------------------------------
        # GEX Flip
        # ----------------------------------------------------

        flip = self._find_gex_flip(
            result
        )

        # ----------------------------------------------------
        # Top absolute GEX
        # ----------------------------------------------------

        top_absolute = (
            result.assign(
                abs_gex=
                result["net_gex"].abs()
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

        # ----------------------------------------------------
        # Top positive
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Top negative
        # ----------------------------------------------------

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
            "current_price": spot,
            "total_call_gex": total_call_gex,
            "total_put_gex": total_put_gex,
            "total_net_gex": total_net_gex,
            "max_positive": max_positive,
            "max_negative": max_negative,
            "gex_flip": flip,
            "table": result.to_dict(
                orient="records"
            ),
            "top_absolute": top_absolute.to_dict(
                orient="records"
            ),
            "top_positive": top_positive.to_dict(
                orient="records"
            ),
            "top_negative": top_negative.to_dict(
                orient="records"
            ),
            "error": None,
        }

    # ========================================================
    # GEX FLIP
    # ========================================================

    def _find_gex_flip(
        self,
        result: pd.DataFrame,
    ) -> dict[str, Any] | None:

        if result.empty:
            return None

        previous_row = None

        for _, row in result.iterrows():

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
            # Exact zero
            # ------------------------------------------------

            if current_gex == 0:

                return {
                    "strike": float(
                        row["strike"]
                    ),
                    "lower_strike": float(
                        previous_row["strike"]
                    ),
                    "upper_strike": float(
                        row["strike"]
                    ),
                    "method": "ZERO",
                }

            # ------------------------------------------------
            # Sign change
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

                # Linear interpolation.
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
                    "strike": float(
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
            [],
        )

        spot = result.get(
            "current_price"
        )

        if not table or spot is None:

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

    # --------------------------------------------------------
    # TOTAL GEX
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FLIP
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EXTREMES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TOP ABSOLUTE
    # --------------------------------------------------------

    print()

    print(
        "🎯 TOP GEX STRIKES"
    )

    print(
        "-" * 70
    )

    for index, row in enumerate(
        result["top_absolute"],
        start=1,
    ):

        net = row[
            "net_gex"
        ]

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
            f" | "
            f"CALL "
            f"{format_gex(row['call_gex'])}"
            f" | "
            f"PUT "
            f"{format_gex(row['put_gex'])}"
            f" | "
            f"NET "
            f"{format_gex(net)}"
        )

    # --------------------------------------------------------
    # NEAR SPOT
    # --------------------------------------------------------

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

        net = row[
            "net_gex"
        ]

        if net > 0:

            marker = "🟢"

        elif net < 0:

            marker = "🔴"

        else:

            marker = "🟡"

        print(
            f"{marker} "
            f"{format_price(row['strike'])}"
            f" | "
            f"Net GEX "
            f"{format_gex(net)}"
        )

    print()

    print(
        "⚠️ GEX is a structural estimate based on "
        "Gamma × Open Interest."
    )

    print(
        "⚠️ It does not directly reveal dealer or institutional positions."
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
    # DataFrame
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
    # GEX
    # --------------------------------------------------------

    calculator = GEXCalculator(
        normalized_df,
        current_price=raw_result[
            "current_price"
        ],
    )

    result = calculator.calculate()

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
