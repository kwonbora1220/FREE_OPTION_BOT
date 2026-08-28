"""
FREE OPTION BOT - GEX ANALYSIS

Gamma × Open Interest 기반 구조적 GEX 분석.

CALL GEX:
    Gamma × Open Interest × 100 × Spot² × 0.01

PUT GEX:
    -Gamma × Open Interest × 100 × Spot² × 0.01

주의:
- GEX는 실제 딜러/기관 포지션을 직접 관측하지 않는다.
- 무료 데이터 기반 구조적 추정치다.
- 0DTE에서는 Gamma가 급격하게 변할 수 있다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import DEFAULT_SYMBOL
from option_collector import OptionCollector
from normalizer import normalize_options, print_normalizer_debug


# ============================================================
# CONFIG
# ============================================================

CONTRACT_MULTIPLIER = 100
GEX_PERCENT_MOVE = 0.01
NEAR_SPOT_COUNT = 9


# ============================================================
# FORMATTERS
# ============================================================

def format_price(value: float | None) -> str:
    if value is None:
        return "N/A"

    return f"${value:,.2f}"


def format_gex(value: float) -> str:
    sign = "+" if value > 0 else ""

    absolute = abs(value)

    if absolute >= 1_000_000_000:
        return f"{sign}{value / 1_000_000_000:.2f}B"

    if absolute >= 1_000_000:
        return f"{sign}{value / 1_000_000:.2f}M"

    if absolute >= 1_000:
        return f"{sign}{value / 1_000:.2f}K"

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
        self.current_price = current_price

        self._prepare()

    # ========================================================
    # PREPARE
    # ========================================================

    def _prepare(self):

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
                self.df[column] = 0.0

        # ----------------------------------------------------
        # NUMERIC
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
        # CLEAN
        # ----------------------------------------------------

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
                ["CALL", "PUT"]
            )
        ].copy()

    # ========================================================
    # SPOT
    # ========================================================

    def _get_spot(self):

        if (
            self.current_price is not None
            and self.current_price > 0
        ):
            return float(self.current_price)

        if "underlying_price" in self.df.columns:

            prices = pd.to_numeric(
                self.df["underlying_price"],
                errors="coerce",
            ).dropna()

            prices = prices[
                prices > 0
            ]

            if not prices.empty:
                return float(prices.iloc[0])

        return None

    # ========================================================
    # CALCULATE
    # ========================================================

    def calculate(self) -> dict[str, Any]:

        spot = self._get_spot()

        # ----------------------------------------------------
        # EMPTY DATA
        # ----------------------------------------------------

        if self.df.empty:

            return {
                "success": False,
                "error": "Empty option data.",
                "current_price": spot,
                "table": [],
            }

        # ----------------------------------------------------
        # NO SPOT
        # ----------------------------------------------------

        if spot is None:

            return {
                "success": False,
                "error": "Current price unavailable.",
                "current_price": None,
                "table": [],
            }

        # ====================================================
        # DATA QUALITY
        # ====================================================

        total_rows = len(self.df)

        gamma_nonzero = int(
            (
                self.df["gamma"] > 0
            ).sum()
        )

        gamma_ratio = (
            gamma_nonzero
            / total_rows
            * 100
            if total_rows > 0
            else 0.0
        )

        # ----------------------------------------------------
        # GAMMA UNAVAILABLE
        # ----------------------------------------------------

        if gamma_nonzero == 0:

            return {
                "success": False,
                "error":
                    "Gamma data is unavailable "
                    "or all gamma values are zero.",

                "current_price": spot,

                "table": [],

                "gamma_available": False,

                "gamma_nonzero_rows":
                    gamma_nonzero,

                "total_rows":
                    total_rows,

                "strike_rows":
                    int(
                        self.df["strike"].nunique()
                    ),

                "gamma_ratio":
                    gamma_ratio,
            }

        # ====================================================
        # GEX BASE
        # ====================================================

        self.df["gex_base"] = (

            self.df["gamma"]

            * self.df["openInterest"]

            * CONTRACT_MULTIPLIER

            * (spot ** 2)

            * GEX_PERCENT_MOVE

        )

        # ====================================================
        # CALL / PUT GEX
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
            "call_gex"
        ] = self.df.loc[
            call_mask,
            "gex_base"
        ]

        self.df.loc[
            put_mask,
            "put_gex"
        ] = -self.df.loc[
            put_mask,
            "gex_base"
        ]

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
            .sort_values("strike")
            .reset_index(drop=True)
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
            * 100.0
        )

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
        # TOTAL GEX
        # ====================================================

        total_call_gex = float(
            result["call_gex"].sum()
        )

        total_put_gex = float(
            result["put_gex"].sum()
        )

        total_net_gex = float(
            result["net_gex"].sum()
        )

        # ====================================================
        # POSITIVE / NEGATIVE
        # ====================================================

        positive = result[
            result["net_gex"] > 0
        ].copy()

        negative = result[
            result["net_gex"] < 0
        ].copy()

        # ====================================================
        # MAX POSITIVE
        # ====================================================

        max_positive = None

        if not positive.empty:

            row = positive.loc[
                positive["net_gex"].idxmax()
            ]

            max_positive = {

                "strike":
                    float(row["strike"]),

                "net_gex":
                    float(row["net_gex"]),
            }

        # ====================================================
        # MAX NEGATIVE
        # ====================================================

        max_negative = None

        if not negative.empty:

            row = negative.loc[
                negative["net_gex"].idxmin()
            ]

            max_negative = {

                "strike":
                    float(row["strike"]),

                "net_gex":
                    float(row["net_gex"]),
            }

        # ====================================================
        # GEX FLIP
        # ====================================================

        flip = self._find_gex_flip(
            result
        )

        # ====================================================
        # TOP ABSOLUTE GEX
        # ====================================================

        top_absolute = (

            result
            .assign(
                abs_gex=
                    result["net_gex"].abs()
            )
            .sort_values(
                "abs_gex",
                ascending=False,
            )
            .head(10)
            .drop(
                columns=["abs_gex"]
            )
        )

        # ====================================================
        # TOP POSITIVE
        # ====================================================

        top_positive = (

            positive
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

            negative
            .sort_values(
                "net_gex",
                ascending=True,
            )
            .head(5)
        )

        # ====================================================
        # RETURN
        # ====================================================

        return {

            "success": True,

            "current_price":
                spot,

            # DATA QUALITY
            "gamma_available":
                True,

            "total_rows":
                total_rows,

            "gamma_nonzero_rows":
                gamma_nonzero,

            "strike_rows":
                int(
                    result["strike"].nunique()
                ),

            "gamma_ratio":
                gamma_ratio,

            # GEX
            "total_call_gex":
                total_call_gex,

            "total_put_gex":
                total_put_gex,

            "total_net_gex":
                total_net_gex,

            # EXTREMES
            "max_positive":
                max_positive,

            "max_negative":
                max_negative,

            # FLIP
            "gex_flip":
                flip,

            # TABLE
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
    # GEX FLIP
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

            crossed = (

                (
                    prev_gex < 0
                    and curr_gex > 0
                )

                or

                (
                    prev_gex > 0
                    and curr_gex < 0
                )
            )

            if crossed:

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

    @staticmethod
    def near_spot(
        result: dict[str, Any],
        count: int = NEAR_SPOT_COUNT,
    ):

        table = result.get(
            "table",
            []
        )

        spot = result.get(
            "current_price"
        )

        if not table or spot is None:
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
    result: dict[str, Any]
) -> None:

    print()

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print("⚡ GEX ANALYSIS")

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    # ========================================================
    # FAILURE
    # ========================================================

    if not result["success"]:

        print(
            f"❌ Failed: "
            f"{result['error']}"
        )

        print(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        return

    # ========================================================
    # BASIC
    # ========================================================

    spot = result[
        "current_price"
    ]

    net = result[
        "total_net_gex"
    ]

    if net > 0:

        structure = (
            "🟢 POSITIVE GEX"
        )

    elif net < 0:

        structure = (
            "🔴 NEGATIVE GEX"
        )

    else:

        structure = (
            "🟡 NEUTRAL GEX"
        )

    print(
        f"💰 Current Price : "
        f"{format_price(spot)}"
    )

    print()

    # ========================================================
    # TOTAL GEX
    # ========================================================

    print("⚡ TOTAL GEX")

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
        f"{format_gex(net)}"
    )

    print(
        f"STRUCTURE: "
        f"{structure}"
    )

    print()

    # ========================================================
    # FLIP
    # ========================================================

    print("🔄 GEX FLIP")

    flip = result.get(
        "gex_flip"
    )

    if flip:

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

    else:

        print(
            "GEX Flip : N/A"
        )

    print()

    # ========================================================
    # EXTREMES
    # ========================================================

    print("🔥 GEX EXTREMES")

    pos = result.get(
        "max_positive"
    )

    neg = result.get(
        "max_negative"
    )

    if pos:

        print(
            f"🟢 Max Positive : "
            f"{format_price(pos['strike'])}"
            f" "
            f"({format_gex(pos['net_gex'])})"
        )

    else:

        print(
            "🟢 Max Positive : N/A"
        )

    if neg:

        print(
            f"🔴 Max Negative : "
            f"{format_price(neg['strike'])}"
            f" "
            f"({format_gex(neg['net_gex'])})"
        )

    else:

        print(
            "🔴 Max Negative : N/A"
        )

    print()

    # ========================================================
    # TOP GEX
    # ========================================================

    print(
        "🎯 TOP GEX STRIKES"
    )

    print(
        "----------------------------------------------------------------------"
    )

    for i, row in enumerate(
        result["top_absolute"],
        1,
    ):

        if row["net_gex"] > 0:

            emoji = "🟢"

        elif row["net_gex"] < 0:

            emoji = "🔴"

        else:

            emoji = "🟡"

        print(

            f"{emoji} "

            f"{i:02d}. "

            f"{format_price(row['strike'])}"

            f" | CALL "
            f"{format_gex(row['call_gex'])}"

            f" | PUT "
            f"{format_gex(row['put_gex'])}"

            f" | NET "
            f"{format_gex(row['net_gex'])}"
        )

    print()

    # ========================================================
    # NEAR CURRENT PRICE
    # ========================================================

    print(
        "📍 NEAR CURRENT PRICE"
    )

    print(
        "----------------------------------------------------------------------"
    )

    near_rows = (
        GEXCalculator.near_spot(
            result,
            NEAR_SPOT_COUNT,
        )
    )

    for row in near_rows:

        if row["net_gex"] > 0:

            emoji = "🟢"

        elif row["net_gex"] < 0:

            emoji = "🔴"

        else:

            emoji = "🟡"

        print(

            f"{emoji} "

            f"{format_price(row['strike'])}"

            f" | Net GEX "

            f"{format_gex(row['net_gex'])}"
        )

    print()

    # ========================================================
    # DATA QUALITY
    # ========================================================

    print(
        "📊 DATA QUALITY"
    )

    print(
        "----------------------------------------------------------------------"
    )

    print(
        f"Option rows : "
        f"{result['total_rows']:,}"
    )

    print(
        f"Gamma rows  : "
        f"{result['gamma_nonzero_rows']:,}"
    )

    print(
        f"Strike rows : "
        f"{result['strike_rows']:,}"
    )

    print(
        f"Gamma ratio : "
        f"{result['gamma_ratio']:.1f}%"
    )

    print()

    # ========================================================
    # WARNINGS
    # ========================================================

    print(
        "⚠️ GEX is a structural estimate "
        "based on model-derived Gamma × Open Interest."
    )

    print(
        "⚠️ It does not directly reveal "
        "dealer or institutional positions."
    )

    # 0DTE WARNING
    if "DTE" in result.get(
        "table",
        [{}]
    )[0]:

        dtes = [
            row.get("DTE")
            for row in result["table"]
            if row.get("DTE") is not None
        ]

        if dtes:

            try:

                min_dte = min(
                    float(x)
                    for x in dtes
                )

                if min_dte <= 0:

                    print(
                        "⚠️ 0DTE GEX can change "
                        "rapidly near expiration."
                    )

            except Exception:
                pass

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


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

    data = collector.collect()

    if (
        data is None
        or data.empty
    ):

        print(
            "❌ No option data collected."
        )

        return

    # ========================================================
    # NORMALIZER
    # ========================================================

    normalized = normalize_options(
        data
    )

    # ========================================================
    # NORMALIZER DEBUG
    # ========================================================

    print_normalizer_debug(
        normalized
    )

    # ========================================================
    # GEX
    # ========================================================

    calculator = GEXCalculator(

        normalized,

        current_price=
            collector.current_price,
    )

    result = calculator.calculate()

    # ========================================================
    # REPORT
    # ========================================================

    print_report(
        result
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
