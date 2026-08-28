"""
FREE OPTION BOT
GEX ANALYSIS

Yahoo Finance 무료 옵션체인 + 자체 Black-Scholes Greeks 기반 GEX.

핵심:
- OptionCollector.collect_one_day() 사용
- 미국 동부시간 기준 만기 유지
- DTE=0 지원
- CALL GEX / PUT GEX / NET GEX
- 현재가 주변의 GEX Flip 우선 탐색
- GEX Extremes
- TOP GEX Strikes
- Current Price 주변 GEX
- Data Quality

주의:
GEX는 Gamma × Open Interest 기반의 구조적 추정치이며
실제 딜러/기관 포지션을 직접 보여주지 않는다.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from config import DEFAULT_SYMBOL
from option_collector import OptionCollector


# ============================================================
# CONFIG
# ============================================================

CONTRACT_MULTIPLIER = 100

# 전체 옵션체인을 GEX 계산에 사용
FILTER_TO_NEAR_SPOT = False

# FILTER_TO_NEAR_SPOT=True일 경우
SPOT_RANGE_PERCENT = 0.50

# ------------------------------------------------------------
# GEX FLIP CONFIG
# ------------------------------------------------------------

# GEX Flip을 찾을 때 현재가에서 너무 먼 strike를 제외한다.
#
# 예:
# 현재가 220
# FLIP_SEARCH_RANGE_PERCENT = 0.20
#
# => 176 ~ 264 범위에서만 Flip 탐색
#
# 전체 GEX 계산에는 영향을 주지 않는다.
FLIP_SEARCH_RANGE_PERCENT = 0.20

# 현재가 주변의 실제 strike 구조를 우선한다.
# True 권장.
USE_LOCAL_FLIP_SEARCH = True

# 현재가 주변 몇 개 strike를 Flip 후보로 사용할지
LOCAL_FLIP_STRIKE_COUNT = 21


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value: Any) -> float:

    try:

        if value is None:
            return 0.0

        value = float(value)

        if math.isnan(value):
            return 0.0

        if math.isinf(value):
            return 0.0

        return value

    except (
        TypeError,
        ValueError,
    ):

        return 0.0


# ============================================================
# FORMAT
# ============================================================

def format_money(value: float) -> str:

    value = safe_float(value)

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


def format_signed_money(value: float) -> str:

    return format_money(value)


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_gex_data(
    df: pd.DataFrame,
    current_price: float,
) -> pd.DataFrame:

    if (
        df is None
        or df.empty
    ):

        return pd.DataFrame()

    result = df.copy()

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required = [
        "option_type",
        "strike",
        "openInterest",
        "gamma",
    ]

    for column in required:

        if column not in result.columns:

            result[column] = 0.0

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    for column in [
        "strike",
        "openInterest",
        "gamma",
    ]:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).fillna(0.0)

    # --------------------------------------------------------
    # OPTION TYPE
    # --------------------------------------------------------

    result["option_type"] = (
        result["option_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # REMOVE INVALID
    # --------------------------------------------------------

    result = result[
        result["strike"] > 0
    ].copy()

    result = result[
        result["openInterest"] >= 0
    ].copy()

    result = result[
        result["gamma"] >= 0
    ].copy()

    # --------------------------------------------------------
    # OPTIONAL SPOT FILTER
    # --------------------------------------------------------

    if FILTER_TO_NEAR_SPOT:

        lower = (
            current_price
            * (
                1.0
                - SPOT_RANGE_PERCENT
            )
        )

        upper = (
            current_price
            * (
                1.0
                + SPOT_RANGE_PERCENT
            )
        )

        result = result[
            (
                result["strike"]
                >= lower
            )
            & (
                result["strike"]
                <= upper
            )
        ].copy()

    # --------------------------------------------------------
    # GEX
    #
    # Gamma × OI × 100 × Spot² × 0.01
    #
    # CALL = positive
    # PUT  = negative
    #
    # 실제 dealer positioning을 의미하지 않음.
    # --------------------------------------------------------

    result["gex"] = (
        result["gamma"]
        * result["openInterest"]
        * CONTRACT_MULTIPLIER
        * (
            current_price
            ** 2
        )
        * 0.01
    )

    result["gex"] = (
        result["gex"]
        .replace(
            [float("inf"), float("-inf")],
            0.0,
        )
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # SIGN
    # --------------------------------------------------------

    result["signed_gex"] = 0.0

    call_mask = (
        result["option_type"]
        == "CALL"
    )

    put_mask = (
        result["option_type"]
        == "PUT"
    )

    result.loc[
        call_mask,
        "signed_gex",
    ] = result.loc[
        call_mask,
        "gex",
    ]

    result.loc[
        put_mask,
        "signed_gex",
    ] = -result.loc[
        put_mask,
        "gex",
    ]

    return result.reset_index(
        drop=True
    )


# ============================================================
# AGGREGATE BY STRIKE
# ============================================================

def aggregate_by_strike(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if (
        df is None
        or df.empty
    ):

        return pd.DataFrame()

    calls = (
        df[
            df["option_type"]
            == "CALL"
        ]
        .groupby(
            "strike"
        )["gex"]
        .sum()
    )

    puts = (
        df[
            df["option_type"]
            == "PUT"
        ]
        .groupby(
            "strike"
        )["gex"]
        .sum()
    )

    strikes = sorted(
        set(calls.index)
        | set(puts.index)
    )

    rows = []

    for strike in strikes:

        call_gex = safe_float(
            calls.get(
                strike,
                0.0,
            )
        )

        put_gex_abs = safe_float(
            puts.get(
                strike,
                0.0,
            )
        )

        # PUT GEX는 표시할 때 음수
        put_gex = -put_gex_abs

        net_gex = (
            call_gex
            - put_gex_abs
        )

        rows.append(
            {
                "strike":
                    float(strike),

                "call_gex":
                    call_gex,

                "put_gex":
                    put_gex,

                "net_gex":
                    net_gex,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TOTAL GEX
# ============================================================

def calculate_total_gex(
    df: pd.DataFrame,
) -> dict[str, float]:

    if (
        df is None
        or df.empty
    ):

        return {
            "call_gex": 0.0,
            "put_gex": 0.0,
            "net_gex": 0.0,
        }

    calls = df[
        df["option_type"]
        == "CALL"
    ]

    puts = df[
        df["option_type"]
        == "PUT"
    ]

    call_gex = safe_float(
        calls["gex"].sum()
    )

    put_gex_abs = safe_float(
        puts["gex"].sum()
    )

    net_gex = (
        call_gex
        - put_gex_abs
    )

    return {
        "call_gex":
            call_gex,

        "put_gex":
            -put_gex_abs,

        "net_gex":
            net_gex,
    }


# ============================================================
# LOCAL FLIP DATA
# ============================================================

def get_flip_search_data(
    strike_df: pd.DataFrame,
    current_price: float,
) -> pd.DataFrame:

    if (
        strike_df is None
        or strike_df.empty
    ):

        return pd.DataFrame()

    data = (
        strike_df.copy()
        .sort_values("strike")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 1. 현재가 ±20% 범위
    # --------------------------------------------------------

    lower_price = (
        current_price
        * (
            1.0
            - FLIP_SEARCH_RANGE_PERCENT
        )
    )

    upper_price = (
        current_price
        * (
            1.0
            + FLIP_SEARCH_RANGE_PERCENT
        )
    )

    local = data[
        (
            data["strike"]
            >= lower_price
        )
        & (
            data["strike"]
            <= upper_price
        )
    ].copy()

    # --------------------------------------------------------
    # 2. 범위 안에 strike가 너무 적으면
    #    현재가와 가까운 strike를 사용
    # --------------------------------------------------------

    if len(local) < 2:

        local = (
            data.assign(
                distance=(
                    data["strike"]
                    - current_price
                ).abs()
            )
            .sort_values(
                "distance"
            )
            .head(
                LOCAL_FLIP_STRIKE_COUNT
            )
            .sort_values(
                "strike"
            )
            .drop(
                columns=["distance"]
            )
        )

    return local.reset_index(
        drop=True
    )


# ============================================================
# GEX FLIP
# ============================================================

def find_gex_flip(
    strike_df: pd.DataFrame,
    current_price: float,
) -> dict[str, Any]:

    empty_result = {
        "flip": None,
        "lower": None,
        "upper": None,
        "method": "NONE",
    }

    if (
        strike_df is None
        or strike_df.empty
    ):

        return empty_result

    # --------------------------------------------------------
    # IMPORTANT
    #
    # 전체 chain에서 가장 먼 sign change를 찾지 않는다.
    #
    # 현재가 주변에서 먼저 탐색한다.
    # --------------------------------------------------------

    if USE_LOCAL_FLIP_SEARCH:

        data = get_flip_search_data(
            strike_df,
            current_price,
        )

    else:

        data = (
            strike_df
            .sort_values("strike")
            .reset_index(drop=True)
        )

    if data.empty:

        return empty_result

    # --------------------------------------------------------
    # ZERO GEX
    #
    # 단순히 $130 같은 먼 strike의
    # 0 GEX를 Flip으로 잡지 않도록
    # 현재가에 가까운 zero만 허용.
    # --------------------------------------------------------

    zero_rows = data[
        data["net_gex"].abs()
        < 1e-9
    ].copy()

    if not zero_rows.empty:

        zero_rows["distance"] = (
            zero_rows["strike"]
            - current_price
        ).abs()
        )

        closest = (
            zero_rows
            .sort_values("distance")
            .iloc[0]
        )

        flip = safe_float(
            closest["strike"]
        )

        return {
            "flip":
                flip,

            "lower":
                flip,

            "upper":
                flip,

            "method":
                "ZERO",
        }

    # --------------------------------------------------------
    # SIGN CHANGE
    # --------------------------------------------------------

    candidates = []

    previous_strike = None
    previous_gex = None

    for _, row in data.iterrows():

        strike = safe_float(
            row["strike"]
        )

        gex = safe_float(
            row["net_gex"]
        )

        # ----------------------------------------------------
        # 0은 sign change 판단에서 건너뛴다.
        # ----------------------------------------------------

        if abs(gex) < 1e-12:

            continue

        if previous_gex is not None:

            sign_change = (
                (
                    previous_gex < 0
                    and gex > 0
                )
                or
                (
                    previous_gex > 0
                    and gex < 0
                )
            )

            if sign_change:

                candidates.append(
                    {
                        "lower":
                            previous_strike,

                        "upper":
                            strike,

                        "gex_lower":
                            previous_gex,

                        "gex_upper":
                            gex,
                    }
                )

        previous_strike = strike
        previous_gex = gex

    if not candidates:

        return empty_result

    # --------------------------------------------------------
    # CURRENT PRICE와 가장 가까운 sign change
    # --------------------------------------------------------

    def candidate_distance(
        candidate: dict[str, Any],
    ) -> float:

        midpoint = (
            candidate["lower"]
            + candidate["upper"]
        ) / 2.0

        return abs(
            midpoint
            - current_price
        )

    candidate = min(
        candidates,
        key=candidate_distance,
    )

    lower = safe_float(
        candidate["lower"]
    )

    upper = safe_float(
        candidate["upper"]
    )

    gex_lower = safe_float(
        candidate["gex_lower"]
    )

    gex_upper = safe_float(
        candidate["gex_upper"]
    )

    # --------------------------------------------------------
    # LINEAR INTERPOLATION
    # --------------------------------------------------------

    denominator = (
        gex_upper
        - gex_lower
    )

    if abs(denominator) < 1e-12:

        flip = (
            lower
            + upper
        ) / 2.0

    else:

        ratio = (
            -gex_lower
            / denominator
        )

        ratio = max(
            0.0,
            min(
                1.0,
                ratio,
            )
        )

        flip = (
            lower
            + ratio
            * (
                upper
                - lower
            )
        )

    return {
        "flip":
            safe_float(flip),

        "lower":
            lower,

        "upper":
            upper,

        "method":
            "LOCAL_SIGN_CHANGE",
    }


# ============================================================
# DATA QUALITY
# ============================================================

def calculate_quality(
    df: pd.DataFrame,
) -> dict[str, Any]:

    if (
        df is None
        or df.empty
    ):

        return {
            "rows":
                0,

            "gamma_rows":
                0,

            "strike_rows":
                0,

            "gamma_ratio":
                0.0,

            "oi_rows":
                0,
        }

    rows = len(df)

    gamma_rows = int(
        (
            df["gamma"]
            > 0
        ).sum()
    )

    oi_rows = int(
        (
            df["openInterest"]
            > 0
        ).sum()
    )

    strike_rows = int(
        df["strike"]
        .nunique()
    )

    gamma_ratio = (
        gamma_rows
        / rows
        * 100.0
    )

    return {
        "rows":
            rows,

        "gamma_rows":
            gamma_rows,

        "strike_rows":
            strike_rows,

        "gamma_ratio":
            gamma_ratio,

        "oi_rows":
            oi_rows,
    }


# ============================================================
# STRUCTURE
# ============================================================

def get_structure(
    net_gex: float,
) -> str:

    if net_gex > 0:

        return "🟢 POSITIVE GEX"

    if net_gex < 0:

        return "🔴 NEGATIVE GEX"

    return "🟡 NEUTRAL GEX"


def get_marker(
    value: float,
) -> str:

    if value > 0:

        return "🟢"

    if value < 0:

        return "🔴"

    return "🟡"


# ============================================================
# NEAR CURRENT PRICE
# ============================================================

def get_near_current(
    strike_df: pd.DataFrame,
    current_price: float,
    count: int = 9,
) -> pd.DataFrame:

    if (
        strike_df is None
        or strike_df.empty
    ):

        return pd.DataFrame()

    return (
        strike_df.assign(
            distance=(
                strike_df["strike"]
                - current_price
            ).abs()
        )
        .sort_values(
            [
                "distance",
                "strike",
            ]
        )
        .head(count)
        .drop(
            columns=[
                "distance"
            ]
        )
    )


# ============================================================
# TOP GEX
# ============================================================

def get_top_gex(
    strike_df: pd.DataFrame,
    count: int = 10,
) -> pd.DataFrame:

    if (
        strike_df is None
        or strike_df.empty
    ):

        return pd.DataFrame()

    return (
        strike_df.assign(
            abs_net=(
                strike_df[
                    "net_gex"
                ].abs()
            )
        )
        .sort_values(
            "abs_net",
            ascending=False,
        )
        .head(count)
        .drop(
            columns=[
                "abs_net"
            ]
        )
    )


# ============================================================
# EXTREMES
# ============================================================

def get_extremes(
    strike_df: pd.DataFrame,
) -> dict[str, Any]:

    if (
        strike_df is None
        or strike_df.empty
    ):

        return {
            "max_positive":
                None,

            "max_negative":
                None,
        }

    positive = strike_df[
        strike_df["net_gex"]
        > 0
    ]

    negative = strike_df[
        strike_df["net_gex"]
        < 0
    ]

    max_positive = None
    max_negative = None

    if not positive.empty:

        row = positive.loc[
            positive[
                "net_gex"
            ].idxmax()
        ]

        max_positive = (
            safe_float(
                row["strike"]
            ),
            safe_float(
                row["net_gex"]
            ),
        )

    if not negative.empty:

        row = negative.loc[
            negative[
                "net_gex"
            ].idxmin()
        ]

        max_negative = (
            safe_float(
                row["strike"]
            ),
            safe_float(
                row["net_gex"]
            ),
        )

    return {
        "max_positive":
            max_positive,

        "max_negative":
            max_negative,
    }


# ============================================================
# DEBUG
# ============================================================

def print_debug(
    df: pd.DataFrame,
) -> None:

    quality = calculate_quality(
        df
    )

    print()

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        "🔍 GEX DATA QUALITY"
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        f"Rows           : "
        f"{quality['rows']}"
    )

    print(
        f"Gamma rows > 0 : "
        f"{quality['gamma_rows']}"
    )

    print(
        f"Gamma ratio    : "
        f"{quality['gamma_ratio']:.1f}%"
    )

    print(
        f"OI rows > 0    : "
        f"{quality['oi_rows']}"
    )

    print(
        f"Strike count   : "
        f"{quality['strike_rows']}"
    )

    if not df.empty:

        if (
            "impliedVolatility"
            in df.columns
        ):

            valid_iv = df[
                df[
                    "impliedVolatility"
                ] > 0
            ][
                "impliedVolatility"
            ]

            if not valid_iv.empty:

                print()

                print(
                    "IV RANGE"
                )

                print(
                    "-" * 70
                )

                print(
                    f"Min IV         : "
                    f"{valid_iv.min():.4f}"
                )

                print(
                    f"Max IV         : "
                    f"{valid_iv.max():.4f}"
                )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    symbol = (
        DEFAULT_SYMBOL
        .upper()
        .strip()
    )

    print(
        f"Collecting {symbol}..."
    )

    collector = OptionCollector(
        symbol
    )

    # --------------------------------------------------------
    # 미국 동부시간 기준 오늘/가장 가까운 만기
    #
    # 한국 토요일 8/29
    # 미국 금요일 8/28
    #
    # => 8/28 만기 DTE=0 정상
    # --------------------------------------------------------

    result = (
        collector.collect_one_day()
    )

    if not result.get(
        "success",
        False,
    ):

        print()

        print(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        print(
            "❌ GEX COLLECTION FAILED"
        )

        print(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        print(
            f"Symbol     : "
            f"{symbol}"
        )

        print(
            f"Expiration : "
            f"{result.get('expiration')}"
        )

        print(
            f"Error      : "
            f"{result.get('error')}"
        )

        print(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        return

    current_price = safe_float(
        result.get(
            "current_price"
        )
    )

    expiration = result.get(
        "expiration"
    )

    dte = result.get(
        "DTE"
    )

    rows = result.get(
        "data",
        []
    )

    df = pd.DataFrame(
        rows
    )

    if (
        df.empty
        or current_price <= 0
    ):

        print(
            "❌ Invalid option data."
        )

        return

    print(
        f"Expiration: "
        f"{expiration}"
    )

    print_debug(
        df
    )

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    gex_df = prepare_gex_data(
        df,
        current_price,
    )

    if gex_df.empty:

        print()

        print(
            "❌ GEX calculation failed:"
            " no usable option rows."
        )

        return

    # --------------------------------------------------------
    # GAMMA CHECK
    # --------------------------------------------------------

    gamma_rows = int(
        (
            gex_df["gamma"]
            > 0
        ).sum()
    )

    if gamma_rows == 0:

        print()

        print(
            "❌ Failed: Gamma data is "
            "unavailable or all gamma "
            "values are zero."
        )

        return

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    totals = calculate_total_gex(
        gex_df
    )

    # --------------------------------------------------------
    # STRIKE
    # --------------------------------------------------------

    strike_df = aggregate_by_strike(
        gex_df
    )

    if strike_df.empty:

        print(
            "❌ Could not aggregate GEX."
        )

        return

    # --------------------------------------------------------
    # FLIP
    # --------------------------------------------------------

    flip = find_gex_flip(
        strike_df,
        current_price,
    )

    # --------------------------------------------------------
    # EXTREMES
    # --------------------------------------------------------

    extremes = get_extremes(
        strike_df
    )

    # --------------------------------------------------------
    # TOP
    # --------------------------------------------------------

    top = get_top_gex(
        strike_df,
        10,
    )

    # --------------------------------------------------------
    # NEAR
    # --------------------------------------------------------

    near = get_near_current(
        strike_df,
        current_price,
        9,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

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

    print(
        f"💰 Current Price : "
        f"${current_price:.2f}"
    )

    print(
        f"📅 Expiration    : "
        f"{expiration}"
    )

    print(
        f"⏳ DTE           : "
        f"{dte}"
    )

    print()

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    print(
        "⚡ TOTAL GEX"
    )

    print()

    print(
        f"CALL GEX : "
        f"{format_money(totals['call_gex'])}"
    )

    print(
        f"PUT GEX  : "
        f"{format_money(totals['put_gex'])}"
    )

    print(
        f"NET GEX  : "
        f"{format_money(totals['net_gex'])}"
    )

    print()

    print(
        f"STRUCTURE: "
        f"{get_structure(totals['net_gex'])}"
    )

    # --------------------------------------------------------
    # FLIP
    # --------------------------------------------------------

    print()

    print(
        "🔄 GEX FLIP"
    )

    if flip["flip"] is None:

        print(
            "GEX Flip : N/A"
        )

        print(
            "Range    : N/A"
        )

    else:

        print(
            f"GEX Flip : "
            f"${flip['flip']:.2f}"
        )

        print(
            f"Range    : "
            f"${flip['lower']:.2f}"
            f" ~ "
            f"${flip['upper']:.2f}"
        )

        print(
            f"Method   : "
            f"{flip['method']}"
        )

    # --------------------------------------------------------
    # EXTREMES
    # --------------------------------------------------------

    print()

    print(
        "🔥 GEX EXTREMES"
    )

    positive = (
        extremes["max_positive"]
    )

    negative = (
        extremes["max_negative"]
    )

    if positive is None:

        print(
            "🟢 Max Positive : N/A"
        )

    else:

        print(
            f"🟢 Max Positive : "
            f"${positive[0]:.2f} "
            f"({format_money(positive[1])})"
        )

    if negative is None:

        print(
            "🔴 Max Negative : N/A"
        )

    else:

        print(
            f"🔴 Max Negative : "
            f"${negative[0]:.2f} "
            f"({format_money(negative[1])})"
        )

    # --------------------------------------------------------
    # TOP
    # --------------------------------------------------------

    print()

    print(
        "🎯 TOP GEX STRIKES"
    )

    print(
        "-" * 70
    )

    for number, (_, row) in enumerate(
        top.iterrows(),
        start=1,
    ):

        net = safe_float(
            row["net_gex"]
        )

        marker = get_marker(
            net
        )

        print(
            f"{marker} "
            f"{number:02d}. "
            f"${row['strike']:.2f} | "
            f"CALL "
            f"{format_money(row['call_gex'])} | "
            f"PUT "
            f"{format_money(row['put_gex'])} | "
            f"NET "
            f"{format_money(net)}"
        )

    # --------------------------------------------------------
    # NEAR
    # --------------------------------------------------------

    print()

    print(
        "📍 NEAR CURRENT PRICE"
    )

    print(
        "-" * 70
    )

    for _, row in near.iterrows():

        net = safe_float(
            row["net_gex"]
        )

        marker = get_marker(
            net
        )

        print(
            f"{marker} "
            f"${row['strike']:.2f} | "
            f"Net GEX "
            f"{format_money(net)}"
        )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    quality = calculate_quality(
        df
    )

    print()

    print(
        "📊 DATA QUALITY"
    )

    print(
        f"Rows       : "
        f"{quality['rows']}"
    )

    print(
        f"Gamma rows : "
        f"{quality['gamma_rows']}"
    )

    print(
        f"Strike rows: "
        f"{quality['strike_rows']}"
    )

    print(
        f"Gamma ratio: "
        f"{quality['gamma_ratio']:.1f}%"
    )

    # --------------------------------------------------------
    # WARNING
    # --------------------------------------------------------

    print()

    print(
        "⚠️ GEX is a structural estimate "
        "based on model-derived Gamma × "
        "Open Interest."
    )

    print(
        "⚠️ It does not directly reveal "
        "dealer or institutional positions."
    )

    if dte == 0:

        print(
            "⚠️ DTE=0: Greeks use a minimum "
            "1-day model time value for "
            "numerical stability."
        )

    print()

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
