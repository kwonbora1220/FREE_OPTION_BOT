"""
FREE OPTION BOT
GEX ANALYSIS

Yahoo Finance 무료 옵션체인 + 자체 Black-Scholes Greeks 기반 GEX.

핵심:
- OptionCollector.collect_one_day() 사용
- 미국 동부시간 기준 만기
- 한국시간 토요일이어도 미국 금요일 장중이면 DTE=0 정상
- CALL GEX / PUT GEX / NET GEX
- GEX Flip
- GEX Extremes
- TOP GEX Strikes
- Current Price 주변 GEX
- Data Quality
- DTE=0 지원

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

# 현재가에서 너무 먼 strike를 제거할지 여부
FILTER_TO_NEAR_SPOT = False

# FILTER_TO_NEAR_SPOT=True일 때 사용
SPOT_RANGE_PERCENT = 0.50

# GEX Flip 탐색 범위
#
# 현재가 주변의 flip만 찾는다.
# 먼 OTM/ITM 영역의 의미 없는 sign change를
# GEX Flip으로 선택하는 문제를 방지한다.
FLIP_RANGE_PERCENT = 0.25

# 현재가 주변 GEX 표시 개수
NEAR_COUNT = 9

# TOP GEX 개수
TOP_COUNT = 10


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


# ============================================================
# PREPARE GEX DATA
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

    if current_price <= 0:
        return pd.DataFrame()

    result = df.copy()

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "option_type",
        "strike",
        "openInterest",
        "gamma",
    ]

    for column in required_columns:

        if column not in result.columns:

            result[column] = 0.0

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    numeric_columns = [
        "strike",
        "openInterest",
        "gamma",
    ]

    for column in numeric_columns:

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
        result["openInterest"] > 0
    ].copy()

    result = result[
        result["gamma"] > 0
    ].copy()

    result = result[
        result["option_type"].isin(
            [
                "CALL",
                "PUT",
            ]
        )
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

    if result.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # GEX
    #
    # Gamma × OI × 100 × Spot² × 0.01
    #
    # CALL = positive
    # PUT  = negative
    #
    # 이것은 구조적 추정값이다.
    # 실제 dealer positioning을 의미하지 않는다.
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
        pd.to_numeric(
            result["gex"],
            errors="coerce",
        )
        .replace(
            [
                float("inf"),
                float("-inf"),
            ],
            0.0,
        )
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # SIGNED GEX
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
        .groupby("strike")["gex"]
        .sum()
    )

    puts = (
        df[
            df["option_type"]
            == "PUT"
        ]
        .groupby("strike")["gex"]
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
                    -put_gex_abs,

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
    }

    if (
        strike_df is None
        or strike_df.empty
        or current_price <= 0
    ):
        return empty_result

    data = (
        strike_df[
            [
                "strike",
                "net_gex",
            ]
        ]
        .copy()
        .sort_values("strike")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 현재가 주변만 탐색
    #
    # 예:
    # 현재가 220.70
    # FLIP_RANGE_PERCENT = 0.25
    #
    # 165.5 ~ 275.9
    #
    # $130 같은 먼 strike를 flip으로 잡지 않는다.
    # --------------------------------------------------------

    lower_bound = (
        current_price
        * (
            1.0
            - FLIP_RANGE_PERCENT
        )
    )

    upper_bound = (
        current_price
        * (
            1.0
            + FLIP_RANGE_PERCENT
        )
    )

    data = data[
        (
            data["strike"]
            >= lower_bound
        )
        & (
            data["strike"]
            <= upper_bound
        )
    ].copy()

    if data.empty:
        return empty_result

    # --------------------------------------------------------
    # ZERO
    # --------------------------------------------------------

    zero_rows = data[
        data["net_gex"] == 0
    ].copy()

    if not zero_rows.empty:

        zero_rows["distance"] = (
            zero_rows["strike"]
            - current_price
        ).abs()
        
        closest = (
            zero_rows
            .sort_values("distance")
            .iloc[0]
        )

        flip = safe_float(
            closest["strike"]
        )

        return {
            "flip": flip,
            "lower": flip,
            "upper": flip,
        }

    # --------------------------------------------------------
    # SIGN CHANGE
    # --------------------------------------------------------

    candidates = []

    for i in range(
        1,
        len(data),
    ):

        previous = data.iloc[
            i - 1
        ]

        current = data.iloc[
            i
        ]

        previous_strike = safe_float(
            previous["strike"]
        )

        current_strike = safe_float(
            current["strike"]
        )

        previous_gex = safe_float(
            previous["net_gex"]
        )

        current_gex = safe_float(
            current["net_gex"]
        )

        # 둘 중 하나가 0이면 이미 위에서 처리
        if (
            previous_gex == 0
            or current_gex == 0
        ):
            continue

        sign_change = (
            (
                previous_gex < 0
                and current_gex > 0
            )
            or
            (
                previous_gex > 0
                and current_gex < 0
            )
        )

        if not sign_change:
            continue

        midpoint = (
            previous_strike
            + current_strike
        ) / 2.0

        distance = abs(
            midpoint
            - current_price
        )

        candidates.append(
            {
                "lower":
                    previous_strike,

                "upper":
                    current_strike,

                "gex_lower":
                    previous_gex,

                "gex_upper":
                    current_gex,

                "distance":
                    distance,
            }
        )

    if not candidates:
        return empty_result

    # --------------------------------------------------------
    # 현재가에 가장 가까운 SIGN CHANGE
    # --------------------------------------------------------

    candidate = min(
        candidates,
        key=lambda x:
            x["distance"],
    )

    lower = candidate[
        "lower"
    ]

    upper = candidate[
        "upper"
    ]

    gex_lower = candidate[
        "gex_lower"
    ]

    gex_upper = candidate[
        "gex_upper"
    ]

    # --------------------------------------------------------
    # LINEAR INTERPOLATION
    # --------------------------------------------------------

    denominator = (
        gex_upper
        - gex_lower
    )

    if abs(
        denominator
    ) < 1e-12:

        flip = (
            lower
            + upper
        ) / 2.0

    else:

        flip = (
            lower
            + (
                -gex_lower
                / denominator
            )
            * (
                upper
                - lower
            )
        )

    # --------------------------------------------------------
    # 안전 범위
    # --------------------------------------------------------

    flip = max(
        lower,
        min(
            upper,
            flip,
        ),
    )

    return {
        "flip":
            safe_float(flip),

        "lower":
            safe_float(lower),

        "upper":
            safe_float(upper),
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
            "rows": 0,
            "gamma_rows": 0,
            "strike_rows": 0,
            "gamma_ratio": 0.0,
            "oi_rows": 0,
        }

    rows = len(df)

    gamma_rows = int(
        (
            pd.to_numeric(
                df["gamma"],
                errors="coerce",
            ).fillna(0)
            > 0
        ).sum()
    )

    oi_rows = int(
        (
            pd.to_numeric(
                df["openInterest"],
                errors="coerce",
            ).fillna(0)
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
    count: int = NEAR_COUNT,
) -> pd.DataFrame:

    if (
        strike_df is None
        or strike_df.empty
    ):
        return pd.DataFrame()

    result = strike_df.copy()

    result["distance"] = (
        result["strike"]
        - current_price
    ).abs()

    result = (
        result
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

    return result


# ============================================================
# TOP GEX
# ============================================================

def get_top_gex(
    strike_df: pd.DataFrame,
    count: int = TOP_COUNT,
) -> pd.DataFrame:

    if (
        strike_df is None
        or strike_df.empty
    ):
        return pd.DataFrame()

    result = strike_df.copy()

    result["abs_net"] = (
        result["net_gex"]
        .abs()
    )

    result = (
        result
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

    return result


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
            "max_positive": None,
            "max_negative": None,
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

    if (
        not df.empty
        and "impliedVolatility"
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
    # IMPORTANT
    #
    # OptionCollector에는 collect()가 없음.
    #
    # 반드시 collect_one_day() 사용.
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
        TOP_COUNT,
    )

    # --------------------------------------------------------
    # NEAR
    # --------------------------------------------------------

    near = get_near_current(
        strike_df,
        current_price,
        NEAR_COUNT,
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

    # ========================================================
    # TOTAL GEX
    # ========================================================

    print()

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

    # ========================================================
    # FLIP
    # ========================================================

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

        print(
            "⚠️ No sign change found "
            "near current price."
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

    # ========================================================
    # EXTREMES
    # ========================================================

    print()

    print(
        "🔥 GEX EXTREMES"
    )

    positive = (
        extremes[
            "max_positive"
        ]
    )

    negative = (
        extremes[
            "max_negative"
        ]
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
            f"${safe_float(row['strike']):.2f} | "
            f"CALL "
            f"{format_money(row['call_gex'])} | "
            f"PUT "
            f"{format_money(row['put_gex'])} | "
            f"NET "
            f"{format_money(net)}"
        )

    # ========================================================
    # NEAR CURRENT PRICE
    # ========================================================

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
            f"${safe_float(row['strike']):.2f} | "
            f"Net GEX "
            f"{format_money(net)}"
        )

    # ========================================================
    # DATA QUALITY
    # ========================================================

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

    # ========================================================
    # WARNING
    # ========================================================

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
