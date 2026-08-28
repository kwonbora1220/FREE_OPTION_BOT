"""
FREE OPTION BOT
OPTION DATA NORMALIZER

Collector에서 받은 Yahoo Finance 옵션 데이터를
분석 모듈에서 사용할 수 있는 표준 형태로 정규화한다.

중요 컬럼:
- symbol
- expiration
- DTE
- option_type
- strike
- lastPrice
- bid
- ask
- volume
- openInterest
- impliedVolatility
- delta
- gamma
- vega
- underlying_price
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "symbol",
    "expiration",
    "DTE",
    "option_type",
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "volume",
    "openInterest",
    "impliedVolatility",
    "delta",
    "gamma",
    "vega",
    "underlying_price",
]


# ============================================================
# NUMERIC COLUMNS
# ============================================================

NUMERIC_COLUMNS = [
    "DTE",
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "volume",
    "openInterest",
    "impliedVolatility",
    "delta",
    "gamma",
    "vega",
    "underlying_price",
]


# ============================================================
# NORMALIZER
# ============================================================

def normalize_options(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Collector DataFrame을 표준 옵션 데이터로 변환한다.

    Returns
    -------
    normalized_df
    quality
    """

    if df is None or df.empty:

        quality = {
            "quality": "LOW",
            "score": 0,
            "rows": 0,
            "missing_columns": REQUIRED_COLUMNS,
            "gamma_available": False,
        }

        return (
            pd.DataFrame(
                columns=REQUIRED_COLUMNS
            ),
            quality,
        )

    result = df.copy()

    # ========================================================
    # COLUMN NAME NORMALIZATION
    # ========================================================

    rename_map = {
        # common aliases
        "type": "option_type",
        "optionType": "option_type",
        "option_type": "option_type",

        "open_interest": "openInterest",
        "openinterest": "openInterest",
        "oi": "openInterest",

        "last": "lastPrice",
        "last_price": "lastPrice",

        "implied_volatility":
            "impliedVolatility",

        "underlyingPrice":
            "underlying_price",

        "underlying_price":
            "underlying_price",

        # Greeks
        "Delta": "delta",
        "Gamma": "gamma",
        "Vega": "vega",

        "delta": "delta",
        "gamma": "gamma",
        "vega": "vega",
    }

    result = result.rename(
        columns=rename_map
    )

    # ========================================================
    # ENSURE REQUIRED COLUMNS
    # ========================================================

    for column in REQUIRED_COLUMNS:

        if column not in result.columns:

            result[column] = pd.NA

    # ========================================================
    # OPTION TYPE
    # ========================================================

    result["option_type"] = (
        result["option_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Handle common Yahoo representations

    result["option_type"] = (
        result["option_type"]
        .replace(
            {
                "C": "CALL",
                "CALLS": "CALL",
                "P": "PUT",
                "PUTS": "PUT",
            }
        )
    )

    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

    for column in NUMERIC_COLUMNS:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    # ========================================================
    # DEFAULT NUMERIC VALUES
    # ========================================================

    # These fields may legitimately be missing
    # in some Yahoo chains.

    for column in [
        "DTE",
        "lastPrice",
        "bid",
        "ask",
        "volume",
        "openInterest",
        "impliedVolatility",
        "delta",
        "gamma",
        "vega",
    ]:

        result[column] = (
            result[column]
            .fillna(0)
        )

    # ========================================================
    # NEGATIVE VALUES
    # ========================================================

    result["volume"] = (
        result["volume"]
        .clip(lower=0)
    )

    result["openInterest"] = (
        result["openInterest"]
        .clip(lower=0)
    )

    result["gamma"] = (
        result["gamma"]
        .clip(lower=0)
    )

    result["vega"] = (
        result["vega"]
        .clip(lower=0)
    )

    # ========================================================
    # IV NORMALIZATION
    # ========================================================

    # Yahoo can return IV either as:
    #
    # 0.36
    #
    # or
    #
    # 36
    #
    # Keep internal representation as decimal.

    iv_mask = (
        result["impliedVolatility"]
        > 5
    )

    result.loc[
        iv_mask,
        "impliedVolatility",
    ] = (
        result.loc[
            iv_mask,
            "impliedVolatility",
        ]
        / 100
    )

    # ========================================================
    # EXPIRATION
    # ========================================================

    if "expiration" in result.columns:

        result["expiration"] = (
            result["expiration"]
            .astype(str)
        )

    # ========================================================
    # SORT
    # ========================================================

    result = result[
        result["strike"]
        .notna()
    ].copy()

    result = result[
        result["option_type"].isin(
            [
                "CALL",
                "PUT",
            ]
        )
    ].copy()

    result = result.sort_values(
        [
            "option_type",
            "strike",
        ]
    )

    result = result.reset_index(
        drop=True
    )

    # ========================================================
    # QUALITY
    # ========================================================

    rows = len(result)

    if rows == 0:

        quality = {
            "quality": "LOW",
            "score": 0,
            "rows": 0,
            "missing_columns": [],
            "gamma_available": False,
            "gamma_nonzero_rows": 0,
        }

        return (
            result,
            quality,
        )

    # --------------------------------------------------------
    # Gamma quality
    # --------------------------------------------------------

    gamma_nonzero = int(
        (
            result["gamma"]
            > 0
        ).sum()
    )

    gamma_available = (
        gamma_nonzero > 0
    )

    gamma_ratio = (
        gamma_nonzero / rows
    )

    # --------------------------------------------------------
    # OI quality
    # --------------------------------------------------------

    oi_nonzero = int(
        (
            result["openInterest"]
            > 0
        ).sum()
    )

    oi_ratio = (
        oi_nonzero / rows
    )

    # --------------------------------------------------------
    # Volume quality
    # --------------------------------------------------------

    volume_nonzero = int(
        (
            result["volume"]
            > 0
        ).sum()
    )

    volume_ratio = (
        volume_nonzero / rows
    )

    # --------------------------------------------------------
    # IV quality
    # --------------------------------------------------------

    iv_nonzero = int(
        (
            result["impliedVolatility"]
            > 0
        ).sum()
    )

    iv_ratio = (
        iv_nonzero / rows
    )

    # ========================================================
    # SCORE
    # ========================================================

    score = 0

    score += int(
        gamma_ratio * 30
    )

    score += int(
        oi_ratio * 25
    )

    score += int(
        volume_ratio * 20
    )

    score += int(
        iv_ratio * 15
    )

    price_available = int(
        (
            result["underlying_price"]
            > 0
        ).sum()
        > 0
    )

    score += (
        10
        if price_available
        else 0
    )

    score = max(
        0,
        min(
            100,
            score,
        )
    )

    if score >= 85:

        quality_label = "HIGH"

    elif score >= 60:

        quality_label = "MEDIUM"

    else:

        quality_label = "LOW"

    # ========================================================
    # QUALITY RESULT
    # ========================================================

    quality = {
        "quality": quality_label,
        "score": score,
        "rows": rows,

        "gamma_available":
            gamma_available,

        "gamma_nonzero_rows":
            gamma_nonzero,

        "gamma_ratio":
            gamma_ratio,

        "oi_nonzero_rows":
            oi_nonzero,

        "oi_ratio":
            oi_ratio,

        "volume_nonzero_rows":
            volume_nonzero,

        "volume_ratio":
            volume_ratio,

        "iv_nonzero_rows":
            iv_nonzero,

        "iv_ratio":
            iv_ratio,

        "underlying_price_available":
            bool(price_available),

        "missing_columns": [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ],
    }

    return (
        result,
        quality,
    )


# ============================================================
# DEBUG FUNCTION
# ============================================================

def print_normalizer_debug(
    df: pd.DataFrame,
    quality: dict[str, Any],
) -> None:
    """
    Greeks가 제대로 전달되는지 확인하기 위한
    디버그 출력.
    """

    print()
    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        "🔍 NORMALIZER DEBUG"
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        f"Rows           : "
        f"{len(df):,}"
    )

    print(
        f"Gamma rows > 0 : "
        f"{quality.get('gamma_nonzero_rows', 0):,}"
    )

    print(
        f"Gamma ratio    : "
        f"{quality.get('gamma_ratio', 0) * 100:.1f}%"
    )

    print(
        f"OI rows > 0    : "
        f"{quality.get('oi_nonzero_rows', 0):,}"
    )

    print(
        f"Volume rows >0 : "
        f"{quality.get('volume_nonzero_rows', 0):,}"
    )

    print()

    print(
        "SAMPLE GREEKS"
    )

    print(
        "-" * 70
    )

    columns = [
        "option_type",
        "strike",
        "openInterest",
        "impliedVolatility",
        "delta",
        "gamma",
        "vega",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    if not df.empty:

        print(
            df[
                available
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print()
