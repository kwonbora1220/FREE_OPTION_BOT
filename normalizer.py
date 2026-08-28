"""
FREE OPTION BOT
OPTION NORMALIZER

Yahoo Finance에서 받은 옵션체인을
봇 전체에서 사용할 표준 형태로 정규화한다.

Greeks:
- Yahoo가 직접 제공하는 값에 의존하지 않는다.
- Black-Scholes 모델로 Delta / Gamma / Vega / Theta 계산.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import math
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

RISK_FREE_RATE = 0.04
DIVIDEND_YIELD = 0.0

CONTRACT_MULTIPLIER = 100

MIN_TIME_TO_EXPIRY = 1.0 / (365.0 * 24.0)


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_float(
    value: Any,
) -> float | None:

    if value is None:
        return None

    try:

        value = float(value)

        if math.isnan(value):
            return None

        if math.isinf(value):
            return None

        return value

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# NORMAL DISTRIBUTION
# ============================================================

def normal_pdf(
    x: float,
) -> float:

    return (
        math.exp(
            -0.5 * x * x
        )
        / math.sqrt(
            2.0 * math.pi
        )
    )


def normal_cdf(
    x: float,
) -> float:

    return (
        0.5
        * (
            1.0
            + math.erf(
                x / math.sqrt(2.0)
            )
        )
    )


# ============================================================
# DTE
# ============================================================

def calculate_dte(
    expiration: str,
) -> int:

    expiration_date = datetime.strptime(
        str(expiration),
        "%Y-%m-%d",
    ).date()

    return (
        expiration_date
        - date.today()
    ).days


# ============================================================
# BLACK-SCHOLES
# ============================================================

def calculate_greeks(
    spot: float,
    strike: float,
    iv: float,
    dte: int,
    option_type: str,
) -> dict[str, float]:

    option_type = (
        str(option_type)
        .upper()
        .strip()
    )

    if (
        spot <= 0
        or strike <= 0
        or iv is None
        or iv <= 0
    ):

        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
        }

    # --------------------------------------------------------
    # Yahoo IV safety
    #
    # 0.36 = 36%
    # 36   = 3600% → convert
    # --------------------------------------------------------

    if iv > 5:

        iv = iv / 100.0

    sigma = float(iv)

    T = max(
        float(dte) / 365.0,
        MIN_TIME_TO_EXPIRY,
    )

    r = RISK_FREE_RATE
    q = DIVIDEND_YIELD

    sqrt_T = math.sqrt(T)

    try:

        d1 = (
            math.log(
                spot / strike
            )
            + (
                r
                - q
                + 0.5 * sigma * sigma
            )
            * T
        ) / (
            sigma * sqrt_T
        )

        d2 = (
            d1
            - sigma * sqrt_T
        )

    except (
        ValueError,
        ZeroDivisionError,
    ):

        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
        }

    pdf = normal_pdf(d1)

    cdf1 = normal_cdf(d1)

    cdf2 = normal_cdf(d2)

    # --------------------------------------------------------
    # DELTA
    # --------------------------------------------------------

    if option_type == "CALL":

        delta = (
            math.exp(-q * T)
            * cdf1
        )

    else:

        delta = (
            math.exp(-q * T)
            * (
                cdf1 - 1.0
            )
        )

    # --------------------------------------------------------
    # GAMMA
    # --------------------------------------------------------

    gamma = (
        math.exp(-q * T)
        * pdf
        / (
            spot
            * sigma
            * sqrt_T
        )
    )

    # --------------------------------------------------------
    # VEGA
    #
    # per 1% IV
    # --------------------------------------------------------

    vega = (
        spot
        * math.exp(-q * T)
        * pdf
        * sqrt_T
        / 100.0
    )

    # --------------------------------------------------------
    # THETA
    #
    # daily
    # --------------------------------------------------------

    first = (
        -(
            spot
            * math.exp(-q * T)
            * pdf
            * sigma
        )
        / (
            2.0
            * sqrt_T
        )
    )

    if option_type == "CALL":

        theta = (
            first
            - r
            * strike
            * math.exp(-r * T)
            * cdf2
            + q
            * spot
            * math.exp(-q * T)
            * cdf1
        )

    else:

        theta = (
            first
            + r
            * strike
            * math.exp(-r * T)
            * normal_cdf(-d2)
            - q
            * spot
            * math.exp(-q * T)
            * normal_cdf(-d1)
        )

    theta /= 365.0

    return {
        "delta": safe_float(delta) or 0.0,
        "gamma": safe_float(gamma) or 0.0,
        "vega": safe_float(vega) or 0.0,
        "theta": safe_float(theta) or 0.0,
    }


# ============================================================
# NORMALIZE OPTIONS
# ============================================================

def normalize_options(
    df: pd.DataFrame,
    current_price: float | None = None,
) -> pd.DataFrame:

    if df is None or df.empty:

        return pd.DataFrame()

    result = df.copy()

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "contractSymbol",
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
    ]

    for column in required_columns:

        if column not in result.columns:

            result[column] = None

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "volume",
        "openInterest",
        "impliedVolatility",
    ]

    for column in numeric_columns:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Option type
    # --------------------------------------------------------

    result["option_type"] = (
        result["option_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # DTE
    # --------------------------------------------------------

    result["DTE"] = result.apply(
        lambda row:
        calculate_dte(
            row["expiration"]
        )
        if pd.notna(
            row["expiration"]
        )
        else 0,
        axis=1,
    )

    # --------------------------------------------------------
    # Spot
    # --------------------------------------------------------

    if current_price is not None:

        result[
            "underlying_price"
        ] = float(current_price)

    elif (
        "underlying_price"
        not in result.columns
    ):

        result[
            "underlying_price"
        ] = None

    # --------------------------------------------------------
    # Mid price
    # --------------------------------------------------------

    result["midPrice"] = (
        result["bid"]
        + result["ask"]
    ) / 2.0

    result["midPrice"] = (
        result["midPrice"]
        .where(
            result["midPrice"].notna(),
            result["lastPrice"],
        )
    )

    # --------------------------------------------------------
    # Premium
    # --------------------------------------------------------

    result["premium"] = (
        result["midPrice"]
        * result["volume"]
        * CONTRACT_MULTIPLIER
    )

    # --------------------------------------------------------
    # Moneyness
    # --------------------------------------------------------

    if (
        current_price is not None
        and current_price > 0
    ):

        result["moneyness"] = (
            result["strike"]
            / float(current_price)
        )

    else:

        result["moneyness"] = None

    # --------------------------------------------------------
    # Greeks
    # --------------------------------------------------------

    result["delta"] = 0.0
    result["gamma"] = 0.0
    result["vega"] = 0.0
    result["theta"] = 0.0

    if (
        current_price is not None
        and current_price > 0
    ):

        for index, row in result.iterrows():

            strike = safe_float(
                row["strike"]
            )

            iv = safe_float(
                row["impliedVolatility"]
            )

            dte = int(
                row["DTE"]
                if pd.notna(
                    row["DTE"]
                )
                else 0
            )

            option_type = (
                row["option_type"]
            )

            if (
                strike is None
                or strike <= 0
                or iv is None
                or iv <= 0
            ):

                continue

            greeks = calculate_greeks(
                spot=float(
                    current_price
                ),
                strike=strike,
                iv=iv,
                dte=dte,
                option_type=option_type,
            )

            result.at[
                index,
                "delta"
            ] = greeks["delta"]

            result.at[
                index,
                "gamma"
            ] = greeks["gamma"]

            result.at[
                index,
                "vega"
            ] = greeks["vega"]

            result.at[
                index,
                "theta"
            ] = greeks["theta"]

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    result["volume"] = (
        result["volume"]
        .fillna(0)
        .clip(lower=0)
    )

    result["openInterest"] = (
        result["openInterest"]
        .fillna(0)
        .clip(lower=0)
    )

    result["gamma"] = (
        result["gamma"]
        .fillna(0)
        .clip(lower=0)
    )

    return result.reset_index(
        drop=True
    )


# ============================================================
# DEBUG
# ============================================================

def print_normalizer_debug(
    df: pd.DataFrame,
) -> None:

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

    if df is None or df.empty:

        print(
            "Rows           : 0"
        )

        print(
            "Gamma rows > 0 : 0"
        )

        print(
            "Gamma ratio    : 0.0%"
        )

        return

    rows = len(df)

    gamma_rows = int(
        (
            pd.to_numeric(
                df["gamma"],
                errors="coerce",
            )
            > 0
        ).sum()
    )

    oi_rows = int(
        (
            pd.to_numeric(
                df["openInterest"],
                errors="coerce",
            )
            > 0
        ).sum()
    )

    volume_rows = int(
        (
            pd.to_numeric(
                df["volume"],
                errors="coerce",
            )
            > 0
        ).sum()
    )

    ratio = (
        gamma_rows
        / rows
        * 100
        if rows > 0
        else 0
    )

    print(
        f"Rows           : {rows}"
    )

    print(
        f"Gamma rows > 0 : {gamma_rows}"
    )

    print(
        f"Gamma ratio    : {ratio:.1f}%"
    )

    print(
        f"OI rows > 0    : {oi_rows}"
    )

    print(
        f"Volume rows >0 : {volume_rows}"
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

    columns = [
        c
        for c in columns
        if c in df.columns
    ]

    print(
        df[
            columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
