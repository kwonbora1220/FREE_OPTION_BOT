"""
FREE OPTION BOT
OPTION NORMALIZER

Yahoo Finance 옵션체인을
봇 전체에서 사용하는 표준 데이터로 변환한다.

핵심:
- IV decimal normalization
- 비정상 IV 제거
- Black-Scholes Greeks
- DTE=0 지원
- 미국 동부시간 기준 DTE
"""

from __future__ import annotations

import math

from datetime import date, datetime
from typing import Any

from zoneinfo import ZoneInfo

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

RISK_FREE_RATE = 0.04
DIVIDEND_YIELD = 0.0

CONTRACT_MULTIPLIER = 100

US_EASTERN = ZoneInfo(
    "America/New_York"
)

# DTE=0 안정성
MIN_TIME_TO_EXPIRY = 1.0 / 365.0

# 모델 계산용 IV 상한
#
# 일반적인 NVDA 옵션 분석에서는
# 300% 이상의 IV는 대부분 데이터 이상치/극단 OTM noise로 취급.
#
# GEX 계산에 극단 IV가 들어가는 것을 방지.
MAX_MODEL_IV = 3.0


# ============================================================
# US DATE
# ============================================================

def get_us_date() -> date:
    return datetime.now(
        US_EASTERN
    ).date()


# ============================================================
# SAFE FLOAT
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
# IV NORMALIZATION
# ============================================================

def normalize_iv(
    value: Any,
) -> float:

    iv = safe_float(value)

    if iv is None or iv <= 0:
        return 0.0

    """
    공급원에 따라 IV가 다음처럼 들어올 수 있다.

    0.235  -> 23.5%
    23.5   -> 23.5%
    235    -> 235%

    내부에서는 항상 decimal.

    0.235
    0.50
    1.00
    """

    # 5 이상이면 percentage 형태로 판단
    if iv >= 5.0:
        iv = iv / 100.0

    # 비정상 IV 제거
    if iv <= 0:
        return 0.0

    if iv > MAX_MODEL_IV:
        return 0.0

    return float(iv)


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

    expiration_date = (
        datetime.strptime(
            str(expiration),
            "%Y-%m-%d",
        ).date()
    )

    return (
        expiration_date
        - get_us_date()
    ).days


# ============================================================
# BLACK-SCHOLES GREEKS
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
    ):
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
        }

    sigma = normalize_iv(iv)

    if sigma <= 0:
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
        }

    # --------------------------------------------------------
    # DTE=0
    #
    # T=0을 직접 사용하면 division by zero 발생.
    # 따라서 1일을 최소값으로 사용.
    # --------------------------------------------------------

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
                + 0.5
                * sigma
                * sigma
            )
            * T
        ) / (
            sigma
            * sqrt_T
        )

        d2 = (
            d1
            - sigma
            * sqrt_T
        )

    except (
        ValueError,
        ZeroDivisionError,
        OverflowError,
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

    elif option_type == "PUT":

        delta = (
            math.exp(-q * T)
            * (cdf1 - 1.0)
        )

    else:

        delta = 0.0

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
    # 1% IV change 기준
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

    elif option_type == "PUT":

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

    else:

        theta = 0.0

    theta /= 365.0

    return {
        "delta":
            safe_float(delta) or 0.0,

        "gamma":
            safe_float(gamma) or 0.0,

        "vega":
            safe_float(vega) or 0.0,

        "theta":
            safe_float(theta) or 0.0,
    }


# ============================================================
# NORMALIZE OPTIONS
# ============================================================

def normalize_options(
    df: pd.DataFrame,
    current_price: float | None = None,
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

    required_columns = [
        "contractSymbol",
        "expiration",
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
    # NUMERIC
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
    # OPTION TYPE
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

    result["DTE"] = result[
        "expiration"
    ].apply(
        lambda expiration:
            calculate_dte(
                expiration
            )
            if pd.notna(expiration)
            else 0
    )

    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    if (
        current_price is not None
        and current_price > 0
    ):

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
    # IV NORMALIZATION
    # --------------------------------------------------------

    raw_iv = result[
        "impliedVolatility"
    ].copy()

    result[
        "impliedVolatility"
    ] = raw_iv.apply(
        normalize_iv
    )

    # --------------------------------------------------------
    # IV VALID FLAG
    # --------------------------------------------------------

    result["iv_valid"] = (
        result[
            "impliedVolatility"
        ] > 0
    )

    # --------------------------------------------------------
    # MID PRICE
    # --------------------------------------------------------

    result["midPrice"] = (
        result["bid"]
        + result["ask"]
    ) / 2.0

    invalid_mid = (
        result["midPrice"].isna()
        | (
            result["midPrice"]
            <= 0
        )
    )

    result.loc[
        invalid_mid,
        "midPrice",
    ] = result.loc[
        invalid_mid,
        "lastPrice",
    ]

    # --------------------------------------------------------
    # PREMIUM
    # --------------------------------------------------------

    result["premium"] = (
        result["midPrice"]
        * result["volume"]
        * CONTRACT_MULTIPLIER
    )

    # --------------------------------------------------------
    # MONEYNESS
    # --------------------------------------------------------

    if (
        current_price is not None
        and current_price > 0
    ):

        spot = float(
            current_price
        )

        result["moneyness"] = (
            result["strike"]
            / spot
        )

        result[
            "distance_percent"
        ] = (
            (
                result["strike"]
                - spot
            )
            / spot
            * 100.0
        )

    else:

        result["moneyness"] = None

        result[
            "distance_percent"
        ] = None

    # --------------------------------------------------------
    # GREEKS
    # --------------------------------------------------------

    result["delta"] = 0.0
    result["gamma"] = 0.0
    result["vega"] = 0.0
    result["theta"] = 0.0

    if (
        current_price is not None
        and current_price > 0
    ):

        spot = float(
            current_price
        )

        for index, row in result.iterrows():

            strike = safe_float(
                row["strike"]
            )

            iv = safe_float(
                row[
                    "impliedVolatility"
                ]
            )

            try:

                dte = int(
                    row["DTE"]
                )

            except (
                TypeError,
                ValueError,
            ):

                dte = 0

            option_type = (
                row["option_type"]
            )

            # ------------------------------------------------
            # invalid data
            # ------------------------------------------------

            if (
                strike is None
                or strike <= 0
                or iv is None
                or iv <= 0
            ):

                continue

            greeks = calculate_greeks(
                spot=spot,
                strike=strike,
                iv=iv,
                dte=dte,
                option_type=option_type,
            )

            result.at[
                index,
                "delta"
            ] = greeks[
                "delta"
            ]

            result.at[
                index,
                "gamma"
            ] = greeks[
                "gamma"
            ]

            result.at[
                index,
                "vega"
            ] = greeks[
                "vega"
            ]

            result.at[
                index,
                "theta"
            ] = greeks[
                "theta"
            ]

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    result["volume"] = (
        pd.to_numeric(
            result["volume"],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
    )

    result["openInterest"] = (
        pd.to_numeric(
            result["openInterest"],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
    )

    result["gamma"] = (
        pd.to_numeric(
            result["gamma"],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
    )

    result["delta"] = (
        pd.to_numeric(
            result["delta"],
            errors="coerce",
        )
        .fillna(0)
    )

    result["vega"] = (
        pd.to_numeric(
            result["vega"],
            errors="coerce",
        )
        .fillna(0)
    )

    result["theta"] = (
        pd.to_numeric(
            result["theta"],
            errors="coerce",
        )
        .fillna(0)
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

    if (
        df is None
        or df.empty
    ):

        print(
            "Rows           : 0"
        )

        return

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

    volume_rows = int(
        (
            df["volume"]
            > 0
        ).sum()
    )

    print(
        f"Rows           : {rows}"
    )

    print(
        f"Gamma rows > 0 : "
        f"{gamma_rows}"
    )

    print(
        f"Gamma ratio    : "
        f"{gamma_rows / rows * 100:.1f}%"
    )

    print(
        f"OI rows > 0    : "
        f"{oi_rows}"
    )

    print(
        f"Volume rows >0 : "
        f"{volume_rows}"
    )

    # --------------------------------------------------------
    # IV RANGE
    # --------------------------------------------------------

    print()

    print(
        "IV RANGE"
    )

    print(
        "-" * 70
    )

    valid_iv = df[
        df["impliedVolatility"] > 0
    ]["impliedVolatility"]

    if valid_iv.empty:

        print(
            "No valid IV"
        )

    else:

        print(
            f"Min IV         : "
            f"{valid_iv.min():.4f}"
        )

        print(
            f"Max IV         : "
            f"{valid_iv.max():.4f}"
        )

        # ----------------------------------------------------
        # ATM-ish IV
        #
        # 현재가 ±10%만 사용
        # ----------------------------------------------------

        if (
            "underlying_price"
            in df.columns
        ):

            spot_values = pd.to_numeric(
                df[
                    "underlying_price"
                ],
                errors="coerce",
            ).dropna()

            if not spot_values.empty:

                spot = float(
                    spot_values.iloc[0]
                )

                atm_iv = df[
                    (
                        df["strike"]
                        >= spot * 0.90
                    )
                    & (
                        df["strike"]
                        <= spot * 1.10
                    )
                    & (
                        df[
                            "impliedVolatility"
                        ] > 0
                    )
                ][
                    "impliedVolatility"
                ]

                if not atm_iv.empty:

                    print(
                        f"ATM-ish IV     : "
                        f"{atm_iv.median():.4f}"
                    )

                else:

                    print(
                        "ATM-ish IV     : N/A"
                    )

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

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
        column
        for column in columns
        if column in df.columns
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

    # --------------------------------------------------------
    # ATM GAMMA CHECK
    # --------------------------------------------------------

    if (
        "underlying_price"
        in df.columns
    ):

        spot_values = pd.to_numeric(
            df[
                "underlying_price"
            ],
            errors="coerce",
        ).dropna()

        if not spot_values.empty:

            spot = float(
                spot_values.iloc[0]
            )

            atm = (
                df.assign(
                    atm_distance=(
                        df["strike"]
                        - spot
                    ).abs()
                )
                .sort_values(
                    "atm_distance"
                )
                .head(6)
            )

            print()

            print(
                "ATM GAMMA CHECK"
            )

            print(
                "-" * 70
            )

            print(
                atm[
                    [
                        "option_type",
                        "strike",
                        "impliedVolatility",
                        "openInterest",
                        "delta",
                        "gamma",
                    ]
                ]
                .to_string(
                    index=False
                )
            )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
