"""
FREE OPTION BOT
Normalizer

Collector가 Yahoo Finance에서 받은 원본 옵션 데이터를
BOT 내부에서 사용할 표준 데이터 구조로 정리한다.

원칙:
- 원본 데이터 훼손 최소화
- 숫자형 강제 변환
- NaN / inf 정리
- CALL / PUT 구분
- 기본 품질 검사
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


# ============================================================
# STANDARD COLUMNS
# ============================================================

STANDARD_COLUMNS = [
    "symbol",
    "contractSymbol",
    "expiration",
    "DTE",
    "option_type",
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "midPrice",
    "volume",
    "openInterest",
    "impliedVolatility",
    "premium",
    "underlying_price",
    "moneyness",
]


NUMERIC_COLUMNS = [
    "DTE",
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "midPrice",
    "volume",
    "openInterest",
    "impliedVolatility",
    "premium",
    "underlying_price",
    "moneyness",
]


# ============================================================
# UTILITIES
# ============================================================

def clean_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    숫자형 변환 + inf 제거.
    """

    result = pd.to_numeric(
        series,
        errors="coerce",
    )

    result = result.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    return result


def safe_number(
    value: Any,
) -> float | None:
    """
    단일 값을 안전하게 float 변환.
    """

    if value is None:
        return None

    try:

        result = float(value)

        if math.isnan(result):
            return None

        if math.isinf(result):
            return None

        return result

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# NORMALIZER
# ============================================================

class OptionNormalizer:
    """
    옵션 데이터 표준화 클래스.
    """

    def normalize(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Collector 결과를 표준 DataFrame으로 변환한다.
        """

        if df is None or df.empty:

            return pd.DataFrame(
                columns=STANDARD_COLUMNS
            )

        result = df.copy()

        # ----------------------------------------------------
        # Ensure columns
        # ----------------------------------------------------

        for column in STANDARD_COLUMNS:

            if column not in result.columns:

                result[column] = pd.NA

        # ----------------------------------------------------
        # String normalization
        # ----------------------------------------------------

        if "symbol" in result.columns:

            result["symbol"] = (
                result["symbol"]
                .astype("string")
                .str.upper()
                .str.strip()
            )

        if "option_type" in result.columns:

            result["option_type"] = (
                result["option_type"]
                .astype("string")
                .str.upper()
                .str.strip()
            )

        if "contractSymbol" in result.columns:

            result["contractSymbol"] = (
                result["contractSymbol"]
                .astype("string")
                .str.strip()
            )

        if "expiration" in result.columns:

            result["expiration"] = (
                result["expiration"]
                .astype("string")
                .str.strip()
            )

        # ----------------------------------------------------
        # Numeric normalization
        # ----------------------------------------------------

        for column in NUMERIC_COLUMNS:

            result[column] = clean_numeric(
                result[column]
            )

        # ----------------------------------------------------
        # Option type validation
        # ----------------------------------------------------

        result = result[
            result["option_type"].isin(
                [
                    "CALL",
                    "PUT",
                ]
            )
        ].copy()

        # ----------------------------------------------------
        # Strike validation
        # ----------------------------------------------------

        result = result[
            result["strike"].notna()
            & (result["strike"] > 0)
        ].copy()

        # ----------------------------------------------------
        # Volume / OI cleanup
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Bid / Ask cleanup
        # ----------------------------------------------------

        result["bid"] = (
            result["bid"]
            .fillna(0)
            .clip(lower=0)
        )

        result["ask"] = (
            result["ask"]
            .fillna(0)
            .clip(lower=0)
        )

        result["lastPrice"] = (
            result["lastPrice"]
            .fillna(0)
            .clip(lower=0)
        )

        # ----------------------------------------------------
        # Mid price
        # ----------------------------------------------------

        calculated_mid = (
            result["bid"]
            + result["ask"]
        ) / 2

        # bid/ask 둘 다 0이면 lastPrice 사용
        result["midPrice"] = calculated_mid

        invalid_mid = (
            result["midPrice"].isna()
            | (result["midPrice"] <= 0)
        )

        result.loc[
            invalid_mid,
            "midPrice",
        ] = result.loc[
            invalid_mid,
            "lastPrice",
        ]

        result["midPrice"] = (
            result["midPrice"]
            .fillna(0)
            .clip(lower=0)
        )

        # ----------------------------------------------------
        # Premium
        # ----------------------------------------------------

        # 옵션 1계약 = 100주
        result["premium"] = (
            result["midPrice"]
            * result["volume"]
            * 100
        )

        # ----------------------------------------------------
        # Moneyness
        # ----------------------------------------------------

        valid_underlying = (
            result["underlying_price"].notna()
            & (result["underlying_price"] > 0)
        )

        result["moneyness"] = pd.NA

        result.loc[
            valid_underlying,
            "moneyness",
        ] = (
            result.loc[
                valid_underlying,
                "strike",
            ]
            / result.loc[
                valid_underlying,
                "underlying_price",
            ]
        )

        # ----------------------------------------------------
        # Remove impossible values
        # ----------------------------------------------------

        result["impliedVolatility"] = (
            result["impliedVolatility"]
            .where(
                result["impliedVolatility"] >= 0
            )
        )

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        result = result.sort_values(
            by=[
                "option_type",
                "strike",
            ],
            ascending=[
                True,
                True,
            ],
        )

        # ----------------------------------------------------
        # Reset index
        # ----------------------------------------------------

        result = result.reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # Standard column order
        # ----------------------------------------------------

        result = result[
            STANDARD_COLUMNS
        ]

        return result


# ============================================================
# QUALITY REPORT
# ============================================================

def build_quality_report(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    표준화된 옵션 데이터의 기본 품질을 계산한다.
    """

    if df is None or df.empty:

        return {
            "quality": "LOW",
            "score": 0,
            "rows": 0,
            "calls": 0,
            "puts": 0,
            "missing_volume": 0,
            "missing_oi": 0,
            "missing_price": 0,
        }

    rows = len(df)

    calls = int(
        (
            df["option_type"]
            == "CALL"
        ).sum()
    )

    puts = int(
        (
            df["option_type"]
            == "PUT"
        ).sum()
    )

    missing_volume = int(
        df["volume"]
        .isna()
        .sum()
    )

    missing_oi = int(
        df["openInterest"]
        .isna()
        .sum()
    )

    missing_price = int(
        (
            df["midPrice"]
            <= 0
        ).sum()
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    score = 100

    if rows < 20:
        score -= 30

    if calls == 0:
        score -= 25

    if puts == 0:
        score -= 25

    if missing_volume > 0:
        score -= min(
            10,
            missing_volume
        )

    if missing_oi > 0:
        score -= min(
            10,
            missing_oi
        )

    if missing_price > 0:
        score -= min(
            10,
            missing_price
        )

    score = max(
        0,
        score,
    )

    if score >= 90:
        quality = "HIGH"

    elif score >= 70:
        quality = "MEDIUM"

    else:
        quality = "LOW"

    return {
        "quality": quality,
        "score": score,
        "rows": rows,
        "calls": calls,
        "puts": puts,
        "missing_volume": missing_volume,
        "missing_oi": missing_oi,
        "missing_price": missing_price,
    }


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def normalize_options(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:

    normalizer = OptionNormalizer()

    normalized = (
        normalizer.normalize(df)
    )

    quality = (
        build_quality_report(
            normalized
        )
    )

    return (
        normalized,
        quality,
    )
