"""
FREE OPTION BOT
OPTION COLLECTOR

Yahoo Finance / yfinance를 이용해서
미국 주식 옵션체인을 수집한다.

무료 데이터 원칙:
- Yahoo Finance
- 무료
- 공개 데이터
- 유료 API 없음
- Unusual Whales 없음
- 실시간 옵션 데이터 아님

IMPORTANT
---------
Yahoo Finance 옵션체인은 기본적으로
strike / price / volume / OI / IV 등의 데이터를 제공한다.

Delta / Gamma / Vega / Theta는 Yahoo에서 직접 받지 않고
Black-Scholes 모델을 이용해 이 코드에서 계산한다.

따라서 Greeks는:

    MODEL-DERIVED GREEKS

이다.

특히 GEX는:

    Gamma × Open Interest

를 이용한 구조적 추정치이며
실제 딜러 포지션을 직접 관측하는 값이 아니다.
"""

from __future__ import annotations

import math
import time

from datetime import datetime, date

from typing import Any

import pandas as pd

import yfinance as yf

from config import (
    DEFAULT_SYMBOL,
    ONE_DAY_MAX_DTE,
    REQUEST_DELAY_SECONDS,
    REQUIRED_OPTION_COLUMNS,
)


# ============================================================
# CONFIG
# ============================================================

# 미국 단기 무위험금리의 근사값.
#
# 무료 데이터만 사용하는 프로젝트이므로
# 외부 금리 API를 추가하지 않는다.
#
# 필요하면 config.py로 이동할 수 있다.
RISK_FREE_RATE = 0.04


# 배당수익률.
#
# 현재 단계에서는 무료/안정성을 위해
# 기본 0으로 둔다.
#
# 이후 필요하면 Yahoo dividendYield를 연결할 수 있다.
DEFAULT_DIVIDEND_YIELD = 0.0


# DTE = 0인 경우 Black-Scholes 계산에서
# T=0이 되면 division by zero가 발생한다.
#
# 따라서 최소 1시간을 사용한다.
MIN_TIME_TO_EXPIRY = 1.0 / (365.0 * 24.0)


# 옵션 계약 승수
CONTRACT_MULTIPLIER = 100


# ============================================================
# UTILITIES
# ============================================================

def safe_float(
    value: Any,
) -> float | None:
    """
    숫자로 변환할 수 없는 값을 None으로 반환한다.
    """

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


def calculate_dte(
    expiration: str,
) -> int:
    """
    만기일까지 남은 날짜를 계산한다.
    """

    expiration_date = datetime.strptime(
        expiration,
        "%Y-%m-%d",
    ).date()

    today = date.today()

    return (
        expiration_date
        - today
    ).days


# ============================================================
# BLACK-SCHOLES FUNCTIONS
# ============================================================

def normal_pdf(
    x: float,
) -> float:
    """
    Standard Normal PDF.
    """

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
    """
    Standard Normal CDF.

    scipy 없이 math.erf 사용.
    """

    return (
        0.5
        * (
            1.0
            + math.erf(
                x / math.sqrt(2.0)
            )
        )
    )


def black_scholes_greeks(
    spot: float,
    strike: float,
    volatility: float,
    time_to_expiry: float,
    option_type: str,
    risk_free_rate: float = RISK_FREE_RATE,
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
) -> dict[str, float]:
    """
    Black-Scholes 기반 Greeks 계산.

    Parameters
    ----------
    spot:
        현재 주가

    strike:
        옵션 행사가

    volatility:
        IV, decimal format.
        예:
            0.36 = 36%

    time_to_expiry:
        years

    option_type:
        CALL / PUT

    Returns
    -------
    delta
    gamma
    vega
    theta
    """

    option_type = (
        str(option_type)
        .upper()
        .strip()
    )

    # --------------------------------------------------------
    # 기본 검증
    # --------------------------------------------------------

    if spot <= 0:

        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
        }

    if strike <= 0:

        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
        }

    if volatility <= 0:

        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
        }

    # --------------------------------------------------------
    # Expiration protection
    # --------------------------------------------------------

    T = max(
        float(time_to_expiry),
        MIN_TIME_TO_EXPIRY,
    )

    sigma = float(
        volatility
    )

    r = float(
        risk_free_rate
    )

    q = float(
        dividend_yield
    )

    # --------------------------------------------------------
    # d1 / d2
    # --------------------------------------------------------

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
            ) * T
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

    # --------------------------------------------------------
    # Common terms
    # --------------------------------------------------------

    pdf_d1 = normal_pdf(
        d1
    )

    cdf_d1 = normal_cdf(
        d1
    )

    cdf_d2 = normal_cdf(
        d2
    )

    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    if option_type == "CALL":

        delta = (
            math.exp(
                -q * T
            )
            * cdf_d1
        )

    else:

        delta = (
            math.exp(
                -q * T
            )
            * (
                cdf_d1 - 1.0
            )
        )

    # --------------------------------------------------------
    # Gamma
    #
    # Gamma is identical for CALL / PUT.
    # --------------------------------------------------------

    gamma = (
        math.exp(
            -q * T
        )
        * pdf_d1
        / (
            spot
            * sigma
            * sqrt_T
        )
    )

    # --------------------------------------------------------
    # Vega
    #
    # This is vega for a 1.00 volatility change.
    #
    # For human display, usually vega per 1% IV change
    # is more intuitive, so divide by 100.
    # --------------------------------------------------------

    vega = (
        spot
        * math.exp(
            -q * T
        )
        * pdf_d1
        * sqrt_T
        / 100.0
    )

    # --------------------------------------------------------
    # Theta
    #
    # Approximate annualized theta converted to daily.
    # --------------------------------------------------------

    first_term = (
        -(
            spot
            * math.exp(
                -q * T
            )
            * pdf_d1
            * sigma
        )
        / (
            2.0
            * sqrt_T
        )
    )

    if option_type == "CALL":

        theta = (
            first_term
            - r
            * strike
            * math.exp(
                -r * T
            )
            * cdf_d2
            + q
            * spot
            * math.exp(
                -q * T
            )
            * cdf_d1
        )

    else:

        theta = (
            first_term
            + r
            * strike
            * math.exp(
                -r * T
            )
            * normal_cdf(
                -d2
            )
            - q
            * spot
            * math.exp(
                -q * T
            )
            * normal_cdf(
                -d1
            )
        )

    # Annual theta → daily theta

    theta_daily = (
        theta / 365.0
    )

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    values = {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta_daily,
    }

    for key in values:

        value = safe_float(
            values[key]
        )

        if value is None:

            values[key] = 0.0

        else:

            values[key] = value

    return values


# ============================================================
# OPTION COLLECTOR
# ============================================================

class OptionCollector:
    """
    Yahoo Finance 옵션체인 Collector.
    """

    def __init__(
        self,
        symbol: str = DEFAULT_SYMBOL,
    ):

        self.symbol = (
            symbol
            .upper()
            .strip()
        )

        if not self.symbol:

            raise ValueError(
                "Symbol is empty."
            )

        self.ticker = yf.Ticker(
            self.symbol
        )

    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    def get_current_price(
        self,
    ) -> float | None:
        """
        현재 가격을 가져온다.
        """

        try:

            fast_info = (
                self.ticker.fast_info
            )

            price = fast_info.get(
                "lastPrice"
            )

            price = safe_float(
                price
            )

            if price is not None:

                return price

        except Exception:

            pass

        try:

            history = (
                self.ticker.history(
                    period="1d",
                    interval="1m",
                )
            )

            if (
                history is not None
                and not history.empty
            ):

                close = (
                    history[
                        "Close"
                    ]
                    .dropna()
                )

                if not close.empty:

                    return safe_float(
                        close.iloc[-1]
                    )

        except Exception:

            pass

        try:

            history = (
                self.ticker.history(
                    period="5d",
                    interval="1d",
                )
            )

            if (
                history is not None
                and not history.empty
            ):

                close = (
                    history[
                        "Close"
                    ]
                    .dropna()
                )

                if not close.empty:

                    return safe_float(
                        close.iloc[-1]
                    )

        except Exception:

            pass

        return None

    # --------------------------------------------------------
    # EXPIRATIONS
    # --------------------------------------------------------

    def get_expirations(
        self,
    ) -> list[str]:

        try:

            expirations = (
                self.ticker.options
            )

        except Exception:

            return []

        if not expirations:

            return []

        return list(
            expirations
        )

    # --------------------------------------------------------
    # NEAREST EXPIRATION
    # --------------------------------------------------------

    def get_nearest_expiration(
        self,
        max_dte: int = ONE_DAY_MAX_DTE,
    ) -> str | None:

        expirations = (
            self.get_expirations()
        )

        if not expirations:

            return None

        today = date.today()

        candidates = []

        for expiration in expirations:

            try:

                expiration_date = (
                    datetime.strptime(
                        expiration,
                        "%Y-%m-%d",
                    ).date()
                )

            except ValueError:

                continue

            dte = (
                expiration_date
                - today
            ).days

            if (
                0 <= dte <= max_dte
            ):

                candidates.append(
                    (
                        dte,
                        expiration,
                    )
                )

        if not candidates:

            return None

        candidates.sort(
            key=lambda x: x[0]
        )

        return candidates[0][1]

    # --------------------------------------------------------
    # FETCH OPTION CHAIN
    # --------------------------------------------------------

    def fetch_chain(
        self,
        expiration: str,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

        chain = (
            self.ticker.option_chain(
                expiration
            )
        )

        calls = chain.calls.copy()

        puts = chain.puts.copy()

        calls["option_type"] = (
            "CALL"
        )

        puts["option_type"] = (
            "PUT"
        )

        calls["expiration"] = (
            expiration
        )

        puts["expiration"] = (
            expiration
        )

        return (
            calls,
            puts,
        )

    # --------------------------------------------------------
    # CALCULATE GREEKS
    # --------------------------------------------------------

    def calculate_greeks(
        self,
        df: pd.DataFrame,
        current_price: float | None,
    ) -> pd.DataFrame:
        """
        Yahoo IV를 이용하여
        Delta / Gamma / Vega / Theta 계산.
        """

        result = df.copy()

        # 기본 컬럼
        result["delta"] = 0.0

        result["gamma"] = 0.0

        result["vega"] = 0.0

        result["theta"] = 0.0

        if (
            result.empty
            or current_price is None
            or current_price <= 0
        ):

            return result

        # ----------------------------------------------------
        # DTE
        # ----------------------------------------------------

        expiration = None

        if (
            "expiration"
            in result.columns
        ):

            expiration_values = (
                result[
                    "expiration"
                ]
                .dropna()
                .astype(str)
            )

            if not expiration_values.empty:

                expiration = (
                    expiration_values.iloc[0]
                )

        if expiration is None:

            return result

        try:

            dte = calculate_dte(
                expiration
            )

        except Exception:

            dte = 0

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        time_to_expiry = max(
            dte / 365.0,
            MIN_TIME_TO_EXPIRY,
        )

        # ----------------------------------------------------
        # IV normalization
        # ----------------------------------------------------

        result[
            "impliedVolatility"
        ] = pd.to_numeric(
            result[
                "impliedVolatility"
            ],
            errors="coerce",
        )

        # Yahoo normally returns decimal:
        #
        # 0.36 = 36%
        #
        # But protect against 36.

        iv_mask = (
            result[
                "impliedVolatility"
            ]
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
            / 100.0
        )

        # ----------------------------------------------------
        # Row calculation
        # ----------------------------------------------------

        for index, row in result.iterrows():

            strike = safe_float(
                row.get(
                    "strike"
                )
            )

            iv = safe_float(
                row.get(
                    "impliedVolatility"
                )
            )

            option_type = str(
                row.get(
                    "option_type",
                    "",
                )
            ).upper()

            if (
                strike is None
                or strike <= 0
            ):

                continue

            if (
                iv is None
                or iv <= 0
            ):

                continue

            try:

                greeks = (
                    black_scholes_greeks(
                        spot=current_price,
                        strike=strike,
                        volatility=iv,
                        time_to_expiry=
                            time_to_expiry,
                        option_type=
                            option_type,
                    )
                )

            except Exception:

                continue

            result.at[
                index,
                "delta",
            ] = greeks[
                "delta"
            ]

            result.at[
                index,
                "gamma",
            ] = greeks[
                "gamma"
            ]

            result.at[
                index,
                "vega",
            ] = greeks[
                "vega"
            ]

            result.at[
                index,
                "theta",
            ] = greeks[
                "theta"
            ]

        return result

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    def normalize(
        self,
        df: pd.DataFrame,
        current_price: float | None,
    ) -> pd.DataFrame:

        if (
            df is None
            or df.empty
        ):

            return pd.DataFrame()

        result = df.copy()

        # ----------------------------------------------------
        # Required columns
        # ----------------------------------------------------

        for column in (
            REQUIRED_OPTION_COLUMNS
        ):

            if (
                column
                not in result.columns
            ):

                result[column] = None

        # ----------------------------------------------------
        # Numeric
        # ----------------------------------------------------

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

            result[column] = (
                pd.to_numeric(
                    result[column],
                    errors="coerce",
                )
            )

        # ----------------------------------------------------
        # DTE
        # ----------------------------------------------------

        result["DTE"] = (
            result[
                "expiration"
            ].apply(
                lambda x:
                calculate_dte(x)
                if isinstance(
                    x,
                    str,
                )
                else None
            )
        )

        # ----------------------------------------------------
        # Underlying
        # ----------------------------------------------------

        result[
            "underlying_price"
        ] = current_price

        # ----------------------------------------------------
        # Mid
        # ----------------------------------------------------

        result["midPrice"] = (
            result["bid"]
            + result["ask"]
        ) / 2.0

        result["midPrice"] = (
            result["midPrice"]
            .where(
                result[
                    "midPrice"
                ].notna(),
                result[
                    "lastPrice"
                ],
            )
        )

        # ----------------------------------------------------
        # Premium proxy
        # ----------------------------------------------------

        result["premium"] = (
            result["midPrice"]
            * result["volume"]
            * CONTRACT_MULTIPLIER
        )

        # ----------------------------------------------------
        # Moneyness
        # ----------------------------------------------------

        if (
            current_price is not None
            and current_price > 0
        ):

            result["moneyness"] = (
                result["strike"]
                / current_price
            )

        else:

            result["moneyness"] = None

        # ----------------------------------------------------
        # Greeks
        # ----------------------------------------------------

        result = (
            self.calculate_greeks(
                result,
                current_price,
            )
        )

        # ----------------------------------------------------
        # Final columns
        # ----------------------------------------------------

        preferred_columns = [

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

            "delta",

            "gamma",

            "vega",

            "theta",

            "premium",

            "underlying_price",

            "moneyness",
        ]

        existing_columns = [
            column
            for column
            in preferred_columns
            if column
            in result.columns
        ]

        result = result[
            existing_columns
        ]

        return result

    # --------------------------------------------------------
    # COLLECT EXPIRATION
    # --------------------------------------------------------

    def collect_expiration(
        self,
        expiration: str,
    ) -> pd.DataFrame:

        current_price = (
            self.get_current_price()
        )

        calls, puts = (
            self.fetch_chain(
                expiration
            )
        )

        calls = self.normalize(
            calls,
            current_price,
        )

        puts = self.normalize(
            puts,
            current_price,
        )

        result = pd.concat(
            [
                calls,
                puts,
            ],
            ignore_index=True,
        )

        if result.empty:

            return result

        result.insert(
            0,
            "symbol",
            self.symbol,
        )

        return result

    # --------------------------------------------------------
    # COLLECT ONE DAY
    # --------------------------------------------------------

    def collect_one_day(
        self,
    ) -> dict[str, Any]:

        current_price = (
            self.get_current_price()
        )

        expiration = (
            self.get_nearest_expiration(
                ONE_DAY_MAX_DTE
            )
        )

        if expiration is None:

            return {
                "success": False,

                "symbol":
                    self.symbol,

                "current_price":
                    current_price,

                "expiration":
                    None,

                "DTE":
                    None,

                "rows":
                    0,

                "data":
                    [],

                "error":
                    (
                        "No expiration found "
                        f"within DTE <= "
                        f"{ONE_DAY_MAX_DTE}"
                    ),
            }

        data = (
            self.collect_expiration(
                expiration
            )
        )

        dte = (
            calculate_dte(
                expiration
            )
        )

        return {
            "success": True,

            "symbol":
                self.symbol,

            "current_price":
                current_price,

            "expiration":
                expiration,

            "DTE":
                dte,

            "rows":
                len(data),

            "data":
                data.to_dict(
                    orient="records"
                ),

            "error":
                None,
        }


# ============================================================
# CONSOLE TEST
# ============================================================

def print_summary(
    result: dict[str, Any],
) -> None:

    print()

    print(
        "=" * 70
    )

    print(
        "FREE OPTION BOT - "
        "OPTION COLLECTOR TEST"
    )

    print(
        "=" * 70
    )

    print(
        f"Symbol        : "
        f"{result.get('symbol')}"
    )

    print(
        f"Current Price : "
        f"{result.get('current_price')}"
    )

    print(
        f"Expiration    : "
        f"{result.get('expiration')}"
    )

    print(
        f"DTE           : "
        f"{result.get('DTE')}"
    )

    print(
        f"Rows          : "
        f"{result.get('rows')}"
    )

    print(
        f"Success       : "
        f"{result.get('success')}"
    )

    if result.get("error"):

        print(
            f"Error         : "
            f"{result.get('error')}"
        )

    data = result.get(
        "data",
        [],
    )

    if data:

        df = pd.DataFrame(
            data
        )

        print()

        print(
            "CALL / PUT COUNTS"
        )

        print(
            "-" * 70
        )

        if (
            "option_type"
            in df.columns
        ):

            print(
                df[
                    "option_type"
                ]
                .value_counts()
                .to_string()
            )

        print()

        print(
            "SAMPLE DATA"
        )

        print(
            "-" * 70
        )

        display_columns = [

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

            "theta",
        ]

        display_columns = [
            column
            for column
            in display_columns
            if column
            in df.columns
        ]

        print(
            df[
                display_columns
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

        print()

        # ----------------------------------------------------
        # Greek quality
        # ----------------------------------------------------

        gamma_count = int(
            (
                pd.to_numeric(
                    df["gamma"],
                    errors="coerce",
                )
                > 0
            ).sum()
        )

        delta_count = int(
            (
                pd.to_numeric(
                    df["delta"],
                    errors="coerce",
                )
                != 0
            ).sum()
        )

        vega_count = int(
            (
                pd.to_numeric(
                    df["vega"],
                    errors="coerce",
                )
                > 0
            ).sum()
        )

        print(
            "GREEKS QUALITY"
        )

        print(
            "-" * 70
        )

        print(
            f"Delta rows : "
            f"{delta_count:,}"
        )

        print(
            f"Gamma rows : "
            f"{gamma_count:,}"
        )

        print(
            f"Vega rows  : "
            f"{vega_count:,}"
        )

        print()

        print(
            "⚠️ Greeks are "
            "Black-Scholes model-derived "
            "using Yahoo Finance IV."
        )

    print(
        "=" * 70
    )

    print()


def main():

    collector = OptionCollector(
        DEFAULT_SYMBOL
    )

    result = (
        collector.collect_one_day()
    )

    print_summary(
        result
    )


if __name__ == "__main__":

    main()
