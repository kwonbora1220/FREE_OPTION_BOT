"""
FREE OPTION BOT
Option Collector

Yahoo Finance / yfinance를 이용해서
미국 주식 옵션체인을 수집한다.

주의:
- 실시간 옵션 데이터가 아니다.
- Yahoo Finance에서 제공하는 데이터를 사용한다.
- 기관/고래 거래를 직접 식별하지 않는다.
"""

from __future__ import annotations

import json
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
# UTILITIES
# ============================================================

def safe_float(value: Any) -> float | None:
    """숫자로 변환할 수 없는 값을 None으로 반환한다."""

    if value is None:
        return None

    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def calculate_dte(expiration: str) -> int:
    """만기일까지 남은 날짜를 계산한다."""

    expiration_date = datetime.strptime(
        expiration,
        "%Y-%m-%d",
    ).date()

    today = date.today()

    return (expiration_date - today).days


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
        self.symbol = symbol.upper().strip()

        if not self.symbol:
            raise ValueError("Symbol is empty.")

        self.ticker = yf.Ticker(self.symbol)

    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    def get_current_price(self) -> float | None:
        """
        현재 가격을 가져온다.

        Yahoo Finance 데이터 특성상 일부 필드는
        상황에 따라 None일 수 있으므로 fallback을 사용한다.
        """

        try:
            fast_info = self.ticker.fast_info

            price = fast_info.get("lastPrice")

            price = safe_float(price)

            if price is not None:
                return price

        except Exception:
            pass

        try:
            history = self.ticker.history(
                period="1d",
                interval="1m",
            )

            if history is not None and not history.empty:

                close = history["Close"].dropna()

                if not close.empty:
                    return safe_float(close.iloc[-1])

        except Exception:
            pass

        try:
            history = self.ticker.history(
                period="5d",
                interval="1d",
            )

            if history is not None and not history.empty:

                close = history["Close"].dropna()

                if not close.empty:
                    return safe_float(close.iloc[-1])

        except Exception:
            pass

        return None

    # --------------------------------------------------------
    # EXPIRATIONS
    # --------------------------------------------------------

    def get_expirations(self) -> list[str]:
        """
        Yahoo Finance에서 제공하는 옵션 만기 목록.
        """

        expirations = self.ticker.options

        if not expirations:
            return []

        return list(expirations)

    # --------------------------------------------------------
    # NEAREST EXPIRATION
    # --------------------------------------------------------

    def get_nearest_expiration(
        self,
        max_dte: int = ONE_DAY_MAX_DTE,
    ) -> str | None:
        """
        오늘부터 max_dte 이내의 가장 가까운 만기를 찾는다.

        예:
            max_dte=1

        오늘 만기 또는 다음 거래일 만기 중
        Yahoo가 제공하는 가장 가까운 만기를 반환한다.
        """

        expirations = self.get_expirations()

        if not expirations:
            return None

        today = date.today()

        candidates = []

        for expiration in expirations:

            try:
                expiration_date = datetime.strptime(
                    expiration,
                    "%Y-%m-%d",
                ).date()

            except ValueError:
                continue

            dte = (expiration_date - today).days

            if 0 <= dte <= max_dte:
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
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        지정 만기의 CALL / PUT 옵션체인을 가져온다.
        """

        time.sleep(REQUEST_DELAY_SECONDS)

        chain = self.ticker.option_chain(expiration)

        calls = chain.calls.copy()
        puts = chain.puts.copy()

        calls["option_type"] = "CALL"
        puts["option_type"] = "PUT"

        calls["expiration"] = expiration
        puts["expiration"] = expiration

        return calls, puts

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    def normalize(
        self,
        df: pd.DataFrame,
        current_price: float | None,
    ) -> pd.DataFrame:
        """
        내부 표준 옵션 데이터 구조로 변환한다.
        """

        if df is None or df.empty:
            return pd.DataFrame()

        result = df.copy()

        # 필수 컬럼이 없으면 None으로 생성
        for column in REQUIRED_OPTION_COLUMNS:

            if column not in result.columns:
                result[column] = None

        if "option_type" not in result.columns:
            result["option_type"] = None

        if "expiration" not in result.columns:
            result["expiration"] = None

        # 숫자형 컬럼
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

        # DTE
        result["DTE"] = result["expiration"].apply(
            lambda x: calculate_dte(x)
            if isinstance(x, str)
            else None
        )

        # Underlying price
        result["underlying_price"] = current_price

        # Mid price
        result["midPrice"] = (
            result["bid"] + result["ask"]
        ) / 2

        # Bid / Ask가 없는 경우 lastPrice fallback
        result["midPrice"] = result["midPrice"].where(
            result["midPrice"].notna(),
            result["lastPrice"],
        )

        # Premium proxy
        #
        # 옵션 계약 1개 = 100주
        result["premium"] = (
            result["midPrice"]
            * result["volume"]
            * 100
        )

        # Moneyness
        if current_price is not None:

            result["moneyness"] = (
                result["strike"] / current_price
            )

        else:

            result["moneyness"] = None

        # 내부 표준 컬럼 순서
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
            "premium",
            "underlying_price",
            "moneyness",
        ]

        existing_columns = [
            column
            for column in preferred_columns
            if column in result.columns
        ]

        result = result[
            existing_columns
        ]

        return result

    # --------------------------------------------------------
    # COLLECT ONE EXPIRATION
    # --------------------------------------------------------

    def collect_expiration(
        self,
        expiration: str,
    ) -> pd.DataFrame:
        """
        특정 만기의 CALL + PUT 데이터를 하나로 합친다.
        """

        current_price = self.get_current_price()

        calls, puts = self.fetch_chain(
            expiration
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
        """
        1일 옵션 데이터를 수집한다.
        """

        current_price = self.get_current_price()

        expiration = self.get_nearest_expiration(
            ONE_DAY_MAX_DTE
        )

        if expiration is None:

            return {
                "success": False,
                "symbol": self.symbol,
                "current_price": current_price,
                "expiration": None,
                "DTE": None,
                "rows": 0,
                "data": [],
                "error": (
                    "No expiration found "
                    f"within DTE <= {ONE_DAY_MAX_DTE}"
                ),
            }

        data = self.collect_expiration(
            expiration
        )

        dte = calculate_dte(
            expiration
        )

        return {
            "success": True,
            "symbol": self.symbol,
            "current_price": current_price,
            "expiration": expiration,
            "DTE": dte,
            "rows": len(data),
            "data": data.to_dict(
                orient="records"
            ),
            "error": None,
        }


# ============================================================
# CONSOLE TEST
# ============================================================

def print_summary(
    result: dict[str, Any],
) -> None:

    print()
    print("=" * 70)
    print("FREE OPTION BOT - OPTION COLLECTOR TEST")
    print("=" * 70)

    print(
        f"Symbol        : {result.get('symbol')}"
    )

    print(
        f"Current Price : {result.get('current_price')}"
    )

    print(
        f"Expiration    : {result.get('expiration')}"
    )

    print(
        f"DTE           : {result.get('DTE')}"
    )

    print(
        f"Rows          : {result.get('rows')}"
    )

    print(
        f"Success       : {result.get('success')}"
    )

    if result.get("error"):
        print(
            f"Error         : {result.get('error')}"
        )

    data = result.get("data", [])

    if data:

        df = pd.DataFrame(data)

        print()
        print("CALL / PUT COUNTS")
        print("-" * 70)

        if "option_type" in df.columns:

            print(
                df["option_type"]
                .value_counts()
                .to_string()
            )

        print()
        print("SAMPLE DATA")
        print("-" * 70)

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
        ]

        display_columns = [
            column
            for column in display_columns
            if column in df.columns
        ]

        print(
            df[display_columns]
            .head(10)
            .to_string(index=False)
        )

    print("=" * 70)
    print()


def main():

    collector = OptionCollector(
        DEFAULT_SYMBOL
    )

    result = collector.collect_one_day()

    print_summary(result)

    # JSON 저장은 하지 않고 현재 단계에서는
    # 콘솔 결과 검증만 수행한다.


if __name__ == "__main__":
    main()
