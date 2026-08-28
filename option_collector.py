"""
FREE OPTION BOT
OPTION COLLECTOR

Yahoo Finance / yfinance에서
옵션체인을 수집한다.

Collector는 데이터 수집만 담당하고
정규화 및 Greeks 계산은 normalizer.py에 위임한다.
"""

from __future__ import annotations

import time

from datetime import date, datetime
from typing import Any

import pandas as pd
import yfinance as yf

from config import (
    DEFAULT_SYMBOL,
    ONE_DAY_MAX_DTE,
    REQUEST_DELAY_SECONDS,
)

from normalizer import (
    normalize_options,
)


class OptionCollector:

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

    # ========================================================
    # CURRENT PRICE
    # ========================================================

    def get_current_price(
        self,
    ) -> float | None:

        try:

            fast_info = (
                self.ticker.fast_info
            )

            price = fast_info.get(
                "lastPrice"
            )

            if price is not None:

                return float(price)

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
                    history["Close"]
                    .dropna()
                )

                if not close.empty:

                    return float(
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
                    history["Close"]
                    .dropna()
                )

                if not close.empty:

                    return float(
                        close.iloc[-1]
                    )

        except Exception:
            pass

        return None

    # ========================================================
    # EXPIRATIONS
    # ========================================================

    def get_expirations(
        self,
    ) -> list[str]:

        try:

            options = (
                self.ticker.options
            )

            if not options:

                return []

            return list(options)

        except Exception:

            return []

    # ========================================================
    # NEAREST EXPIRATION
    # ========================================================

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

                exp_date = datetime.strptime(
                    expiration,
                    "%Y-%m-%d",
                ).date()

            except ValueError:

                continue

            dte = (
                exp_date
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

    # ========================================================
    # FETCH CHAIN
    # ========================================================

    def fetch_chain(
        self,
        expiration: str,
    ) -> pd.DataFrame:

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

        chain = (
            self.ticker.option_chain(
                expiration
            )
        )

        calls = (
            chain.calls.copy()
        )

        puts = (
            chain.puts.copy()
        )

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

        return pd.concat(
            [
                calls,
                puts,
            ],
            ignore_index=True,
        )

    # ========================================================
    # COLLECT EXPIRATION
    # ========================================================

    def collect_expiration(
        self,
        expiration: str,
    ) -> pd.DataFrame:

        current_price = (
            self.get_current_price()
        )

        raw = (
            self.fetch_chain(
                expiration
            )
        )

        normalized = (
            normalize_options(
                raw,
                current_price,
            )
        )

        if normalized.empty:

            return normalized

        normalized.insert(
            0,
            "symbol",
            self.symbol,
        )

        return normalized

    # ========================================================
    # COLLECT ONE DAY
    # ========================================================

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
                "symbol": self.symbol,
                "current_price":
                    current_price,
                "expiration": None,
                "DTE": None,
                "rows": 0,
                "data": [],
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
            datetime.strptime(
                expiration,
                "%Y-%m-%d",
            ).date()
            - date.today()
        ).days

        return {
            "success": True,
            "symbol": self.symbol,
            "current_price":
                current_price,
            "expiration":
                expiration,
            "DTE": dte,
            "rows":
                len(data),
            "data":
                data.to_dict(
                    orient="records"
                ),
            "error": None,
        }


# ============================================================
# TEST
# ============================================================

def main():

    collector = OptionCollector(
        DEFAULT_SYMBOL
    )

    result = (
        collector.collect_one_day()
    )

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
        f"{result['symbol']}"
    )

    print(
        f"Current Price : "
        f"{result['current_price']}"
    )

    print(
        f"Expiration    : "
        f"{result['expiration']}"
    )

    print(
        f"DTE           : "
        f"{result['DTE']}"
    )

    print(
        f"Rows          : "
        f"{result['rows']}"
    )

    print(
        f"Success       : "
        f"{result['success']}"
    )

    if result["data"]:

        df = pd.DataFrame(
            result["data"]
        )

        print()

        print(
            "CALL / PUT COUNTS"
        )

        print(
            "-" * 70
        )

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

        columns = [
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

        print(
            df[
                [
                    c
                    for c in columns
                    if c in df.columns
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

        print()

        gamma_rows = int(
            (
                df["gamma"]
                > 0
            ).sum()
        )

        print(
            f"Gamma rows > 0 : "
            f"{gamma_rows}"
        )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()
