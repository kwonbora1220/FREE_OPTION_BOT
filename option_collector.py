"""
FREE OPTION BOT
OPTION COLLECTOR

Yahoo Finance / yfinance 옵션체인 수집.

중요:
- 만기 판단은 미국 동부시간 기준
- 이미 지나간 만기는 제외
- 주말에는 다음 유효 만기 선택
"""

from __future__ import annotations

import time

from datetime import date, datetime
from typing import Any

from zoneinfo import ZoneInfo

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


US_EASTERN = ZoneInfo(
    "America/New_York"
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
    # US DATE
    # ========================================================

    @staticmethod
    def get_us_date() -> date:

        return datetime.now(
            US_EASTERN
        ).date()

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

                return float(
                    price
                )

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

            return list(
                options
            )

        except Exception:

            return []

    # ========================================================
    # PARSE EXPIRATION
    # ========================================================

    @staticmethod
    def parse_expiration(
        expiration: str,
    ) -> date | None:

        try:

            return datetime.strptime(
                expiration,
                "%Y-%m-%d",
            ).date()

        except (
            ValueError,
            TypeError,
        ):

            return None

    # ========================================================
    # NEXT VALID EXPIRATION
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

        today = (
            self.get_us_date()
        )

        candidates = []

        for expiration in expirations:

            exp_date = (
                self.parse_expiration(
                    expiration
                )
            )

            if exp_date is None:

                continue

            # ------------------------------------------------
            # 이미 지난 만기 제거
            # ------------------------------------------------

            if exp_date < today:

                continue

            dte = (
                exp_date
                - today
            ).days

            # ------------------------------------------------
            # 정상적인 평일에는
            # 오늘/내일 범위 우선
            # ------------------------------------------------

            if (
                dte <= max_dte
            ):

                candidates.append(
                    (
                        dte,
                        expiration,
                    )
                )

        if candidates:

            candidates.sort(
                key=lambda x: x[0]
            )

            return candidates[0][1]

        # ----------------------------------------------------
        # 주말 / 휴일 대응
        #
        # 예:
        # 한국 토요일
        # 미국 금요일
        #
        # 금요일 만기는 이미 exp_date < today
        # 다음 월요일을 찾아야 한다.
        #
        # max_dte 때문에 월요일이 제외될 수 있으므로
        # 다음 유효 만기를 fallback으로 선택.
        # ----------------------------------------------------

        future = []

        for expiration in expirations:

            exp_date = (
                self.parse_expiration(
                    expiration
                )
            )

            if exp_date is None:

                continue

            if exp_date < today:

                continue

            future.append(
                (
                    exp_date,
                    expiration,
                )
            )

        if not future:

            return None

        future.sort(
            key=lambda x: x[0]
        )

        return future[0][1]

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

        calls[
            "option_type"
        ] = "CALL"

        puts[
            "option_type"
        ] = "PUT"

        calls[
            "expiration"
        ] = expiration

        puts[
            "expiration"
        ] = expiration

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
                        "No valid future "
                        "expiration found."
                    ),
            }

        data = (
            self.collect_expiration(
                expiration
            )
        )

        expiration_date = (
            self.parse_expiration(
                expiration
            )
        )

        today = (
            self.get_us_date()
        )

        if expiration_date:

            dte = (
                expiration_date
                - today
            ).days

        else:

            dte = None

        return {

            "success":
                not data.empty,

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
                None
                if not data.empty
                else "Option chain empty.",
        }


# ============================================================
# TEST
# ============================================================

def main():

    collector = OptionCollector(
        DEFAULT_SYMBOL
    )

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
        f"US Eastern Date : "
        f"{collector.get_us_date()}"
    )

    result = (
        collector.collect_one_day()
    )

    print()

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

    print()

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()
