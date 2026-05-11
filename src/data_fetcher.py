"""Data fetcher: downloads OHLCV data from yfinance and computes technical indicators."""

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pandas_ta as ta
import yfinance as yf

logger = logging.getLogger(__name__)

_PERIOD = "2y"
_INTERVAL = "1d"
_MIN_ROWS = 220
_MAX_GAP_DAYS = 7
_CRITICAL_COLS = ["Close", "High", "Low", "Volume"]


def descargar_y_validar(ticker: str) -> pd.DataFrame | None:
    """Download 2 years of daily OHLCV data from yfinance and validate its integrity.

    Validation rules applied in order:
    1. Fill small NaN gaps with forward-fill then backward-fill.
    2. Require at least 220 rows (enough warm-up for SMA_200 + buffer).
    3. Reject if NaN remain in Close / High / Low / Volume after fill.
    4. Reject if any gap between consecutive sessions exceeds 7 calendar days.

    Args:
        ticker: Ticker symbol (e.g. 'AAPL').

    Returns:
        Validated OHLCV DataFrame indexed by date, or None on any failure.
    """
    try:
        raw: pd.DataFrame = yf.download(
            ticker,
            period=_PERIOD,
            interval=_INTERVAL,
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        logger.error("%s: yfinance download raised: %s", ticker, exc)
        return None

    if raw is None or raw.empty:
        logger.warning("%s: empty response from yfinance", ticker)
        return None

    # yfinance ≥1.0 may return a MultiIndex (column, ticker) for single tickers
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.copy()

    # --- Fill small NaN gaps ---
    df[_CRITICAL_COLS] = df[_CRITICAL_COLS].ffill().bfill()

    # --- Validate minimum row count ---
    if len(df) < _MIN_ROWS:
        logger.warning(
            "%s: only %d rows available (minimum required: %d)",
            ticker, len(df), _MIN_ROWS,
        )
        return None

    # --- Validate no remaining NaN ---
    if df[_CRITICAL_COLS].isnull().any().any():
        logger.warning("%s: NaN persist in critical columns after forward/backward fill", ticker)
        return None

    # --- Validate no session gap > 7 calendar days ---
    date_diffs = df.index.to_series().diff().dropna()
    max_gap = date_diffs.max()
    if pd.notna(max_gap) and max_gap.days > _MAX_GAP_DAYS:
        logger.warning(
            "%s: price series has a gap of %d calendar days (max allowed: %d)",
            ticker, max_gap.days, _MAX_GAP_DAYS,
        )
        return None

    logger.debug("%s: validated OK (%d rows)", ticker, len(df))
    return df


def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Append SMA_50, SMA_200, RSI_14, ADX_14, ATR_14 and VOL_MA20 columns to the DataFrame.

    All calculations use pandas_ta. Rows produced during the indicator warm-up
    period (the first ~200 candles) are dropped, so the returned DataFrame only
    contains rows where every indicator has a valid value.

    Args:
        df: Validated OHLCV DataFrame from descargar_y_validar.

    Returns:
        DataFrame with six additional indicator columns; NaN rows removed.
    """
    df = df.copy()

    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["SMA_200"] = ta.sma(df["Close"], length=200)
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx_df is not None and "ADX_14" in adx_df.columns:
        df["ADX_14"] = adx_df["ADX_14"]
    else:
        # Defensive fallback: try the first column that starts with 'ADX'
        if adx_df is not None:
            adx_col = next((c for c in adx_df.columns if c.startswith("ADX")), None)
            df["ADX_14"] = adx_df[adx_col] if adx_col else float("nan")
        else:
            df["ADX_14"] = float("nan")

    # Calculate ATR_14 for dynamic stop loss
    df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    df["VOL_MA20"] = ta.sma(df["Volume"], length=20)

    indicator_cols = ["SMA_50", "SMA_200", "RSI_14", "ADX_14", "ATR_14", "VOL_MA20"]
    df.dropna(subset=indicator_cols, inplace=True)

    return df


def _procesar_ticker(ticker: str) -> dict | None:
    """Download, validate and compute indicators for a single ticker.

    Sleeps 0.3–0.8 s before the network call to avoid saturating Yahoo Finance.

    Args:
        ticker: Ticker symbol.

    Returns:
        Dict {'ticker': str, 'df': pd.DataFrame} on success, or None on failure.
    """
    time.sleep(random.uniform(0.3, 0.8))

    df = descargar_y_validar(ticker)
    if df is None:
        return None

    try:
        df = calcular_indicadores(df)
    except Exception as exc:
        logger.error("%s: error computing indicators: %s", ticker, exc)
        return None

    if df.empty:
        logger.warning("%s: DataFrame is empty after indicator calculation", ticker)
        return None

    return {"ticker": ticker, "df": df}


def procesar_todos_tickers(tickers: list[str]) -> list[dict]:
    """Download and process all tickers in parallel using a thread pool (max 5 workers).

    Each worker adds a random 0.3–0.8 s delay before hitting Yahoo Finance.
    Tickers that fail any step are omitted from the result and logged.

    Args:
        tickers: List of ticker symbols to process.

    Returns:
        List of dicts with keys 'ticker' and 'df', one per successful ticker.
    """
    results: list[dict] = []
    failed: list[str] = []

    logger.info("Starting parallel download — %d tickers, max_workers=5", len(tickers))

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(_procesar_ticker, t): t for t in tickers}

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error("%s: unhandled exception in worker thread: %s", ticker, exc)
                failed.append(ticker)
                continue

            if result is None:
                failed.append(ticker)
            else:
                results.append(result)

    logger.info(
        "Download complete — success: %d/%d | failed (%d): %s",
        len(results), len(tickers),
        len(failed), failed or "none",
    )
    return results
