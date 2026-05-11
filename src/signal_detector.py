"""Signal detector: Golden Cross / Death Cross with cascade filter system."""

import logging
from datetime import date

import pandas as pd

from src.config import (
    ADX_MIN,
    RSI_MAX_GOLDEN,
    RSI_MIN_DEATH,
    VOL_MIN_RATIO,
    calcular_scoring,
)

logger = logging.getLogger(__name__)


def detectar_cruce(df: pd.DataFrame, ticker: str) -> dict | None:
    """Detect a Golden Cross or Death Cross on the most recent session.

    Applies a cascade of three filters to suppress false signals:

      Condition (Golden Cross):  prev.SMA_50 <= prev.SMA_200  AND  today.SMA_50 > today.SMA_200
      Condition (Death Cross):   prev.SMA_50 >= prev.SMA_200  AND  today.SMA_50 < today.SMA_200

      Filter 1 — Trend strength:  ADX_14 >= 25  (rejects sideways markets)
      Filter 2 — Volume confirms: Volume >= VOL_MA20 * 1.2  (rejects low-conviction moves)
      Filter 3 — RSI not extreme:
          Golden Cross: RSI_14 <= 75  (rejects overbought traps)
          Death Cross:  RSI_14 >= 25  (rejects oversold traps)

    Args:
        df: DataFrame with columns Close, Volume, SMA_50, SMA_200, RSI_14, ADX_14, ATR_14, VOL_MA20.
            Must contain at least 2 rows.
        ticker: Ticker symbol used to populate the result dict.

    Returns:
        Dict with all database fields if the signal passes every filter, otherwise None.
    """
    if len(df) < 2:
        logger.warning("%s: not enough rows to detect a cross (need ≥2, got %d)", ticker, len(df))
        return None

    today = df.iloc[-1]
    prev = df.iloc[-2]

    sma50_today: float = float(today["SMA_50"])
    sma200_today: float = float(today["SMA_200"])
    sma50_prev: float = float(prev["SMA_50"])
    sma200_prev: float = float(prev["SMA_200"])

    # --- Detect cross type ---
    golden = sma50_prev <= sma200_prev and sma50_today > sma200_today
    death = sma50_prev >= sma200_prev and sma50_today < sma200_today

    if not golden and not death:
        logger.debug("%s: no cross detected", ticker)
        return None

    tipo_evento = "golden_cross" if golden else "death_cross"

    adx: float = float(today["ADX_14"])
    atr: float = float(today["ATR_14"])
    volume: float = float(today["Volume"])
    vol_ma20: float = float(today["VOL_MA20"])
    rsi: float = float(today["RSI_14"])
    precio_cierre: float = float(today["Close"])

    # --- Filter 1: trend strength (anti-whipsaw) ---
    if adx < ADX_MIN:
        logger.info(
            "%s: %s discarded — ADX %.1f < %d (sideways market)",
            ticker, tipo_evento, adx, ADX_MIN,
        )
        return None

    # --- Filter 2: volume confirmation ---
    volumen_relativo: float = round(volume / vol_ma20, 2) if vol_ma20 > 0 else 0.0
    if volume < vol_ma20 * VOL_MIN_RATIO:
        logger.info(
            "%s: %s discarded — volume ratio %.2fx < %.1fx (no conviction)",
            ticker, tipo_evento, volumen_relativo, VOL_MIN_RATIO,
        )
        return None

    # --- Filter 3: RSI not in extreme opposite zone ---
    if golden and rsi > RSI_MAX_GOLDEN:
        logger.info(
            "%s: golden_cross discarded — RSI %.1f > %d (overbought trap)",
            ticker, rsi, RSI_MAX_GOLDEN,
        )
        return None

    if death and rsi < RSI_MIN_DEATH:
        logger.info(
            "%s: death_cross discarded — RSI %.1f < %d (oversold trap)",
            ticker, rsi, RSI_MIN_DEATH,
        )
        return None

    # --- Scoring ---
    dist_sma_pct: float = round(((sma50_today - sma200_today) / sma200_today) * 100, 4)
    scoring: int = calcular_scoring(dist_sma_pct)

    # --- Stop Loss Level ---
    stop_loss: float = round(precio_cierre - (2 * atr), 2) if golden else round(precio_cierre + (2 * atr), 2)

    fecha_evento: date = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]

    senal: dict = {
        "fecha_evento": fecha_evento,
        "ticker": ticker,
        "tipo_evento": tipo_evento,
        "precio_cierre": round(precio_cierre, 2),
        "sma_50": round(sma50_today, 2),
        "sma_200": round(sma200_today, 2),
        "rsi_14": round(rsi, 2),
        "adx_14": round(adx, 2),
        "atr_14": round(atr, 2),
        "stop_loss": stop_loss,
        "volumen_relativo": volumen_relativo,
        "dist_sma_pct": dist_sma_pct,
        "scoring": scoring,
    }

    logger.info(
        "%s: %s DETECTED — scoring %d/3 | dist %.2f%% | RSI %.1f | ADX %.1f | ATR %.2f | SL %.2f | vol %.2fx",
        ticker, tipo_evento, scoring, dist_sma_pct, rsi, adx, atr, stop_loss, volumen_relativo,
    )
    return senal


def detectar_en_todos(datos_tickers: list[dict]) -> list[dict]:
    """Run cross detection across all processed tickers.

    Args:
        datos_tickers: List of dicts {'ticker': str, 'df': pd.DataFrame} as returned
                       by data_fetcher.procesar_todos_tickers.

    Returns:
        List of signal dicts for tickers where a cross passed all filters.
        May be empty if no qualifying signals exist today.
    """
    senales: list[dict] = []

    for item in datos_tickers:
        ticker: str = item["ticker"]
        df: pd.DataFrame = item["df"]

        try:
            senal = detectar_cruce(df, ticker)
        except Exception as exc:
            logger.error("%s: unhandled error in detectar_cruce: %s", ticker, exc)
            continue

        if senal is not None:
            senales.append(senal)

    logger.info(
        "Detection complete — %d signal(s) found out of %d ticker(s) processed",
        len(senales), len(datos_tickers),
    )
    return senales
