"""Tracker: weekly follow-up for signals older than 30 days.

Fetches the actual prices 7 and 30 days after each signal and updates
retorno_7d / retorno_30d / senal_ok in the database.

Run manually or as a second PythonAnywhere task on Sundays at 09:00 UTC.
"""

import logging
from datetime import datetime, timedelta

import yfinance as yf

from src.database import actualizar_seguimiento, get_senales_para_seguimiento

logger = logging.getLogger(__name__)


def _precio_en_fecha(ticker: str, fecha: datetime) -> float | None:
    """Return the closing price on or just after the given date.

    Downloads 5 days of data around the target date to tolerate weekends
    and market holidays.

    Args:
        ticker: Ticker symbol.
        fecha:  Target date (datetime).

    Returns:
        Closing price as float, or None if data is unavailable.
    """
    start = fecha.strftime("%Y-%m-%d")
    end   = (fecha + timedelta(days=5)).strftime("%Y-%m-%d")

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            logger.warning("%s: no price data between %s and %s", ticker, start, end)
            return None
        return float(df["Close"].iloc[0])
    except Exception as exc:
        logger.error("%s: yfinance error for date %s: %s", ticker, start, exc)
        return None


def run() -> None:
    """Fetch and update follow-up prices for all trackable signals."""
    logger.info("=== Tracker iniciado ===")

    pendientes = get_senales_para_seguimiento()
    if pendientes.empty:
        logger.info("Sin señales pendientes de seguimiento")
        return

    logger.info("%d señal(es) pendientes de seguimiento", len(pendientes))
    actualizadas = 0
    errores = 0

    for _, row in pendientes.iterrows():
        signal_id    = int(row["id"])
        ticker       = str(row["ticker"])
        fecha_evento = row["fecha_evento"]

        if hasattr(fecha_evento, "to_pydatetime"):
            fecha_evento = fecha_evento.to_pydatetime()

        fecha_7d  = fecha_evento + timedelta(days=7)
        fecha_30d = fecha_evento + timedelta(days=30)

        logger.info("[%s] id=%d — buscando precios para %s y %s",
                    ticker, signal_id,
                    fecha_7d.strftime("%Y-%m-%d"),
                    fecha_30d.strftime("%Y-%m-%d"))

        precio_7d  = _precio_en_fecha(ticker, fecha_7d)
        precio_30d = _precio_en_fecha(ticker, fecha_30d)

        if precio_7d is None or precio_30d is None:
            logger.warning("[%s] id=%d — no se pudo obtener uno o ambos precios, saltando",
                           ticker, signal_id)
            errores += 1
            continue

        actualizar_seguimiento(signal_id, precio_7d, precio_30d)
        actualizadas += 1

    logger.info("=== Tracker finalizado — actualizadas=%d errores=%d ===",
                actualizadas, errores)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run()
