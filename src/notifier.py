"""Notifier: sends Telegram alerts for detected signals and daily summaries."""

import logging
import os
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID:   str = os.getenv("TELEGRAM_CHAT_ID", "")
_TG_API:    str = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"

_DASHBOARD_URL = os.getenv("STREAMLIT_URL", "https://tu-app.streamlit.app")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _send(text: str) -> bool:
    """POST a message to the Telegram Bot API.

    Args:
        text: Message body in Telegram Markdown format.

    Returns:
        True if Telegram confirmed delivery (ok=true), False on any error.
    """
    if not _BOT_TOKEN or _BOT_TOKEN == "PENDIENTE_CONFIGURAR":
        logger.warning("TELEGRAM_BOT_TOKEN not configured — message not sent")
        return False
    if not _CHAT_ID or _CHAT_ID == "PENDIENTE_CONFIGURAR":
        logger.warning("TELEGRAM_CHAT_ID not configured — message not sent")
        return False

    try:
        resp = requests.post(
            _TG_API,
            json={
                "chat_id":    _CHAT_ID,
                "text":       text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        resp.raise_for_status()
        result: dict = resp.json()
        if not result.get("ok"):
            logger.error("Telegram API returned ok=false: %s", result)
            return False
        logger.info("Telegram message sent (message_id=%s)", result["result"]["message_id"])
        return True
    except requests.exceptions.Timeout:
        logger.error("Telegram request timed out after 10 s")
        return False
    except Exception as exc:
        logger.error("Failed to send Telegram message: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enviar_alerta(senal: dict) -> bool:
    """Send a signal alert message to Telegram.

    Skips sending (and logs the reason) when senal['descartar'] is True.

    Args:
        senal: Enriched signal dict.  Required keys: ticker, tipo_evento,
               precio_cierre, scoring.  All other keys fall back gracefully.

    Returns:
        True if the message was delivered successfully, False otherwise.
    """
    # --- Skip discarded signals ---
    if senal.get("descartar"):
        ratio = senal.get("ratio_rr", 0)
        logger.info(
            "DESCARTADO %s — ratio R/R %.2f < 1.5, Telegram NO enviado",
            senal.get("ticker", "?"), ratio,
        )
        return False

    ticker:    str   = senal.get("ticker", "???")
    tipo:      str   = senal.get("tipo_evento", "")
    scoring:   int   = int(senal.get("scoring", 1))
    precio:    float = float(senal.get("precio_cierre", 0))
    rsi:       float = float(senal.get("rsi_14", 0))
    adx:       float = float(senal.get("adx_14", 0))
    sector:    str   = senal.get("sector", "N/A") or "N/A"
    sentiment: str   = senal.get("news_sentiment", "N/A") or "N/A"
    fecha:     str   = str(senal.get("fecha_evento", datetime.now().date()))

    # Separate analysis points
    tecnico     = senal.get("analisis_tecnico",     "") or ""
    fundamental = senal.get("analisis_fundamental", "") or ""
    riesgo      = senal.get("analisis_riesgo",      "") or ""

    # Trading levels
    entrada  = float(senal.get("precio_entrada",  precio))
    objetivo = float(senal.get("precio_objetivo", precio * 1.10))
    stop     = float(senal.get("precio_stop",     precio * 0.95))
    ratio_rr = float(senal.get("ratio_rr",        0))

    pct_objetivo = ((objetivo - entrada) / entrada * 100) if entrada else 0
    pct_stop     = ((stop     - entrada) / entrada * 100) if entrada else 0

    # Emojis
    emoji         = "🟢" if tipo == "golden_cross" else "🔴"
    tipo_label    = tipo.replace("_", " ").upper()
    scoring_emoji = "🔥" if scoring == 3 else ("⚡" if scoring == 2 else "·")

    text = (
        f"{emoji} *{tipo_label}* {scoring_emoji} — ${ticker}\n"
        f"\n"
        f"📅 {fecha}\n"
        f"💰 Precio: ${precio:.2f}\n"
        f"⭐ Scoring: {scoring}/3\n"
        f"📊 RSI: {rsi:.0f} | ADX: {adx:.0f}\n"
        f"🏢 Sector: {sector}\n"
        f"📰 Noticias: {sentiment}\n"
        f"\n"
        f"🤖 *Análisis IA:*\n"
        f"1️⃣ {tecnico}\n"
        f"2️⃣ {fundamental}\n"
        f"3️⃣ {riesgo}\n"
        f"\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📈 *Entrada:*   ${entrada:.2f}\n"
        f"🎯 *Objetivo:*  ${objetivo:.2f}  ({pct_objetivo:+.1f}%)\n"
        f"🛑 *Stop Loss:* ${stop:.2f}  ({pct_stop:+.1f}%)\n"
        f"⚖️ *Ratio R/R:* {ratio_rr:.1f}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"\n"
        f"🔗 [Ver gráfico]({_DASHBOARD_URL}?ticker={ticker})"
    )

    logger.info("Sending Telegram alert: %s %s scoring=%d rr=%.2f", ticker, tipo, scoring, ratio_rr)
    return _send(text)


def enviar_resumen_diario(n_senales: int, n_tickers: int, duracion: float) -> None:
    """Send a daily pipeline execution summary to Telegram.

    Args:
        n_senales: Number of signals that passed all filters.
        n_tickers: Number of tickers successfully analyzed.
        duracion:  Total pipeline wall-clock time in seconds.
    """
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    if n_senales > 0:
        status_line = f"✅ *{n_senales} señal(es)* detectada(s)"
    else:
        status_line = "💤 Sin señales hoy — mercado sin cruces que superen los filtros"

    text = (
        f"📊 *Resumen diario — {ahora}*\n\n"
        f"{status_line}\n"
        f"📈 Tickers analizados: {n_tickers}\n"
        f"⏱ Duración pipeline: {duracion:.1f} s\n\n"
        f"🔗 [Ver dashboard]({_DASHBOARD_URL})"
    )

    logger.info("Sending daily summary: %d signals / %d tickers / %.1fs", n_senales, n_tickers, duracion)
    _send(text)
