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

# Updated to the real URL after deploying app.py to Streamlit Cloud
_DASHBOARD_URL = os.getenv("STREAMLIT_URL", "https://tu-app.streamlit.app")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _send(text: str) -> bool:
    """POST a message to the Telegram Bot API.

    Uses Markdown parse mode (v1). Special characters in user-supplied text
    (prices, percentages) are left unescaped — Markdown v1 is tolerant.

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

    Message format follows the CLAUDE.md spec:
      🟢/🔴 + 🔥 (scoring 3) header
      Price, Scoring, RSI, ADX, Sector, News sentiment
      Gemini analysis preview (first 300 chars)
      Dashboard deep-link

    Args:
        senal: Enriched signal dict.  Required keys: ticker, tipo_evento,
               precio_cierre, scoring.  All other keys fall back to 'N/A'/0.

    Returns:
        True if the message was delivered successfully.
    """
    ticker:    str   = senal.get("ticker", "???")
    tipo:      str   = senal.get("tipo_evento", "")
    scoring:   int   = int(senal.get("scoring", 1))
    precio:    float = float(senal.get("precio_cierre", 0))
    rsi:       float = float(senal.get("rsi_14", 0))
    adx:       float = float(senal.get("adx_14", 0))
    sector:    str   = senal.get("sector", "N/A") or "N/A"
    sentiment: str   = senal.get("news_sentiment", "N/A") or "N/A"
    analisis:  str   = senal.get("analisis_gemini") or "Sin análisis disponible."
    fecha:     str   = str(senal.get("fecha_evento", datetime.now().date()))

    emoji      = "🟢" if tipo == "golden_cross" else "🔴"
    fire       = " 🔥" if scoring == 3 else ""
    tipo_label = tipo.replace("_", " ").upper()

    # Truncate Gemini analysis to 300 chars to respect Telegram's 4096-char limit
    analisis_preview = analisis[:300] + ("..." if len(analisis) > 300 else "")

    text = (
        f"{emoji} *{tipo_label}{fire}* — ${ticker}\n"
        f"📅 {fecha}\n\n"
        f"💰 *Precio:* ${precio:.2f}\n"
        f"⭐ *Scoring:* {scoring}/3\n"
        f"📊 *RSI:* {rsi:.0f} | *ADX:* {adx:.0f}\n"
        f"🏢 *Sector:* {sector}\n"
        f"📰 *Noticias:* {sentiment}\n\n"
        f"🤖 *Análisis IA:*\n{analisis_preview}\n\n"
        f"🔗 [Ver gráfico en dashboard]({_DASHBOARD_URL}?ticker={ticker})"
    )

    logger.info("Sending Telegram alert: %s %s scoring=%d", ticker, tipo, scoring)
    return _send(text)


def enviar_resumen_diario(n_senales: int, n_tickers: int, duracion: float) -> None:
    """Send a daily pipeline execution summary to Telegram.

    Called at the end of main.py regardless of whether signals were found.

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
