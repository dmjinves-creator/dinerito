"""AI engine: generates signal analysis using Gemini 2.5 Flash with Google Search grounding.

Uses the new google-genai SDK (google.generativeai is deprecated as of 2025).
Model: gemini-2.5-flash (replaces spec's gemini-1.5-pro, which is no longer available).
Grounding via google_search tool; falls back gracefully to non-grounded generation.
"""

import logging
import time

from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level client — initialised once at import time
# ---------------------------------------------------------------------------
_client = genai.Client(api_key=GEMINI_API_KEY)

_MODEL = "gemini-2.5-flash"

_MAX_RETRIES  = 3
_RETRY_WAIT   = 60   # seconds base wait; doubles on each retry (exponential)

# ---------------------------------------------------------------------------
# Prompts (exact text from CLAUDE.md section 11)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "Eres un analista cuantitativo senior especializado en renta variable americana.\n"
    "Recibes datos técnicos y fundamentales de una señal de mercado y debes proporcionar\n"
    "un análisis conciso, factual y accionable. Basa tu respuesta en datos verificables.\n"
    "Evita especulación sin respaldo. Usa terminología financiera precisa."
)

_NO_GROUNDING_NOTE = "\n\nNota: sin acceso a búsqueda web en tiempo real."


def _build_user_prompt(senal: dict) -> str:
    """Build the dynamic user prompt from the signal dict.

    All numeric fields are cast defensively to float/int so the function
    never crashes on string-typed values coming from enricher.py.

    Args:
        senal: Enriched signal dict.

    Returns:
        Formatted prompt string ready to send to Gemini.
    """
    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(senal.get(key) or default)
        except (ValueError, TypeError):
            return default

    ticker     = senal.get("ticker", "N/A")
    sector     = senal.get("sector", "N/A") or "N/A"
    tipo       = (senal.get("tipo_evento") or "").replace("_", " ").upper()
    fecha      = senal.get("fecha_evento", "N/A")
    pe_ratio   = senal.get("pe_ratio", "N/A") or "N/A"
    eps        = senal.get("eps", "N/A") or "N/A"
    market_cap = senal.get("market_cap", "N/A") or "N/A"
    earnings   = senal.get("earnings_date", "N/A") or "N/A"
    sentiment  = senal.get("news_sentiment", "N/A") or "N/A"

    return (
        f"SEÑAL DETECTADA:\n"
        f"- Ticker: {ticker} ({sector})\n"
        f"- Tipo: {tipo}\n"
        f"- Fecha: {fecha}\n"
        f"- Precio cierre: ${_f('precio_cierre'):.2f}\n"
        f"\n"
        f"DATOS TÉCNICOS:\n"
        f"- SMA 50: ${_f('sma_50'):.2f} | SMA 200: ${_f('sma_200'):.2f}\n"
        f"- Distancia entre medias: {_f('dist_sma_pct'):+.2f}%\n"
        f"- RSI-14: {_f('rsi_14'):.1f} | ADX-14: {_f('adx_14'):.1f} (fuerza tendencia)\n"
        f"- Volumen relativo: {_f('volumen_relativo'):.1f}x sobre media 20 días\n"
        f"- Scoring señal: {int(senal.get('scoring', 1))}/3\n"
        f"\n"
        f"DATOS FUNDAMENTALES:\n"
        f"- P/E Ratio: {pe_ratio} | EPS: {eps}\n"
        f"- Cap. bursátil: {market_cap}\n"
        f"- Próximo earnings: {earnings}\n"
        f"\n"
        f"SENTIMIENTO NOTICIAS (últimos 7 días):\n"
        f"- Sentimiento: {sentiment} (score: {_f('news_score'):.3f})\n"
        f"\n"
        f"CONTEXTO MACROECONÓMICO:\n"
        f"- Fed Funds Rate: {_f('fed_funds_rate'):.2f}%\n"
        f"- CPI Inflación: {_f('cpi_inflacion'):.2f}%\n"
        f"\n"
        f"Busca noticias recientes sobre {ticker} y analiza en exactamente 3 puntos\n"
        f"(máximo 30 palabras cada uno):\n"
        f"1. TÉCNICO: ¿el contexto de precio justifica o contradice el cruce?\n"
        f"2. FUNDAMENTAL: ¿hay catalizador real o es movimiento de mercado general?\n"
        f"3. RIESGO: principal amenaza a vigilar en las próximas 4 semanas."
    )


# ---------------------------------------------------------------------------
# Internal generation helpers
# ---------------------------------------------------------------------------

def _generate(prompt: str, use_grounding: bool) -> str:
    """Execute one Gemini generation call.

    Args:
        prompt:        Full user prompt text.
        use_grounding: If True, attaches the google_search grounding tool.

    Returns:
        Response text from Gemini.

    Raises:
        Any exception from the Gemini SDK (rate limit, invalid arg, etc.).
    """
    config_kwargs: dict = {"system_instruction": _SYSTEM_PROMPT}

    if use_grounding:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    response = _client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    text = response.text
    if not text:
        return "[Análisis no disponible: respuesta vacía de Gemini]"
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analizar_senal(senal: dict) -> str:
    """Generate a structured 3-point analysis of a detected trading signal.

    Execution flow:
      1. Try with Google Search grounding (real-time news access).
      2. If grounding is unavailable (region/plan), retry without grounding
         and append a note to the prompt.
      3. On ResourceExhausted (429) / rate-limit: exponential retry up to
         _MAX_RETRIES attempts (waits 60 s, 120 s, 240 s).

    Args:
        senal: Enriched signal dict with technical, fundamental and macro data.

    Returns:
        Analysis string with 3 labelled points (TÉCNICO / FUNDAMENTAL / RIESGO),
        or a descriptive error message on failure.
    """
    ticker  = senal.get("ticker", "?")
    prompt  = _build_user_prompt(senal)
    grounding_available = True

    for attempt in range(1, _MAX_RETRIES + 1):
        current_prompt = prompt if grounding_available else prompt + _NO_GROUNDING_NOTE

        try:
            text = _generate(current_prompt, use_grounding=grounding_available)
            mode = "grounding" if grounding_available else "sin grounding"
            logger.info("%s: Gemini analysis OK (%s, %d chars)", ticker, mode, len(text))
            return text

        except Exception as exc:
            exc_str = str(exc).lower()
            exc_type = type(exc).__name__

            # --- Grounding not supported → switch off and retry immediately ---
            if (grounding_available and (
                "google_search_retrieval is not supported" in exc_str
                or "google_search" in exc_str
                or "tool" in exc_str
                or "invalid" in exc_str
            )):
                logger.warning(
                    "%s: Google Search grounding unavailable (%s), switching to non-grounded",
                    ticker, exc_type,
                )
                grounding_available = False
                continue   # retry right away without grounding

            # --- Rate limit / quota → exponential backoff ---
            if "429" in exc_str or "quota" in exc_str or "resource_exhausted" in exc_str or "rate" in exc_str:
                wait = _RETRY_WAIT * (2 ** (attempt - 1))
                logger.warning(
                    "%s: Gemini rate limit (attempt %d/%d) — waiting %d s",
                    ticker, attempt, _MAX_RETRIES, wait,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)
                    continue
                return f"[Error IA: límite de cuota de Gemini alcanzado para {ticker}]"

            # --- Any other error ---
            logger.error(
                "%s: Gemini unexpected error (attempt %d/%d) %s: %s",
                ticker, attempt, _MAX_RETRIES, exc_type, str(exc)[:120],
            )
            if attempt < _MAX_RETRIES:
                time.sleep(30)
                continue

    return f"[Error IA: máximo de reintentos alcanzado para {ticker}]"
