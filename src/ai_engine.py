"""AI engine: generates signal analysis using Gemini 2.5 Flash with Google Search grounding.

Uses the new google-genai SDK (google.generativeai is deprecated as of 2025).
Model: gemini-2.5-flash (replaces spec's gemini-1.5-pro, which is no longer available).
Grounding via google_search tool; falls back gracefully to non-grounded generation.
Returns a dict with analysis texts + trading levels (or mathematical fallbacks).
"""

import json
import logging
import re
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
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "Eres un analista cuantitativo senior especializado en renta variable americana.\n"
    "Recibes datos técnicos y fundamentales de una señal de mercado y debes proporcionar\n"
    "un análisis conciso, factual y accionable. Basa tu respuesta en datos verificables.\n"
    "Evita especulación sin respaldo. Usa terminología financiera precisa.\n"
    "IMPORTANTE: responde SIEMPRE con un objeto JSON válido sin texto adicional ni bloques markdown."
)

_NO_GROUNDING_NOTE = "\n\nNota: sin acceso a búsqueda web en tiempo real."


def _build_user_prompt(senal: dict) -> str:
    """Build the dynamic user prompt requesting JSON output with analysis + trading levels."""
    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(senal.get(key) or default)
        except (ValueError, TypeError):
            return default

    ticker      = senal.get("ticker", "N/A")
    sector      = senal.get("sector", "N/A") or "N/A"
    tipo        = (senal.get("tipo_evento") or "").replace("_", " ").upper()
    fecha       = senal.get("fecha_evento", "N/A")
    pe_ratio    = senal.get("pe_ratio", "N/A") or "N/A"
    eps         = senal.get("eps", "N/A") or "N/A"
    market_cap  = senal.get("market_cap", "N/A") or "N/A"
    earnings    = senal.get("earnings_date", "N/A") or "N/A"
    sentiment   = senal.get("news_sentiment", "N/A") or "N/A"
    tipo_evento = senal.get("tipo_evento", "golden_cross")

    precio_cierre = _f("precio_cierre")
    sma_200       = _f("sma_200")

    # Pre-calculate stop reference so Gemini can anchor on the correct value
    stop_ref = round(sma_200 * 0.99, 2) if tipo_evento == "golden_cross" else round(sma_200 * 1.01, 2)

    return (
        f"SEÑAL DETECTADA:\n"
        f"- Ticker: {ticker} ({sector})\n"
        f"- Tipo: {tipo}\n"
        f"- Fecha: {fecha}\n"
        f"- Precio cierre: ${precio_cierre:.2f}\n"
        f"\n"
        f"DATOS TÉCNICOS:\n"
        f"- SMA 50: ${_f('sma_50'):.2f} | SMA 200: ${sma_200:.2f}\n"
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
        f"Busca noticias recientes sobre {ticker} y responde ÚNICAMENTE con el siguiente "
        f"JSON (sin texto adicional, sin bloques ```json, sin comentarios):\n"
        f'{{\n'
        f'  "analisis": {{\n'
        f'    "tecnico": "<máx 30 palabras: justifica o contradice el cruce técnicamente>",\n'
        f'    "fundamental": "<máx 30 palabras: catalizador real o movimiento general>",\n'
        f'    "riesgo": "<máx 30 palabras: principal amenaza próximas 4 semanas>"\n'
        f'  }},\n'
        f'  "niveles": {{\n'
        f'    "entrada": {precio_cierre:.2f},\n'
        f'    "objetivo": <resistencia más cercana o extensión 10-15% Fibonacci sobre {precio_cierre:.2f}>,\n'
        f'    "stop_loss": {stop_ref:.2f},\n'
        f'    "ratio_rr": <(objetivo-entrada)/(entrada-stop_loss) redondeado 2 decimales>\n'
        f'  }}\n'
        f'}}\n'
        f'Si ratio_rr < 1.5 añade "descartar": true al nivel raíz del JSON.'
    )


# ---------------------------------------------------------------------------
# Fallback mathematical level calculator
# ---------------------------------------------------------------------------

def _fallback_levels(senal: dict) -> dict:
    """Calculate trading levels mathematically when Gemini JSON is invalid."""
    def _f(key, default=0.0):
        try:
            return float(senal.get(key) or default)
        except (ValueError, TypeError):
            return default

    precio    = _f("precio_cierre")
    sma_200   = _f("sma_200")
    tipo      = senal.get("tipo_evento", "golden_cross")

    entrada  = precio
    stop     = round(sma_200 * 0.99, 2) if tipo == "golden_cross" else round(sma_200 * 1.01, 2)
    objetivo = round(entrada * 1.10, 2)

    denom    = max(abs(entrada - stop), 0.01)
    ratio_rr = round(abs(objetivo - entrada) / denom, 2)
    descartar = ratio_rr < 1.5

    return {
        "entrada":  entrada,
        "objetivo": objetivo,
        "stop":     stop,
        "ratio_rr": ratio_rr,
        "descartar": descartar,
    }


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

def _parse_response(text: str, senal: dict) -> dict:
    """Extract analysis and trading levels from Gemini's response text.

    Tries to parse JSON (with or without markdown fences).
    Falls back to mathematical levels if JSON is absent or malformed.

    Returns:
        Dict ready to be merged into the signal with keys:
        analisis_gemini, analisis_tecnico, analisis_fundamental, analisis_riesgo,
        precio_entrada, precio_objetivo, precio_stop, ratio_rr, descartar.
    """
    # Strip markdown fences if present
    fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    raw_match   = re.search(r'(\{.*\})', text, re.DOTALL)

    parsed: dict | None = None
    for candidate in [fence_match, raw_match]:
        if candidate:
            try:
                parsed = json.loads(candidate.group(1))
                break
            except (json.JSONDecodeError, AttributeError):
                continue

    if parsed:
        logger.debug("Gemini returned valid JSON (%d chars)", len(text))
    else:
        logger.warning("Gemini response is not valid JSON — using fallback levels")

    # --- Analysis texts ---
    analisis_block = (parsed or {}).get("analisis", {})
    tecnico     = str(analisis_block.get("tecnico",     "")).strip()
    fundamental = str(analisis_block.get("fundamental", "")).strip()
    riesgo      = str(analisis_block.get("riesgo",      "")).strip()

    if not any([tecnico, fundamental, riesgo]):
        # Gemini returned no structured analysis — store raw text
        tecnico     = text[:300].strip()
        fundamental = ""
        riesgo      = ""

    analisis_gemini = "\n".join(filter(None, [
        f"TÉCNICO: {tecnico}"       if tecnico     else None,
        f"FUNDAMENTAL: {fundamental}" if fundamental else None,
        f"RIESGO: {riesgo}"         if riesgo      else None,
    ])) or text[:500]

    # --- Trading levels ---
    fb = _fallback_levels(senal)

    if parsed and "niveles" in parsed:
        try:
            niv      = parsed["niveles"]
            entrada  = float(niv.get("entrada",   fb["entrada"]))
            objetivo = float(niv.get("objetivo",  fb["objetivo"]))
            stop     = float(niv.get("stop_loss", fb["stop"]))
            ratio_rr = float(niv.get("ratio_rr",  fb["ratio_rr"]))
        except (ValueError, TypeError):
            entrada, objetivo, stop, ratio_rr = fb["entrada"], fb["objetivo"], fb["stop"], fb["ratio_rr"]
    else:
        entrada, objetivo, stop, ratio_rr = fb["entrada"], fb["objetivo"], fb["stop"], fb["ratio_rr"]

    # Gemini-declared descartar OR computed from ratio
    descartar = bool((parsed or {}).get("descartar", False)) or ratio_rr < 1.5

    result = {
        "analisis_gemini":      analisis_gemini,
        "analisis_tecnico":     tecnico,
        "analisis_fundamental": fundamental,
        "analisis_riesgo":      riesgo,
        "precio_entrada":       round(entrada,  2),
        "precio_objetivo":      round(objetivo, 2),
        "precio_stop":          round(stop,     2),
        "ratio_rr":             round(ratio_rr, 2),
        "descartar":            descartar,
    }
    logger.info(
        "AI levels — entrada=%.2f objetivo=%.2f stop=%.2f rr=%.2f descartar=%s",
        entrada, objetivo, stop, ratio_rr, descartar,
    )
    return result


# ---------------------------------------------------------------------------
# Internal generation helper
# ---------------------------------------------------------------------------

def _generate(prompt: str, use_grounding: bool) -> str:
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
        return ""
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analizar_senal(senal: dict) -> dict:
    """Generate a structured 3-point analysis and trading levels for a signal.

    Returns a dict with keys ready to be merged into the signal:
      analisis_gemini (str), analisis_tecnico (str), analisis_fundamental (str),
      analisis_riesgo (str), precio_entrada (float), precio_objetivo (float),
      precio_stop (float), ratio_rr (float), descartar (bool).

    On total failure returns mathematical fallback levels with error text in analisis_gemini.
    """
    ticker  = senal.get("ticker", "?")
    prompt  = _build_user_prompt(senal)
    grounding_available = True

    for attempt in range(1, _MAX_RETRIES + 1):
        current_prompt = prompt if grounding_available else prompt + _NO_GROUNDING_NOTE

        try:
            text = _generate(current_prompt, use_grounding=grounding_available)
            mode = "grounding" if grounding_available else "sin grounding"
            logger.info("%s: Gemini response OK (%s, %d chars)", ticker, mode, len(text))
            return _parse_response(text, senal)

        except Exception as exc:
            exc_str  = str(exc).lower()
            exc_type = type(exc).__name__

            # --- Grounding not supported → switch off and retry immediately ---
            if grounding_available and any(k in exc_str for k in (
                "google_search_retrieval is not supported", "google_search", "tool", "invalid"
            )):
                logger.warning(
                    "%s: Google Search grounding unavailable (%s), switching to non-grounded",
                    ticker, exc_type,
                )
                grounding_available = False
                continue

            # --- Rate limit / quota → exponential backoff ---
            if any(k in exc_str for k in ("429", "quota", "resource_exhausted", "rate")):
                wait = _RETRY_WAIT * (2 ** (attempt - 1))
                logger.warning(
                    "%s: Gemini rate limit (attempt %d/%d) — waiting %d s",
                    ticker, attempt, _MAX_RETRIES, wait,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)
                    continue
                break

            # --- Any other error ---
            logger.error(
                "%s: Gemini unexpected error (attempt %d/%d) %s: %s",
                ticker, attempt, _MAX_RETRIES, exc_type, str(exc)[:120],
            )
            if attempt < _MAX_RETRIES:
                time.sleep(30)
                continue

    # All retries exhausted — return mathematical fallback with error text
    fb = _fallback_levels(senal)
    error_msg = f"[Error IA: máximo de reintentos alcanzado para {ticker}]"
    return {
        "analisis_gemini":      error_msg,
        "analisis_tecnico":     error_msg,
        "analisis_fundamental": "",
        "analisis_riesgo":      "",
        "precio_entrada":       fb["entrada"],
        "precio_objetivo":      fb["objetivo"],
        "precio_stop":          fb["stop"],
        "ratio_rr":             fb["ratio_rr"],
        "descartar":            fb["descartar"],
    }
