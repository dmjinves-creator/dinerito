"""Enricher: fetches Alpha Vantage data for detected signals only.

Called exclusively when signals exist — maximum 3 API requests per signal.
Respects the free-tier rate limit of 5 requests/minute via a 12-second delay
after every HTTP call.
"""

import logging
import time

import requests

from src.config import ALPHA_VANTAGE_KEY

logger = logging.getLogger(__name__)

_AV_BASE = "https://www.alphavantage.co/query"
_CALL_DELAY = 12  # seconds between Alpha Vantage calls (free tier: 5 req/min)

# Module-level cache — populated on first call, reused for the entire pipeline run
_macro_cache: dict | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(params: dict) -> dict:
    """Execute one GET request to Alpha Vantage and return the parsed JSON.

    Sleeps _CALL_DELAY seconds after the request to honour the rate limit.
    Returns an empty dict on any error or on API-level warning messages.

    Args:
        params: Query parameters dict, must include 'function' and 'apikey'.

    Returns:
        Parsed JSON as a dict, or {} on failure.
    """
    func = params.get("function", "UNKNOWN")
    try:
        resp = requests.get(_AV_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data: dict = resp.json()
    except Exception as exc:
        logger.error("Alpha Vantage request failed (function=%s): %s", func, exc)
        time.sleep(_CALL_DELAY)
        return {}

    # API returns 200 with an error/rate-limit message inside the body
    if "Information" in data or "Note" in data:
        msg = data.get("Information") or data.get("Note", "")
        logger.warning("Alpha Vantage rate-limit or info (function=%s): %.120s", func, msg)
        time.sleep(_CALL_DELAY)
        return {}

    time.sleep(_CALL_DELAY)
    return data


def _normalizar_label(label: str) -> str:
    """Collapse Alpha Vantage's five sentiment labels into three.

    Bullish / Somewhat-Bullish  → Bullish
    Somewhat-Bearish / Bearish  → Bearish
    Neutral                     → Neutral

    Args:
        label: Raw label string from the API.

    Returns:
        One of 'Bullish', 'Bearish', or 'Neutral'.
    """
    lower = label.lower()
    if "bullish" in lower:
        return "Bullish"
    if "bearish" in lower:
        return "Bearish"
    return "Neutral"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def obtener_noticias_sentimiento(ticker: str) -> dict:
    """Fetch the 5 latest news articles and compute ticker-specific sentiment.

    Uses the ticker_sentiment sub-array of each article when available;
    falls back to the article's overall_sentiment_score otherwise.
    The returned score is the arithmetic mean across all matched articles.

    Args:
        ticker: Ticker symbol (e.g. 'AAPL').

    Returns:
        Dict with keys:
          - 'news_sentiment' (str): 'Bullish', 'Bearish', or 'Neutral'.
          - 'news_score' (float): mean sentiment score in [-1, 1].
        Returns safe defaults {'news_sentiment': 'N/A', 'news_score': 0.0} on failure.
    """
    default = {"news_sentiment": "N/A", "news_score": 0.0}

    data = _get({
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": "5",
        "sort": "LATEST",
        "apikey": ALPHA_VANTAGE_KEY,
    })

    feed: list = data.get("feed", [])
    if not feed:
        logger.warning("%s: empty news feed from Alpha Vantage", ticker)
        return default

    scores: list[float] = []
    labels: list[str] = []

    for article in feed:
        # Prefer the ticker-specific sentiment score over the article's overall score
        ts_matched = [
            ts for ts in article.get("ticker_sentiment", [])
            if ts.get("ticker", "").upper() == ticker.upper()
        ]
        if ts_matched:
            entry = ts_matched[0]
            try:
                scores.append(float(entry["ticker_sentiment_score"]))
                labels.append(entry.get("ticker_sentiment_label", "Neutral"))
            except (ValueError, KeyError):
                pass
        else:
            # Fallback: article-level overall sentiment
            try:
                scores.append(float(article.get("overall_sentiment_score", 0)))
                labels.append(article.get("overall_sentiment_label", "Neutral"))
            except (ValueError, TypeError):
                pass

    if not scores:
        return default

    avg_score = round(sum(scores) / len(scores), 4)
    dominant_label = max(set(labels), key=labels.count)

    logger.info(
        "%s: news sentiment=%s score=%.4f (%d articles)",
        ticker, _normalizar_label(dominant_label), avg_score, len(scores),
    )
    return {
        "news_sentiment": _normalizar_label(dominant_label),
        "news_score": avg_score,
    }


def obtener_overview_empresa(ticker: str) -> dict:
    """Fetch company fundamentals from Alpha Vantage COMPANY_OVERVIEW endpoint.

    Note: Alpha Vantage OVERVIEW does not include a NextEarningsDate field.
    The 'earnings_date' key is populated with LatestQuarter as the closest proxy.

    Args:
        ticker: Ticker symbol (e.g. 'AAPL').

    Returns:
        Dict with keys: sector, pe_ratio, eps, market_cap, earnings_date.
        All values are strings; returns 'N/A' for any missing field.
    """
    default = {
        "sector":        "N/A",
        "pe_ratio":      "N/A",
        "eps":           "N/A",
        "market_cap":    "N/A",
        "earnings_date": "N/A",
    }

    data = _get({
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_KEY,
    })

    if not data or "Symbol" not in data:
        logger.warning("%s: no overview data returned from Alpha Vantage", ticker)
        return default

    result = {
        "sector":        data.get("Sector")               or "N/A",
        "pe_ratio":      data.get("PERatio")              or "N/A",
        "eps":           data.get("EPS")                  or "N/A",
        "market_cap":    data.get("MarketCapitalization") or "N/A",
        # NextEarningsDate absent from Alpha Vantage OVERVIEW — use LatestQuarter
        "earnings_date": data.get("LatestQuarter")        or "N/A",
    }

    logger.info(
        "%s: overview fetched — sector=%s PE=%s EPS=%s mcap=%s",
        ticker, result["sector"], result["pe_ratio"],
        result["eps"], result["market_cap"],
    )
    return result


def obtener_contexto_macro() -> dict:
    """Fetch the latest Federal Funds Rate and CPI from Alpha Vantage.

    Results are cached in a module-level variable so the pipeline only pays
    for two API calls no matter how many signals are enriched.

    Returns:
        Dict with keys:
          - 'fed_rate' (float): most recent monthly Federal Funds Rate.
          - 'cpi' (float): most recent monthly CPI value.
        Returns 0.0 for any value that cannot be parsed.
    """
    global _macro_cache

    if _macro_cache is not None:
        logger.debug(
            "Macro context: cache hit (fed_rate=%.2f, cpi=%.2f)",
            _macro_cache["fed_rate"], _macro_cache["cpi"],
        )
        return _macro_cache

    fed_rate = 0.0
    cpi = 0.0

    # --- Federal Funds Rate ---
    fed_data = _get({
        "function": "FEDERAL_FUNDS_RATE",
        "interval": "monthly",
        "apikey": ALPHA_VANTAGE_KEY,
    })
    if fed_data.get("data"):
        try:
            fed_rate = float(fed_data["data"][0]["value"])
        except (ValueError, KeyError, IndexError) as exc:
            logger.warning("Could not parse Federal Funds Rate: %s", exc)

    # --- CPI ---
    cpi_data = _get({
        "function": "CPI",
        "interval": "monthly",
        "apikey": ALPHA_VANTAGE_KEY,
    })
    if cpi_data.get("data"):
        try:
            cpi = float(cpi_data["data"][0]["value"])
        except (ValueError, KeyError, IndexError) as exc:
            logger.warning("Could not parse CPI: %s", exc)

    _macro_cache = {"fed_rate": round(fed_rate, 2), "cpi": round(cpi, 2)}
    logger.info(
        "Macro context cached — fed_rate=%.2f%% | cpi=%.2f",
        _macro_cache["fed_rate"], _macro_cache["cpi"],
    )
    return _macro_cache


def enriquecer_senal(senal: dict) -> dict:
    """Enrich a detected signal with news sentiment, fundamentals, and macro data.

    Makes at most 3 Alpha Vantage calls (news + overview + macro).
    The macro call is skipped after the first signal (cached).

    Args:
        senal: Signal dict as returned by signal_detector.detectar_cruce.
              Must contain at least 'ticker'.

    Returns:
        New dict merging the original signal with enrichment fields:
        news_sentiment, news_score, sector, pe_ratio, eps,
        market_cap, earnings_date, fed_funds_rate, cpi_inflacion.
    """
    ticker: str = senal["ticker"]
    logger.info("%s: starting enrichment", ticker)

    noticias = obtener_noticias_sentimiento(ticker)
    overview = obtener_overview_empresa(ticker)
    macro    = obtener_contexto_macro()

    enriched = {
        **senal,
        **noticias,
        **overview,
        "fed_funds_rate": macro["fed_rate"],
        "cpi_inflacion":  macro["cpi"],
    }

    logger.info("%s: enrichment complete", ticker)
    return enriched
