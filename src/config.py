"""Configuration module: loads env vars, defines constants and ticker list."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

logger = logging.getLogger(__name__)


def _req(key: str) -> str:
    """Return the value of a required environment variable.

    Args:
        key: Name of the environment variable.

    Returns:
        The string value of the variable.

    Raises:
        ValueError: If the variable is not set or is empty.
    """
    value = os.getenv(key)
    if not value:
        raise ValueError(
            f"Required environment variable '{key}' is not set. "
            f"Check your .env file (see .env.example for reference)."
        )
    return value


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
SUPABASE_URL: str = _req("SUPABASE_URL")
SUPABASE_KEY: str = _req("SUPABASE_KEY")
SUPABASE_DB_PASSWORD: str = _req("SUPABASE_DB_PASSWORD")
GEMINI_API_KEY: str = _req("GEMINI_API_KEY")
ALPHA_VANTAGE_KEY: str = _req("ALPHA_VANTAGE_KEY")
TELEGRAM_BOT_TOKEN: str = _req("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: str = _req("TELEGRAM_CHAT_ID")

# ---------------------------------------------------------------------------
# System constants
# ---------------------------------------------------------------------------
ADX_MIN: int = 25           # Minimum ADX to confirm a real trend (anti-whipsaw)
RSI_MAX_GOLDEN: int = 75    # RSI ceiling for Golden Cross (avoid overbought traps)
RSI_MIN_DEATH: int = 25     # RSI floor for Death Cross (avoid oversold traps)
VOL_MIN_RATIO: float = 1.2  # Minimum volume vs 20-day average to confirm conviction

# ---------------------------------------------------------------------------
# Ticker list — 75 assets
# ---------------------------------------------------------------------------
TICKERS: list[str] = [
    # Mega-cap (20)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "JPM", "V", "MA",
    "UNH", "JNJ", "PG", "HD", "BAC", "WMT", "XOM", "CVX", "LLY", "AVGO",
    # Mid-cap growth (40)
    "COST", "MRK", "ABBV", "CRM", "ACN", "AMD", "NFLX", "TMO", "PEP", "KO",
    "ADBE", "CSCO", "MCD", "ABT", "WFC", "TXN", "NEE", "LIN", "PM", "DHR",
    "INTC", "RTX", "HON", "UPS", "IBM", "CAT", "SBUX", "GS", "BKNG", "SPGI",
    "AMGN", "MDT", "ISRG", "NOW", "PANW", "UBER", "SHOP", "SQ", "SNOW", "ARM",
    # Sector ETFs (10)
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "XLRE",
    # Broad-market indices (5)
    "SPY", "QQQ", "DIA", "IWM", "VTI",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def calcular_scoring(dist_pct: float) -> int:
    """Calculate signal strength score from the SMA distance percentage.

    Formula: dist_pct = ((SMA_50 - SMA_200) / SMA_200) * 100

    Args:
        dist_pct: Signed percentage distance between SMA_50 and SMA_200.

    Returns:
        1 — weak  (|dist_pct| < 0.5%)
        2 — medium (0.5% <= |dist_pct| < 1.5%)
        3 — strong (|dist_pct| >= 1.5%)
    """
    abs_dist = abs(dist_pct)
    if abs_dist < 0.5:
        return 1
    if abs_dist < 1.5:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Tickers cargados: {len(TICKERS)}")
    print(f"ADX_MIN={ADX_MIN} | RSI_MAX_GOLDEN={RSI_MAX_GOLDEN} | "
          f"RSI_MIN_DEATH={RSI_MIN_DEATH} | VOL_MIN_RATIO={VOL_MIN_RATIO}")
    print(f"Scoring test — 0.3%: {calcular_scoring(0.3)} | "
          f"1.0%: {calcular_scoring(1.0)} | 2.5%: {calcular_scoring(2.5)}")
