"""Historical backtesting module: replays signal detection across full price history."""

import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import timedelta

import pandas as pd
import pandas_ta as ta
import psycopg2
import psycopg2.extras
import yfinance as yf

from src.config import (
    ADX_MIN,
    RSI_MAX_GOLDEN,
    RSI_MIN_DEATH,
    SUPABASE_DB_PASSWORD,
    SUPABASE_URL,
    TICKERS,
    VOL_MIN_RATIO,
    calcular_scoring,
)

logger = logging.getLogger(__name__)

BACKTEST_START = "2015-01-01"
_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# DB helpers (self-contained — no dep on database.py to avoid circular import)
# ---------------------------------------------------------------------------

def _build_conn_str() -> str:
    match = re.search(r"https://(.+)\.supabase\.co", SUPABASE_URL)
    if not match:
        raise ValueError(f"Cannot extract project ref from SUPABASE_URL: {SUPABASE_URL}")
    ref = match.group(1)
    region = os.getenv("SUPABASE_POOLER_REGION", "eu-west-1")
    return (
        f"postgresql://postgres.{ref}:{SUPABASE_DB_PASSWORD}"
        f"@aws-0-{region}.pooler.supabase.com:5432/postgres?sslmode=require"
    )


@contextmanager
def _conn():
    conn = psycopg2.connect(_build_conn_str(), connect_timeout=15)
    try:
        yield conn
    finally:
        conn.close()


def create_backtest_table() -> None:
    """Create backtest_resultados table with UNIQUE constraint for idempotent runs."""
    sql = """
        CREATE TABLE IF NOT EXISTS backtest_resultados (
            id              SERIAL PRIMARY KEY,
            ticker          VARCHAR(10),
            sector          VARCHAR(100),
            tipo_evento     VARCHAR(20),
            fecha_senal     DATE,
            precio_entrada  DECIMAL(10,2),
            scoring         INTEGER,
            adx_valor       DECIMAL(5,2),
            rsi_valor       DECIMAL(5,2),
            vol_ratio       DECIMAL(5,2),
            dist_sma_pct    DECIMAL(5,2),
            precio_7d       DECIMAL(10,2),
            precio_15d      DECIMAL(10,2),
            precio_30d      DECIMAL(10,2),
            retorno_7d      DECIMAL(6,2),
            retorno_15d     DECIMAL(6,2),
            retorno_30d     DECIMAL(6,2),
            exito_30d       BOOLEAN,
            año             INTEGER,
            UNIQUE(ticker, tipo_evento, fecha_senal)
        )
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print("  Table backtest_resultados: OK")


def _bulk_insert(rows: list[dict]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO backtest_resultados (
            ticker, sector, tipo_evento, fecha_senal, precio_entrada,
            scoring, adx_valor, rsi_valor, vol_ratio, dist_sma_pct,
            precio_7d, precio_15d, precio_30d,
            retorno_7d, retorno_15d, retorno_30d, exito_30d, año
        ) VALUES %s
        ON CONFLICT (ticker, tipo_evento, fecha_senal) DO NOTHING
    """
    values = [
        (
            r["ticker"], r["sector"], r["tipo_evento"], r["fecha_senal"],
            r["precio_entrada"], r["scoring"], r["adx_valor"], r["rsi_valor"],
            r["vol_ratio"], r["dist_sma_pct"],
            r["precio_7d"], r["precio_15d"], r["precio_30d"],
            r["retorno_7d"], r["retorno_15d"], r["retorno_30d"],
            r["exito_30d"], r["año"],
        )
        for r in rows
    ]
    with _conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, values)
        conn.commit()
    return len(values)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _download_full(ticker: str) -> pd.DataFrame | None:
    """Download OHLCV from BACKTEST_START to today."""
    try:
        raw = yf.download(ticker, start=BACKTEST_START, progress=False, auto_adjust=True)
    except Exception as exc:
        logger.warning("%s: download error: %s", ticker, exc)
        return None

    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.copy()
    crit = ["Close", "High", "Low", "Volume"]
    df[crit] = df[crit].ffill().bfill()

    if len(df) < 220 or df[crit].isnull().any().any():
        return None

    return df


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append SMA50/200, RSI14, ADX14, VOL_MA20 and drop NaN warm-up rows."""
    df = df.copy()
    df["SMA_50"]  = ta.sma(df["Close"], length=50)
    df["SMA_200"] = ta.sma(df["Close"], length=200)
    df["RSI_14"]  = ta.rsi(df["Close"], length=14)

    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx_df is not None:
        adx_col = "ADX_14" if "ADX_14" in adx_df.columns else next(
            (c for c in adx_df.columns if c.startswith("ADX")), None
        )
        df["ADX_14"] = adx_df[adx_col] if adx_col else float("nan")
    else:
        df["ADX_14"] = float("nan")

    df["VOL_MA20"] = ta.sma(df["Volume"], length=20)
    df.dropna(subset=["SMA_50", "SMA_200", "RSI_14", "ADX_14", "VOL_MA20"], inplace=True)
    return df


def _get_sector(ticker: str) -> str:
    """Fetch sector tag from yfinance .info; returns 'ETF' or 'Unknown' as fallback."""
    try:
        info = yf.Ticker(ticker).fast_info
        # fast_info doesn't have sector; fall back to full info for non-ETFs
        quote_type = getattr(info, "quote_type", None) or ""
        if quote_type.upper() in ("ETF", "INDEX"):
            return quote_type.capitalize()
    except Exception:
        pass
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector")
        if sector:
            return sector
        qt = info.get("quoteType", "Unknown")
        return qt.capitalize()
    except Exception:
        return "Unknown"


# ---------------------------------------------------------------------------
# Core per-ticker backtest
# ---------------------------------------------------------------------------

def _backtest_ticker(ticker: str) -> list[dict]:
    """Download full history for one ticker and detect all historical signals."""
    sector = _get_sector(ticker)

    df_raw = _download_full(ticker)
    if df_raw is None:
        return []

    df = _compute_indicators(df_raw)
    if len(df) < 2:
        return []

    # Pre-extract numpy arrays for performance
    dates   = df.index
    closes  = df["Close"].to_numpy(dtype=float)
    sma50   = df["SMA_50"].to_numpy(dtype=float)
    sma200  = df["SMA_200"].to_numpy(dtype=float)
    adx_arr = df["ADX_14"].to_numpy(dtype=float)
    rsi_arr = df["RSI_14"].to_numpy(dtype=float)
    vol_arr = df["Volume"].to_numpy(dtype=float)
    volma   = df["VOL_MA20"].to_numpy(dtype=float)

    resultados: list[dict] = []

    for i in range(1, len(df)):
        golden = sma50[i - 1] <= sma200[i - 1] and sma50[i] > sma200[i]
        death  = sma50[i - 1] >= sma200[i - 1] and sma50[i] < sma200[i]

        if not golden and not death:
            continue

        tipo = "golden_cross" if golden else "death_cross"

        if adx_arr[i] < ADX_MIN:
            continue
        vol_ratio = vol_arr[i] / volma[i] if volma[i] > 0 else 0.0
        if vol_ratio < VOL_MIN_RATIO:
            continue
        if tipo == "golden_cross" and rsi_arr[i] > RSI_MAX_GOLDEN:
            continue
        if tipo == "death_cross" and rsi_arr[i] < RSI_MIN_DEATH:
            continue

        entry_price = float(closes[i])
        entry_date  = dates[i]
        dist_sma    = float(((sma50[i] - sma200[i]) / sma200[i]) * 100)
        scoring     = calcular_scoring(dist_sma)

        def _get_price(days: int) -> float | None:
            target = entry_date + timedelta(days=days)
            pos = dates.searchsorted(target, side="left")
            if pos < len(dates) and pos > i:
                return float(closes[pos])
            return None

        p7  = _get_price(7)
        p15 = _get_price(15)
        p30 = _get_price(30)

        r7  = round((p7  - entry_price) / entry_price * 100, 2) if p7  is not None else None
        r15 = round((p15 - entry_price) / entry_price * 100, 2) if p15 is not None else None
        r30 = round((p30 - entry_price) / entry_price * 100, 2) if p30 is not None else None

        if r30 is not None:
            exito = bool(r30 > 0) if tipo == "golden_cross" else bool(r30 < 0)
        else:
            exito = None

        def _f(v) -> float | None:
            return round(float(v), 2) if v is not None else None

        resultados.append({
            "ticker":         ticker,
            "sector":         sector,
            "tipo_evento":    tipo,
            "fecha_senal":    entry_date.date(),
            "precio_entrada": _f(entry_price),
            "scoring":        int(scoring),
            "adx_valor":      _f(adx_arr[i]),
            "rsi_valor":      _f(rsi_arr[i]),
            "vol_ratio":      _f(vol_ratio),
            "dist_sma_pct":   round(float(dist_sma), 4),
            "precio_7d":      _f(p7),
            "precio_15d":     _f(p15),
            "precio_30d":     _f(p30),
            "retorno_7d":     r7,
            "retorno_15d":    r15,
            "retorno_30d":    r30,
            "exito_30d":      exito,
            "año":            int(entry_date.year),
        })

    return resultados


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(years: int = 10, tickers: list[str] | None = None) -> list[dict]:
    """Run historical backtest for all tickers and persist results to Supabase.

    Args:
        years:   Unused (start date fixed at BACKTEST_START = 2015-01-01).
        tickers: Symbols to process; defaults to full TICKERS list from config.

    Returns:
        All collected result dicts (including previously existing signals).
    """
    if tickers is None:
        tickers = TICKERS

    print(f"\n{'='*55}")
    print(f"DINERITO BACKTESTER — {len(tickers)} tickers from {BACKTEST_START}")
    print(f"{'='*55}")

    create_backtest_table()

    all_results: list[dict] = []
    done = 0
    total = len(tickers)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_backtest_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results = future.result()
            except Exception as exc:
                print(f"  {ticker}: ERROR — {exc}")
                results = []

            all_results.extend(results)
            done += 1

            if done % 10 == 0 or done == total:
                print(f"  [{done:3d}/{total}] {ticker:6s} — {len(results)} signals | total: {len(all_results)}")

    # Persist to Supabase in batches
    if all_results:
        print(f"\nInserting {len(all_results)} results into Supabase...")
        saved = 0
        for i in range(0, len(all_results), _BATCH_SIZE):
            saved += _bulk_insert(all_results[i : i + _BATCH_SIZE])
        print(f"Saved {saved} new rows (ON CONFLICT DO NOTHING for duplicates).")
    else:
        print("No signals detected.")

    _print_summary(all_results)
    return all_results


def _print_summary(results: list[dict]) -> None:
    if not results:
        return

    df = pd.DataFrame(results)
    ev  = df[df["exito_30d"].notna()]
    n   = len(df)
    ne  = len(ev)
    hr  = (ev["exito_30d"] == True).mean() * 100 if ne else 0.0

    gc  = df[df["tipo_evento"] == "golden_cross"]
    dc  = df[df["tipo_evento"] == "death_cross"]
    gce = gc[gc["exito_30d"].notna()]
    dce = dc[dc["exito_30d"].notna()]
    hrg = (gce["exito_30d"] == True).mean() * 100 if len(gce) else 0.0
    hrd = (dce["exito_30d"] == True).mean() * 100 if len(dce) else 0.0
    r30 = ev["retorno_30d"].mean() if ne else 0.0

    print(f"\n{'='*55}")
    print("BACKTEST RESULTS")
    print(f"{'='*55}")
    print(f"  Total signals detected : {n}")
    print(f"  Evaluable (30d data)   : {ne}")
    print(f"  Hit rate global        : {hr:.1f}%")
    print(f"  Hit rate Golden Cross  : {hrg:.1f}%")
    print(f"  Hit rate Death Cross   : {hrd:.1f}%")
    print(f"  Avg return 30d         : {r30:+.2f}%")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# CLI entry point: python -m src.backtester
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    env_tickers = os.getenv("TICKERS_BACKTEST")
    cli_tickers = env_tickers.split(",") if env_tickers else None

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    results = run_backtest(tickers=cli_tickers)

    if results:
        df = pd.DataFrame(results)
        cols = ["ticker", "tipo_evento", "fecha_senal", "scoring", "retorno_7d", "retorno_15d", "retorno_30d", "exito_30d"]
        print(df[cols].tail(20).to_string(index=False))
