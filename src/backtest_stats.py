"""Backtest statistics: loads backtest_resultados from Supabase and computes metrics."""

import logging
import os
import re
from contextlib import contextmanager

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from src.config import SUPABASE_DB_PASSWORD, SUPABASE_URL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB connection (mirrors database.py to avoid circular imports)
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


def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])
    except Exception as exc:
        logger.error("backtest_stats query failed: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def get_backtest_df(
    años: list[int] | None = None,
    sectores: list[str] | None = None,
    tipo: str | None = None,
    scoring_min: int = 1,
) -> pd.DataFrame:
    """Fetch backtest_resultados rows with optional filters.

    Args:
        años:        List of years to include; None = all.
        sectores:    List of sector strings; None = all.
        tipo:        'golden_cross', 'death_cross', or None = both.
        scoring_min: Minimum scoring (1–3).

    Returns:
        DataFrame ordered by fecha_senal ASC.
    """
    conditions = ["scoring >= %s"]
    params: list = [scoring_min]

    if años:
        placeholders = ",".join(["%s"] * len(años))
        conditions.append(f"año IN ({placeholders})")
        params.extend(años)
    if sectores:
        placeholders = ",".join(["%s"] * len(sectores))
        conditions.append(f"sector IN ({placeholders})")
        params.extend(sectores)
    if tipo and tipo not in ("Todas", "all"):
        conditions.append("tipo_evento = %s")
        params.append(tipo)

    where = " AND ".join(conditions)
    sql = f"""
        SELECT *
        FROM backtest_resultados
        WHERE {where}
        ORDER BY fecha_senal ASC
    """
    return _query(sql, tuple(params))


def get_all_sectors() -> list[str]:
    """Return distinct non-null sector values present in backtest_resultados."""
    df = _query(
        "SELECT DISTINCT sector FROM backtest_resultados "
        "WHERE sector IS NOT NULL ORDER BY sector"
    )
    if df.empty:
        return []
    return sorted(df["sector"].dropna().tolist())


def get_all_years() -> list[int]:
    """Return sorted list of years present in backtest_resultados."""
    df = _query(
        "SELECT DISTINCT año FROM backtest_resultados "
        "WHERE año IS NOT NULL ORDER BY año"
    )
    if df.empty:
        return []
    return sorted(df["año"].dropna().astype(int).tolist())


def backtest_table_exists() -> bool:
    """Return True if backtest_resultados table exists and has at least one row."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM backtest_resultados LIMIT 1"
                )
                return cur.fetchone() is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> dict:
    """Compute all backtest statistics from a DataFrame of backtest_resultados rows.

    Args:
        df: DataFrame as returned by get_backtest_df (may include rows without exito_30d).

    Returns:
        Dict with the following keys:
            total_senales, hit_rate_global, hit_rate_golden, hit_rate_death,
            retorno_medio_7d, retorno_medio_15d, retorno_medio_30d,
            hit_rate_por_scoring, hit_rate_por_sector, hit_rate_por_año,
            mejor_ticker, peor_ticker, senales_por_año_media,
            equity_curve (list of {"fecha": str, "capital": float}).
    """
    empty = {
        "total_senales": 0,
        "hit_rate_global": 0.0,
        "hit_rate_golden": 0.0,
        "hit_rate_death": 0.0,
        "retorno_medio_7d": 0.0,
        "retorno_medio_15d": 0.0,
        "retorno_medio_30d": 0.0,
        "hit_rate_por_scoring": {1: 0.0, 2: 0.0, 3: 0.0},
        "hit_rate_por_sector": {},
        "hit_rate_por_año": {},
        "mejor_ticker": "N/A",
        "peor_ticker": "N/A",
        "senales_por_año_media": 0.0,
        "equity_curve": [],
    }

    if df.empty:
        return empty

    # Ensure types
    df = df.copy()
    if "fecha_senal" in df.columns:
        df["fecha_senal"] = pd.to_datetime(df["fecha_senal"])
    if "exito_30d" in df.columns:
        df["exito_30d"] = df["exito_30d"].astype(object)

    ev = df[df["exito_30d"].notna()].copy()

    total = len(df)
    n_ev  = len(ev)

    def _hr(subset: pd.DataFrame) -> float:
        if subset.empty:
            return 0.0
        return round((subset["exito_30d"] == True).mean() * 100, 1)

    hit_global = _hr(ev)

    gc_ev = ev[ev["tipo_evento"] == "golden_cross"]
    dc_ev = ev[ev["tipo_evento"] == "death_cross"]

    hit_golden = _hr(gc_ev)
    hit_death  = _hr(dc_ev)

    # Strategy-adjusted returns: death_cross is a short → negate raw market return
    def _adj(row, col):
        v = row[col]
        return v if row["tipo_evento"] == "golden_cross" else -v

    if n_ev and "retorno_7d" in ev.columns and "retorno_30d" in ev.columns:
        ev_adj = ev.copy()
        for c in ["retorno_7d", "retorno_15d", "retorno_30d"]:
            if c in ev_adj.columns:
                ev_adj[c] = ev_adj.apply(lambda r: _adj(r, c), axis=1)
        r7  = round(ev_adj["retorno_7d"].mean(),  2) if "retorno_7d"  in ev_adj.columns else 0.0
        r15 = round(ev_adj["retorno_15d"].mean(), 2) if "retorno_15d" in ev_adj.columns else 0.0
        r30 = round(ev_adj["retorno_30d"].mean(), 2) if "retorno_30d" in ev_adj.columns else 0.0
    else:
        r7 = r15 = r30 = 0.0

    # Hit rate by scoring
    hr_scoring: dict[int, float] = {}
    for s in [1, 2, 3]:
        sub = ev[ev["scoring"] == s]
        hr_scoring[s] = _hr(sub)

    # Hit rate by sector
    hr_sector: dict[str, float] = {}
    if "sector" in ev.columns:
        for sec, grp in ev.groupby("sector"):
            if sec and str(sec) not in ("nan", "N/A", "Unknown", ""):
                hr_sector[str(sec)] = _hr(grp)

    # Hit rate by year
    hr_año: dict[int, float] = {}
    if "año" in ev.columns:
        for yr, grp in ev.groupby("año"):
            hr_año[int(yr)] = _hr(grp)

    # Best / worst ticker by hit rate (min 3 signals to qualify)
    mejor_ticker = "N/A"
    peor_ticker  = "N/A"
    if "ticker" in ev.columns:
        ticker_hr = (
            ev.groupby("ticker")["exito_30d"]
            .agg(hr=lambda x: (x == True).mean() * 100, n="count")
            .query("n >= 3")
        )
        if not ticker_hr.empty:
            mejor_ticker = str(ticker_hr["hr"].idxmax())
            peor_ticker  = str(ticker_hr["hr"].idxmin())

    # Signals per year (average)
    senales_por_año = 0.0
    if "año" in df.columns and not df.empty:
        por_año = df.groupby("año").size()
        senales_por_año = round(por_año.mean(), 1)

    # Equity curve: invest $1000 in each evaluable signal chronologically
    # Death cross = short strategy → negate raw return to get strategy P&L
    equity: list[dict] = []
    if n_ev > 0 and "retorno_30d" in ev.columns:
        ev_sorted = ev.dropna(subset=["retorno_30d"]).sort_values("fecha_senal")
        capital = 1000.0
        for _, row in ev_sorted.iterrows():
            ret = float(row["retorno_30d"])
            if row.get("tipo_evento") == "death_cross":
                ret = -ret
            capital = capital * (1 + ret / 100)
            fecha = (
                row["fecha_senal"].strftime("%Y-%m-%d")
                if hasattr(row["fecha_senal"], "strftime")
                else str(row["fecha_senal"])[:10]
            )
            equity.append({"fecha": fecha, "capital": round(capital, 2)})

    return {
        "total_senales":        total,
        "hit_rate_global":      hit_global,
        "hit_rate_golden":      hit_golden,
        "hit_rate_death":       hit_death,
        "retorno_medio_7d":     r7,
        "retorno_medio_15d":    r15,
        "retorno_medio_30d":    r30,
        "hit_rate_por_scoring": hr_scoring,
        "hit_rate_por_sector":  hr_sector,
        "hit_rate_por_año":     hr_año,
        "mejor_ticker":         mejor_ticker,
        "peor_ticker":          peor_ticker,
        "senales_por_año_media": senales_por_año,
        "equity_curve":         equity,
    }
