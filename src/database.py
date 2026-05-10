"""Database module: all Supabase PostgreSQL operations via psycopg2.

Connection uses the Supabase Session Pooler endpoint, built dynamically
from SUPABASE_URL, SUPABASE_DB_PASSWORD and SUPABASE_POOLER_REGION.
"""

import logging
import re
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import psycopg2.extras

from src.config import SUPABASE_DB_PASSWORD, SUPABASE_URL
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _build_conn_str() -> str:
    """Build the psycopg2 connection string for Supabase Session Pooler.

    Extracts the project ref from SUPABASE_URL and combines it with the
    pooler region stored in SUPABASE_POOLER_REGION.

    Returns:
        A valid psycopg2 DSN string.

    Raises:
        ValueError: If SUPABASE_URL format is unexpected.
    """
    match = re.search(r'https://(.+)\.supabase\.co', SUPABASE_URL)
    if not match:
        raise ValueError(f"Cannot extract project ref from SUPABASE_URL: {SUPABASE_URL}")
    ref = match.group(1)
    region = os.getenv("SUPABASE_POOLER_REGION", "eu-west-1")
    host = f"aws-0-{region}.pooler.supabase.com"
    return (
        f"postgresql://postgres.{ref}:{SUPABASE_DB_PASSWORD}"
        f"@{host}:5432/postgres?sslmode=require"
    )


@contextmanager
def _conn():
    """Context manager that yields an open psycopg2 connection and closes it on exit.

    Yields:
        psycopg2 connection object.

    Raises:
        psycopg2.Error: On any connection or query failure.
    """
    conn = psycopg2.connect(_build_conn_str(), connect_timeout=10)
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def run_migrations() -> None:
    """Add new trading-level columns to the senales table if they don't exist.

    Safe to call multiple times — uses IF NOT EXISTS.
    """
    statements = [
        "ALTER TABLE senales ADD COLUMN IF NOT EXISTS precio_entrada  DECIMAL(10,2)",
        "ALTER TABLE senales ADD COLUMN IF NOT EXISTS precio_objetivo  DECIMAL(10,2)",
        "ALTER TABLE senales ADD COLUMN IF NOT EXISTS precio_stop      DECIMAL(10,2)",
        "ALTER TABLE senales ADD COLUMN IF NOT EXISTS ratio_rr         DECIMAL(4,2)",
        "ALTER TABLE senales ADD COLUMN IF NOT EXISTS descartar        BOOLEAN DEFAULT false",
    ]
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                for sql in statements:
                    cur.execute(sql)
            conn.commit()
        logger.info("run_migrations: all ALTER TABLE executed successfully")
    except Exception as exc:
        logger.error("run_migrations failed: %s", exc)
        raise


def insertar_senal(senal: dict) -> bool:
    """Insert a detected signal into the senales table.

    Uses ON CONFLICT DO NOTHING to skip duplicates (unique constraint on
    fecha_evento + ticker + tipo_evento).

    Args:
        senal: Dict with signal fields. Optional enrichment fields (sector,
               pe_ratio, etc.) are read with .get() and default to None.

    Returns:
        True if a new row was inserted, False if duplicate or error.
    """
    sql = """
        INSERT INTO senales (
            fecha_evento, ticker, tipo_evento,
            precio_cierre, sma_50, sma_200,
            rsi_14, adx_14, volumen_relativo, dist_sma_pct, scoring,
            sector, pe_ratio, eps, market_cap, earnings_date,
            news_sentiment, news_score, analisis_gemini,
            fed_funds_rate, cpi_inflacion,
            precio_entrada, precio_objetivo, precio_stop, ratio_rr, descartar
        ) VALUES (
            %(fecha_evento)s, %(ticker)s, %(tipo_evento)s,
            %(precio_cierre)s, %(sma_50)s, %(sma_200)s,
            %(rsi_14)s, %(adx_14)s, %(volumen_relativo)s, %(dist_sma_pct)s, %(scoring)s,
            %(sector)s, %(pe_ratio)s, %(eps)s, %(market_cap)s, %(earnings_date)s,
            %(news_sentiment)s, %(news_score)s, %(analisis_gemini)s,
            %(fed_funds_rate)s, %(cpi_inflacion)s,
            %(precio_entrada)s, %(precio_objetivo)s, %(precio_stop)s, %(ratio_rr)s, %(descartar)s
        )
        ON CONFLICT (fecha_evento, ticker, tipo_evento) DO NOTHING
    """
    params = {
        "fecha_evento":     senal.get("fecha_evento"),
        "ticker":           senal.get("ticker"),
        "tipo_evento":      senal.get("tipo_evento"),
        "precio_cierre":    senal.get("precio_cierre"),
        "sma_50":           senal.get("sma_50"),
        "sma_200":          senal.get("sma_200"),
        "rsi_14":           senal.get("rsi_14"),
        "adx_14":           senal.get("adx_14"),
        "volumen_relativo": senal.get("volumen_relativo"),
        "dist_sma_pct":     senal.get("dist_sma_pct"),
        "scoring":          senal.get("scoring"),
        "sector":           senal.get("sector"),
        "pe_ratio":         senal.get("pe_ratio"),
        "eps":              senal.get("eps"),
        "market_cap":       senal.get("market_cap"),
        "earnings_date":    senal.get("earnings_date"),
        "news_sentiment":   senal.get("news_sentiment"),
        "news_score":       senal.get("news_score"),
        "analisis_gemini":  senal.get("analisis_gemini"),
        "fed_funds_rate":   senal.get("fed_funds_rate"),
        "cpi_inflacion":    senal.get("cpi_inflacion"),
        "precio_entrada":   senal.get("precio_entrada"),
        "precio_objetivo":  senal.get("precio_objetivo"),
        "precio_stop":      senal.get("precio_stop"),
        "ratio_rr":         senal.get("ratio_rr"),
        "descartar":        senal.get("descartar", False),
    }
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                inserted = cur.rowcount == 1
                conn.commit()
        if inserted:
            logger.info("Inserted signal: %s %s %s", senal.get("ticker"),
                        senal.get("tipo_evento"), senal.get("fecha_evento"))
        else:
            logger.info("Duplicate signal skipped: %s %s %s", senal.get("ticker"),
                        senal.get("tipo_evento"), senal.get("fecha_evento"))
        return inserted
    except Exception as exc:
        logger.error("insertar_senal failed for %s: %s", senal.get("ticker"), exc)
        return False


def registrar_ejecucion(stats: dict) -> None:
    """Insert a pipeline execution summary into ejecuciones_log.

    Args:
        stats: Dict with keys tickers_analizados, senales_detectadas,
               errores, duracion_seg, estado.
    """
    sql = """
        INSERT INTO ejecuciones_log
            (tickers_analizados, senales_detectadas, errores, duracion_seg, estado)
        VALUES
            (%(tickers_analizados)s, %(senales_detectadas)s,
             %(errores)s, %(duracion_seg)s, %(estado)s)
    """
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, stats)
                conn.commit()
        logger.info("Execution log recorded: %s", stats)
    except Exception as exc:
        logger.error("registrar_ejecucion failed: %s", exc)


def actualizar_seguimiento(id: int, precio_7d: float, precio_30d: float) -> None:
    """Update a signal's follow-up prices and compute performance metrics.

    Fetches precio_cierre and tipo_evento from the existing row to compute:
      retorno_7d  = ((precio_7d  - precio_cierre) / precio_cierre) * 100
      retorno_30d = ((precio_30d - precio_cierre) / precio_cierre) * 100
      senal_ok    = retorno_30d > 0 for golden_cross, < 0 for death_cross

    Args:
        id:        Primary key of the senales row.
        precio_7d:  Price 7 days after the signal date.
        precio_30d: Price 30 days after the signal date.
    """
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT precio_cierre, tipo_evento FROM senales WHERE id = %s", (id,)
                )
                row = cur.fetchone()

            if row is None:
                logger.warning("actualizar_seguimiento: id %d not found", id)
                return

            precio_cierre = float(row["precio_cierre"])
            tipo_evento   = row["tipo_evento"]

            retorno_7d  = round(((precio_7d  - precio_cierre) / precio_cierre) * 100, 2)
            retorno_30d = round(((precio_30d - precio_cierre) / precio_cierre) * 100, 2)

            if tipo_evento == "golden_cross":
                senal_ok = retorno_30d > 0
            else:
                senal_ok = retorno_30d < 0

            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE senales
                       SET precio_7d   = %s,
                           precio_30d  = %s,
                           retorno_7d  = %s,
                           retorno_30d = %s,
                           senal_ok    = %s
                       WHERE id = %s""",
                    (precio_7d, precio_30d, retorno_7d, retorno_30d, senal_ok, id),
                )
                conn.commit()

        logger.info(
            "Follow-up updated id=%d: ret7d=%.2f%% ret30d=%.2f%% ok=%s",
            id, retorno_7d, retorno_30d, senal_ok,
        )
    except Exception as exc:
        logger.error("actualizar_seguimiento failed for id=%d: %s", id, exc)


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def _query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute a SELECT and return results as a pandas DataFrame.

    Uses RealDictCursor to avoid the pandas/SQLAlchemy compatibility warning
    when passing a raw psycopg2 connection to pd.read_sql_query.

    Args:
        sql:    SQL query string.
        params: Query parameters tuple.

    Returns:
        DataFrame with column names from the query, or empty DataFrame on error.
    """
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])
    except Exception as exc:
        logger.error("Query failed: %s | error: %s", sql[:60], exc)
        return pd.DataFrame()


def get_senales_recientes(horas: int = 48) -> pd.DataFrame:
    """Fetch signals detected within the last N hours, newest first.

    Args:
        horas: Lookback window in hours (default 48).

    Returns:
        DataFrame with all senales columns ordered by created_at DESC.
    """
    sql = """
        SELECT *
        FROM senales
        WHERE created_at >= NOW() - make_interval(hours => %s)
        ORDER BY created_at DESC
    """
    return _query_df(sql, (horas,))


def get_historico(
    ticker: str | None = None,
    tipo: str | None = None,
    scoring_min: int = 1,
) -> pd.DataFrame:
    """Fetch historical signals with optional filters for the Bitácora tab.

    Args:
        ticker:      Filter by ticker symbol (exact match). None = all tickers.
        tipo:        Filter by 'golden_cross' or 'death_cross'. None = both.
        scoring_min: Minimum scoring value (1–3, inclusive). Default 1.

    Returns:
        DataFrame ordered by fecha_evento DESC.
    """
    conditions = ["scoring >= %s"]
    params: list = [scoring_min]

    if ticker:
        conditions.append("ticker = %s")
        params.append(ticker.upper())
    if tipo:
        conditions.append("tipo_evento = %s")
        params.append(tipo)

    where = " AND ".join(conditions)
    sql = f"""
        SELECT id, fecha_evento, ticker, tipo_evento, precio_cierre,
               scoring, rsi_14, adx_14, volumen_relativo, dist_sma_pct,
               sector, news_sentiment, analisis_gemini,
               retorno_7d, retorno_30d, senal_ok
        FROM senales
        WHERE {where}
        ORDER BY fecha_evento DESC
    """
    return _query_df(sql, tuple(params))


def get_stats_rendimiento() -> dict:
    """Compute aggregated performance metrics for the Track Record tab.

    Only includes signals where senal_ok IS NOT NULL (i.e., ≥30 days old
    and already tracked by tracker.py).

    Returns:
        Dict with keys: total_senales (int), hit_rate (float, 0-100),
        retorno_medio_30d (float), mejor_sector (str).
        Returns zeros / 'N/A' when no trackable data exists.
    """
    default = {
        "total_senales": 0,
        "hit_rate": 0.0,
        "retorno_medio_30d": 0.0,
        "mejor_sector": "N/A",
    }

    sql_agg = """
        SELECT
            COUNT(*)                                                    AS total_senales,
            ROUND(AVG(CASE WHEN senal_ok THEN 100.0 ELSE 0.0 END), 2)  AS hit_rate,
            ROUND(AVG(retorno_30d), 2)                                  AS retorno_medio_30d
        FROM senales
        WHERE senal_ok IS NOT NULL
    """
    sql_sector = """
        SELECT sector
        FROM senales
        WHERE senal_ok IS NOT NULL
          AND sector IS NOT NULL
          AND sector NOT IN ('N/A', '')
        GROUP BY sector
        ORDER BY AVG(CASE WHEN senal_ok THEN 1.0 ELSE 0.0 END) DESC
        LIMIT 1
    """
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql_agg)
                agg_rows = cur.fetchall()
                cur.execute(sql_sector)
                sec_rows = cur.fetchall()
        agg_df = pd.DataFrame([dict(r) for r in agg_rows]) if agg_rows else pd.DataFrame()
        sec_df = pd.DataFrame([dict(r) for r in sec_rows]) if sec_rows else pd.DataFrame()

        if agg_df.empty or int(agg_df["total_senales"].iloc[0]) == 0:
            return default

        mejor_sector = sec_df["sector"].iloc[0] if not sec_df.empty else "N/A"

        return {
            "total_senales":    int(agg_df["total_senales"].iloc[0]),
            "hit_rate":         float(agg_df["hit_rate"].iloc[0] or 0),
            "retorno_medio_30d": float(agg_df["retorno_medio_30d"].iloc[0] or 0),
            "mejor_sector":     mejor_sector,
        }
    except Exception as exc:
        logger.error("get_stats_rendimiento failed: %s", exc)
        return default


def get_ultima_ejecucion() -> dict | None:
    """Fetch the most recent pipeline execution log entry.

    Returns:
        Dict with keys: created_at, senales_detectadas, duracion_seg, estado.
        None if the table is empty or on error.
    """
    sql = """
        SELECT created_at, senales_detectadas, duracion_seg, estado
        FROM ejecuciones_log
        ORDER BY created_at DESC
        LIMIT 1
    """
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                row = cur.fetchone()
        return dict(row) if row else None
    except Exception as exc:
        logger.error("get_ultima_ejecucion failed: %s", exc)
        return None


def get_senales_para_seguimiento() -> pd.DataFrame:
    """Fetch signals older than 30 days that still have no precio_30d.

    Used by tracker.py to determine which signals need follow-up price data.

    Returns:
        DataFrame with id, fecha_evento, ticker, precio_cierre columns.
    """
    sql = """
        SELECT id, fecha_evento, ticker, precio_cierre, tipo_evento
        FROM senales
        WHERE precio_30d IS NULL
          AND ticker != 'TEST'
          AND fecha_evento < NOW() - INTERVAL '30 days'
        ORDER BY fecha_evento ASC
    """
    return _query_df(sql)
