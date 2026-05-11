"""Historical backtesting module: replays signal detection across full price history."""

import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import timedelta

import numpy as np
import pandas as pd
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
            atr_valor       DECIMAL(5,2),
            vol_ratio       DECIMAL(5,2),
            dist_sma_pct    DECIMAL(5,2),
            stop_loss       DECIMAL(10,2),
            fecha_salida    DATE,
            precio_salida   DECIMAL(10,2),
            exit_type       VARCHAR(20),
            precio_7d       DECIMAL(10,2),
            precio_15d      DECIMAL(10,2),
            precio_30d      DECIMAL(10,2),
            retorno_7d      DECIMAL(6,2),
            retorno_15d     DECIMAL(6,2),
            retorno_30d     DECIMAL(6,2),
            retorno_sl      DECIMAL(6,2),
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


def create_backtest_history_table() -> None:
    """Create backtest_history table to store each backtest run."""
    sql = """
        CREATE TABLE IF NOT EXISTS backtest_history (
            id                  SERIAL PRIMARY KEY,
            fecha_ejecucion     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tickers_procesados  INTEGER,
            senales_total       INTEGER,
            senales_golden      INTEGER,
            senales_death       INTEGER,
            hit_rate_global     DECIMAL(5,2),
            hit_rate_golden     DECIMAL(5,2),
            hit_rate_death      DECIMAL(5,2),
            retorno_promedio    DECIMAL(6,2),
            capital_inicial     DECIMAL(10,2) DEFAULT 1000.00,
            capital_final       DECIMAL(10,2),
            sl_hits             INTEGER,
            sl_avg_return       DECIMAL(6,2),
            analisis_ia         TEXT,
            recomendaciones     TEXT,
            scoring_recomendado INTEGER,
            duracion_seg        INTEGER
        )
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print("  Table backtest_history: OK")


def _generar_analisis_ia(results: list[dict], stats: dict) -> tuple[str, str, int]:
    """Generate AI analysis of backtest results using Gemini."""
    try:
        import google.genai as genai
        from src.config import GEMINI_API_KEY

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Prepare data summary
        total_signals = len(results)
        golden_signals = len([r for r in results if r['tipo_evento'] == 'golden_cross'])
        death_signals = len([r for r in results if r['tipo_evento'] == 'death_cross'])
        sl_hits = len([r for r in results if r.get('exit_type') == 'stop_loss'])
        sl_avg_return = stats.get('avg_sl_return', 0)

        # Sample problematic signals
        death_cross_failures = [r for r in results if r['tipo_evento'] == 'death_cross' and r.get('exit_type') == 'stop_loss']
        death_cross_failures = death_cross_failures[:5]  # Limit to 5 examples

        prompt = f"""
        Analiza estos resultados de backtesting de una estrategia de trading de Golden Cross/Death Cross con Stop Loss dinámico a 2×ATR:

        DATOS GENERALES:
        - Señales totales: {total_signals}
        - Golden Cross: {golden_signals} ({stats.get('hit_rate_golden', 0):.1f}% éxito)
        - Death Cross: {death_signals} ({stats.get('hit_rate_death', 0):.1f}% éxito)
        - Hit Rate Global: {stats.get('hit_rate_global', 0):.1f}%
        - Stop Loss hits: {sl_hits}/{total_signals} ({sl_hits/total_signals*100:.1f}%)
        - Retorno promedio SL: {sl_avg_return:.2f}%
        - Retorno promedio 30d: {stats.get('retorno_medio_30d', 0):.2f}%

        EJEMPLOS DE DEATH CROSS CON PROBLEMAS:
        {chr(10).join([f"- {r['ticker']}: SL={r.get('stop_loss', 0):.2f}, Retorno={r.get('retorno_sl', 0):.2f}%" for r in death_cross_failures])}

        POR FAVOR PROPORCIONA:

        1. ANÁLISIS: ¿Cuáles son las principales causas del bajo rendimiento de Death Cross?

        2. RECOMENDACIONES: ¿Cómo podríamos ajustar la estrategia para mejorar?

        3. SCORING RECOMENDADO: ¿Qué nivel de scoring mínimo recomiendas usar? (1-3)

        Responde de forma concisa pero completa, enfocándote en insights accionables.
        """

        response = client.models.generate_content(
            model='gemini-1.5-pro',
            contents=prompt
        )
        analisis_completo = response.text.strip()

        # Extract recommendations and scoring
        recomendaciones = ""
        scoring_recomendado = 1

        if "recomendaciones" in analisis_completo.lower():
            # Simple extraction - could be improved
            lines = analisis_completo.split('\n')
            for line in lines:
                if "scoring" in line.lower() and any(char.isdigit() for char in line):
                    for char in line:
                        if char.isdigit() and 1 <= int(char) <= 3:
                            scoring_recomendado = int(char)
                            break

        return analisis_completo, recomendaciones, scoring_recomendado

    except Exception as e:
        logger.warning(f"Error generating AI analysis: {e}")
        return "Análisis no disponible - error en API de IA", "Revisar configuración de Gemini API", 1


def _guardar_backtest_history(stats: dict, results: list[dict], tickers_count: int, duration_sec: int) -> None:
    """Save backtest run to history table."""
    try:
        # Calculate final capital from equity curve
        equity_curve = stats.get('equity_curve', [])
        capital_final = 1000.0  # Default initial
        if equity_curve:
            capital_final = equity_curve[-1]['capital']

        # Count signals by type
        golden_count = len([r for r in results if r['tipo_evento'] == 'golden_cross'])
        death_count = len([r for r in results if r['tipo_evento'] == 'death_cross'])

        # Generate AI analysis
        analisis_ia, recomendaciones, scoring_recomendado = _generar_analisis_ia(results, stats)

        sql = """
            INSERT INTO backtest_history (
                tickers_procesados, senales_total, senales_golden, senales_death,
                hit_rate_global, hit_rate_golden, hit_rate_death, retorno_promedio,
                capital_final, sl_hits, sl_avg_return, analisis_ia, recomendaciones,
                scoring_recomendado, duracion_seg
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        params = (
            tickers_count,
            len(results),
            golden_count,
            death_count,
            float(stats.get('hit_rate_global', 0)),
            float(stats.get('hit_rate_golden', 0)),
            float(stats.get('hit_rate_death', 0)),
            float(stats.get('retorno_medio_30d', 0)),
            float(capital_final),
            int(stats.get('sl_hits', 0)),
            float(stats.get('avg_sl_return', 0)),
            analisis_ia,
            recomendaciones,
            int(scoring_recomendado),
            duration_sec
        )

        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()

        print(f"✅ Backtest history saved - Capital: $1,000 → ${capital_final:.2f}")

    except Exception as e:
        logger.error(f"Error saving backtest history: {e}")


def _bulk_insert(rows: list[dict]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO backtest_resultados (
            ticker, sector, tipo_evento, fecha_senal, precio_entrada,
            scoring, adx_valor, rsi_valor, atr_valor, vol_ratio, dist_sma_pct,
            stop_loss, fecha_salida, precio_salida, exit_type,
            precio_7d, precio_15d, precio_30d,
            retorno_7d, retorno_15d, retorno_30d, retorno_sl, exito_30d, año
        ) VALUES %s
        ON CONFLICT (ticker, tipo_evento, fecha_senal) DO NOTHING
    """
    values = [
        (
            r["ticker"], r["sector"], r["tipo_evento"], r["fecha_senal"],
            r["precio_entrada"], r["scoring"], r["adx_valor"], r["rsi_valor"],
            r["atr_valor"], r["vol_ratio"], r["dist_sma_pct"],
            r["stop_loss"], r["fecha_salida"], r["precio_salida"], r["exit_type"],
            r["precio_7d"], r["precio_15d"], r["precio_30d"],
            r["retorno_7d"], r["retorno_15d"], r["retorno_30d"],
            r["retorno_sl"], r["exito_30d"], r["año"],
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


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=n - 1, min_periods=n).mean()
    avg_loss = loss.ewm(com=n - 1, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    up   = high - high.shift()
    down = low.shift() - low
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    atr       = tr.ewm(com=n - 1, min_periods=n).mean()
    plus_di   = 100 * plus_dm.ewm(com=n - 1, min_periods=n).mean() / atr
    minus_di  = 100 * minus_dm.ewm(com=n - 1, min_periods=n).mean() / atr
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(com=n - 1, min_periods=n).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, min_periods=n).mean()


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append SMA50/200, RSI14, ADX14, ATR14, VOL_MA20 and drop NaN warm-up rows."""
    df = df.copy()
    df["SMA_50"]  = _sma(df["Close"], 50)
    df["SMA_200"] = _sma(df["Close"], 200)
    df["RSI_14"]  = _rsi(df["Close"], 14)
    df["ADX_14"]  = _adx(df["High"], df["Low"], df["Close"], 14)
    df["ATR_14"]  = _atr(df["High"], df["Low"], df["Close"], 14)
    df["VOL_MA20"] = _sma(df["Volume"], 20)
    df.dropna(subset=["SMA_50", "SMA_200", "RSI_14", "ADX_14", "ATR_14", "VOL_MA20"], inplace=True)
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
    highs   = df["High"].to_numpy(dtype=float)
    lows    = df["Low"].to_numpy(dtype=float)
    sma50   = df["SMA_50"].to_numpy(dtype=float)
    sma200  = df["SMA_200"].to_numpy(dtype=float)
    adx_arr = df["ADX_14"].to_numpy(dtype=float)
    atr_arr = df["ATR_14"].to_numpy(dtype=float)
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
        atr_value   = float(atr_arr[i])
        
        # Calculate Stop Loss Level: 2 * ATR below entry for long (golden), above entry for short (death)
        stop_loss = entry_price - (2 * atr_value) if golden else entry_price + (2 * atr_value)
        
        dist_sma    = float(((sma50[i] - sma200[i]) / sma200[i]) * 100)
        scoring     = calcular_scoring(dist_sma)

        # ---- Search for Stop Loss Hit ----
        sl_hit = False
        sl_exit_date = None
        sl_exit_price = None
        
        # Search forward up to 30 days or end of data
        for j in range(i + 1, min(i + 30, len(df))):
            if golden:
                # For long: SL is below, trigger on low <= SL
                if lows[j] <= stop_loss:
                    sl_hit = True
                    sl_exit_date = dates[j]
                    sl_exit_price = stop_loss  # Assume execution at SL level
                    break
            else:
                # For short: SL is above, trigger on high >= SL
                if highs[j] >= stop_loss:
                    sl_hit = True
                    sl_exit_date = dates[j]
                    sl_exit_price = stop_loss  # Assume execution at SL level
                    break

        # ---- Calculate exit price and return ----
        exit_price = sl_exit_price if sl_hit and sl_exit_price else None
        retorno_sl = None
        if exit_price is not None:
            retorno_sl = round((exit_price - entry_price) / entry_price * 100, 2) if golden else \
                        round((entry_price - exit_price) / entry_price * 100, 2)
        
        # ---- Calculate fixed-period returns (7d, 15d, 30d) ----
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
            "atr_valor":      _f(atr_value),
            "vol_ratio":      _f(vol_ratio),
            "dist_sma_pct":   round(float(dist_sma), 4),
            "stop_loss":      _f(stop_loss),
            "fecha_salida":   sl_exit_date.date() if sl_exit_date else None,
            "precio_salida":  _f(sl_exit_price) if sl_exit_price else None,
            "exit_type":      "stop_loss" if sl_hit else "timeout",
            "precio_7d":      _f(p7),
            "precio_15d":     _f(p15),
            "precio_30d":     _f(p30),
            "retorno_7d":     r7,
            "retorno_15d":    r15,
            "retorno_30d":    r30,
            "retorno_sl":     retorno_sl,
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
    import time
    start_time = time.time()

    if tickers is None:
        tickers = TICKERS

    print(f"\n{'='*55}")
    print(f"DINERITO BACKTESTER — {len(tickers)} tickers from {BACKTEST_START}")
    print(f"{'='*55}")

    create_backtest_table()
    create_backtest_history_table()

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

    # Generate and print summary with AI analysis
    stats = _print_summary_with_ai(all_results)

    # Save backtest history
    duration_sec = int(time.time() - start_time)
    _guardar_backtest_history(stats, all_results, len(tickers), duration_sec)

    return all_results


def _print_summary_with_ai(results: list[dict]) -> dict:
    """Print summary and return stats dict for history saving."""
    if not results:
        return {}

    df = pd.DataFrame(results)
    ev  = df[df["exito_30d"].notna()]
    n   = len(df)
    ne  = len(ev)
    hr  = (ev["exito_30d"] == True).mean() * 100 if ne else 0.0

    # Stop Loss statistics
    sl_hits = df[df["exit_type"] == "stop_loss"]
    sl_count = len(sl_hits)
    avg_sl_return = df[df["retorno_sl"].notna()]["retorno_sl"].mean() if len(df[df["retorno_sl"].notna()]) > 0 else 0.0

    gc  = df[df["tipo_evento"] == "golden_cross"]
    dc  = df[df["tipo_evento"] == "death_cross"]
    gce = gc[gc["exito_30d"].notna()]
    dce = dc[dc["exito_30d"].notna()]
    hrg = (gce["exito_30d"] == True).mean() * 100 if len(gce) else 0.0
    hrd = (dce["exito_30d"] == True).mean() * 100 if len(dce) else 0.0
    r30 = ev["retorno_30d"].mean() if ne else 0.0

    # Generate AI analysis
    print(f"\n{'='*55}")
    print("BACKTEST RESULTS")
    print(f"{'='*55}")
    print(f"  Total signals detected : {n}")
    print(f"  Evaluable (30d data)   : {ne}")
    print(f"  Hit rate global        : {hr:.1f}%")
    print(f"  Hit rate Golden Cross  : {hrg:.1f}%")
    print(f"  Hit rate Death Cross   : {hrd:.1f}%")
    print(f"  Avg return 30d         : {r30:+.2f}%")
    print(f"  {'-'*55}")
    print(f"  Stop Loss hits         : {sl_count}/{n}")
    print(f"  Avg SL return          : {avg_sl_return:+.2f}%")
    print(f"{'='*55}\n")

    # Generate AI analysis
    print("🤖 GENERANDO ANÁLISIS CON IA...")
    try:
        analisis_ia, recomendaciones, scoring_recomendado = _generar_analisis_ia(results, {
            'hit_rate_global': hr,
            'hit_rate_golden': hrg,
            'hit_rate_death': hrd,
            'retorno_medio_30d': r30,
            'sl_hits': sl_count,
            'avg_sl_return': avg_sl_return
        })
    except Exception as e:
        print(f"⚠️ Error en análisis IA: {e}")
        analisis_ia = f"Análisis automático no disponible. Hit Rate Global: {hr:.1f}%, Golden Cross: {hrg:.1f}%, Death Cross: {hrd:.1f}%. Stop Loss hits: {sl_count}/{n}."
        recomendaciones = "Revisar configuración de API de IA para análisis avanzado."
        scoring_recomendado = 2 if hr > 50 else 1

    print("📊 ANÁLISIS DE IA:")
    print("-" * 55)
    print(analisis_ia)
    print("-" * 55)
    print(f"🎯 SCORING RECOMENDADO: {scoring_recomendado}")
    print(f"💡 RECOMENDACIONES: {recomendaciones}")
    print(f"{'='*55}\n")

    return {
        'hit_rate_global': hr,
        'hit_rate_golden': hrg,
        'hit_rate_death': hrd,
        'retorno_medio_30d': r30,
        'sl_hits': sl_count,
        'avg_sl_return': avg_sl_return,
        'analisis_ia': analisis_ia,
        'recomendaciones': recomendaciones,
        'scoring_recomendado': scoring_recomendado
    }


def _print_summary(results: list[dict]) -> None:
    if not results:
        return

    df = pd.DataFrame(results)
    ev  = df[df["exito_30d"].notna()]
    n   = len(df)
    ne  = len(ev)
    hr  = (ev["exito_30d"] == True).mean() * 100 if ne else 0.0

    # Stop Loss statistics
    sl_hits = df[df["exit_type"] == "stop_loss"]
    sl_count = len(sl_hits)
    avg_sl_return = df[df["retorno_sl"].notna()]["retorno_sl"].mean() if len(df[df["retorno_sl"].notna()]) > 0 else 0.0

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
    print(f"  {'-'*55}")
    print(f"  Stop Loss hits         : {sl_count}/{n}")
    print(f"  Avg SL return          : {avg_sl_return:+.2f}%")
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
