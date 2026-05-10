"""DINERITO — Dashboard Streamlit con 6 pestañas.

Pestaña 1: Radar Activo    — señales recientes con análisis IA
Pestaña 2: Laboratorio     — gráfico de velas + SMA + RSI por ticker
Pestaña 3: Bitácora        — tabla histórica filtrable + exportar CSV
Pestaña 4: Track Record    — métricas de rendimiento y hit-rate
Pestaña 5: Manual          — guía completa del sistema
Pestaña 6: Backtesting     — análisis histórico 2015-hoy con equity curve
"""

import io
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from src.config import TICKERS
from src.database import (
    get_historico,
    get_senales_recientes,
    get_stats_rendimiento,
    get_ultima_ejecucion,
)
from src.backtest_stats import (
    backtest_table_exists,
    compute_stats,
    get_all_sectors,
    get_all_years,
    get_backtest_df,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DINERITO · Plataforma Bursátil IA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

COLORS = {
    "primary":   "#00d4aa",
    "bullish":   "#00ff88",
    "bearish":   "#ff4444",
    "sma50":     "#58a6ff",
    "sma200":    "#f0883e",
    "rsi":       "#bc8cff",
    "gold":      "#ffd700",
    "neutral":   "#8b9cbc",
    "steelblue": "#388bfd",
    "tomato":    "#ff6b6b",
}

SENTIMENT_COLORS = {
    "Bullish": "#00ff88",
    "Bearish": "#ff4444",
    "Neutral": "#8b9cbc",
}

# ---------------------------------------------------------------------------
# CSS global
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

    /* Ocultar branding Streamlit */
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
    header     {visibility: hidden;}

    /* Tabs — estilo pill */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #161b22;
        border-radius: 12px;
        padding: 5px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8b9cbc;
        font-weight: 500;
        padding: 6px 18px;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: #1f6feb !important;
        color: white !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8b9cbc !important;
        font-size: 0.75em !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-size: 1.7em !important;
        font-weight: 700 !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #161b22;
        border: 1px solid #30363d !important;
        border-radius: 10px;
        margin-bottom: 8px;
    }
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #c9d1d9 !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: #1f6feb;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 22px;
    }
    .stDownloadButton > button:hover { background: #388bfd; }

    /* Inputs / selects */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #161b22 !important;
        border-color: #30363d !important;
    }

    /* Radio */
    .stRadio > div { gap: 6px; }

    /* Slider thumb */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #00d4aa !important;
        border-color: #00d4aa !important;
    }

    /* Alert banners */
    .stAlert { border-radius: 10px; border: none; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #30363d; border-radius: 10px; overflow: hidden; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track  { background: #0e1117; }
    ::-webkit-scrollbar-thumb  { background: #30363d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #484f58; }

    /* Headers */
    h1, h2, h3 { color: #e6edf3 !important; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header de marca
# ---------------------------------------------------------------------------

def _render_header() -> None:
    st.markdown("""
    <div style="
        display:flex; align-items:center; justify-content:space-between;
        background:#161b22; border:1px solid #30363d; border-radius:14px;
        padding:16px 28px; margin-bottom:24px;
    ">
        <div style="display:flex; align-items:center; gap:14px;">
            <span style="font-size:2em; line-height:1;">📈</span>
            <div>
                <h1 style="margin:0; color:#e6edf3; font-size:1.5em; font-weight:700;
                           letter-spacing:-0.03em; line-height:1.1;">DINERITO</h1>
                <p style="margin:2px 0 0; color:#8b9cbc; font-size:0.75em; font-weight:500;">
                    Plataforma Bursátil IA &nbsp;·&nbsp; 75 activos monitorizados
                </p>
            </div>
        </div>
        <div style="text-align:right;">
            <p style="margin:0; color:#00d4aa; font-size:0.82em; font-weight:700;
                      letter-spacing:0.04em;">● PIPELINE ACTIVO</p>
            <p style="margin:2px 0 0; color:#8b9cbc; font-size:0.72em;">
                Actualización diaria · 22:00 EST
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _safe_str(val, default: str = "N/A") -> str:
    if val is None:
        return default
    if isinstance(val, float) and pd.isna(val):
        return default
    s = str(val).strip()
    return s if s and s != "nan" else default


def _scoring_dots(scoring: int, color: str) -> str:
    filled = "●" * scoring
    empty  = "○" * (3 - scoring)
    return (
        f'<span style="color:{color}; font-size:1.1em; letter-spacing:2px;">{filled}</span>'
        f'<span style="color:#30363d; font-size:1.1em; letter-spacing:2px;">{empty}</span>'
    )


def _rsi_color(rsi: float, tipo: str) -> str:
    if tipo == "golden_cross":
        return "#ff9f43" if rsi > 65 else COLORS["bullish"]
    return "#ff9f43" if rsi < 35 else COLORS["bearish"]


def _card_style(tipo: str) -> dict:
    if tipo == "golden_cross":
        return {
            "bg":     "#0d1f17",
            "border": "#1a3a2a",
            "accent": COLORS["bullish"],
            "glow":   "0 0 28px rgba(0,255,136,0.10)",
            "emoji":  "🟢",
            "label":  "GOLDEN CROSS",
        }
    return {
        "bg":     "#1f0d0d",
        "border": "#3a1a1a",
        "accent": COLORS["bearish"],
        "glow":   "0 0 28px rgba(255,68,68,0.10)",
        "emoji":  "🔴",
        "label":  "DEATH CROSS",
    }


def _render_signal_card(senal: dict) -> None:
    tipo      = senal.get("tipo_evento", "")
    ticker    = senal.get("ticker", "?")
    precio    = float(senal.get("precio_cierre") or 0)
    scoring   = int(senal.get("scoring") or 1)
    rsi       = float(senal.get("rsi_14") or 0)
    adx       = float(senal.get("adx_14") or 0)
    sector    = _safe_str(senal.get("sector"), "N/A")
    fecha     = str(senal.get("fecha_evento", ""))[:10]
    sentiment = _safe_str(senal.get("news_sentiment"), "N/A")
    analisis  = _safe_str(senal.get("analisis_gemini"), "Sin análisis disponible.")

    style    = _card_style(tipo)
    dots     = _scoring_dots(scoring, style["accent"])
    rsi_col  = _rsi_color(rsi, tipo)
    sent_col = SENTIMENT_COLORS.get(sentiment, COLORS["neutral"])

    st.markdown(f"""
    <div style="
        background:{style['bg']};
        border:1px solid {style['border']};
        border-left:4px solid {style['accent']};
        border-radius:12px;
        padding:20px 22px;
        margin-bottom:8px;
        box-shadow:{style['glow']};
    ">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:14px;">
            <div>
                <h3 style="margin:0; color:#e6edf3; font-size:1.15em; font-weight:700; letter-spacing:-0.01em;">
                    {style['emoji']} {ticker}
                </h3>
                <span style="color:{style['accent']}; font-size:0.75em; font-weight:700; letter-spacing:0.1em;">
                    {style['label']}
                </span>
            </div>
            <div style="text-align:right;">
                <div style="margin-bottom:4px;">{dots}</div>
                <p style="margin:0; color:#8b9cbc; font-size:0.73em; font-weight:500;">{fecha}</p>
            </div>
        </div>
        <div style="display:flex; gap:7px; flex-wrap:wrap;">
            <span style="background:#1c2333; border:1px solid #30363d; border-radius:20px;
                         padding:3px 11px; color:#e6edf3; font-size:0.78em; font-weight:600;">
                💰 ${precio:.2f}
            </span>
            <span style="background:#1c2333; border:1px solid {rsi_col}33; border-radius:20px;
                         padding:3px 11px; color:{rsi_col}; font-size:0.78em; font-weight:600;">
                RSI {rsi:.0f}
            </span>
            <span style="background:#1c2333; border:1px solid #30363d; border-radius:20px;
                         padding:3px 11px; color:#e6edf3; font-size:0.78em; font-weight:600;">
                ADX {adx:.0f}
            </span>
            <span style="background:#1c2333; border:1px solid #30363d; border-radius:20px;
                         padding:3px 11px; color:#8b9cbc; font-size:0.78em;">
                🏢 {sector}
            </span>
            <span style="background:#1c2333; border:1px solid {sent_col}44; border-radius:20px;
                         padding:3px 11px; color:{sent_col}; font-size:0.78em; font-weight:600;">
                📰 {sentiment}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"🤖 Ver análisis IA completo — {ticker}"):
        st.markdown(analisis)


# ---------------------------------------------------------------------------
# Shared column config helper
# ---------------------------------------------------------------------------

def _col_config_signals() -> dict:
    return {
        "fecha_evento":     st.column_config.DatetimeColumn("Fecha", format="DD/MM/YYYY"),
        "ticker":           st.column_config.TextColumn("Ticker"),
        "tipo_evento":      st.column_config.TextColumn("Tipo"),
        "precio_cierre":    st.column_config.NumberColumn("Precio", format="$%.2f"),
        "scoring":          st.column_config.NumberColumn("⭐", format="%d ★"),
        "rsi_14":           st.column_config.NumberColumn("RSI", format="%.0f"),
        "adx_14":           st.column_config.NumberColumn("ADX", format="%.0f"),
        "volumen_relativo": st.column_config.NumberColumn("Vol Rel", format="%.2f×"),
        "news_sentiment":   st.column_config.TextColumn("Sentimiento"),
        "retorno_7d":       st.column_config.NumberColumn("Ret. 7d %", format="%.2f%%"),
        "retorno_30d":      st.column_config.NumberColumn("Ret. 30d %", format="%.2f%%"),
        "analisis_gemini":  st.column_config.TextColumn("Análisis IA", width="large"),
    }


# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

_inject_css()
_render_header()

# ---------------------------------------------------------------------------
# Header extras: última actualización (expander) + forzar pipeline (botón)
# ---------------------------------------------------------------------------

try:
    _hdr_ultima = get_ultima_ejecucion()
except Exception:
    _hdr_ultima = None

try:
    _hdr_admin = st.secrets.get("ADMIN_MODE", "false").lower() == "true"
except Exception:
    _hdr_admin = os.getenv("ADMIN_MODE", "false").lower() == "true"

_hdr_left, _hdr_right = st.columns([5, 2])

with _hdr_left:
    with st.expander("🕐 Ver última actualización"):
        if _hdr_ultima:
            _ca = _hdr_ultima.get("created_at")
            if _ca and hasattr(_ca, "strftime"):
                if _ca.tzinfo is None:
                    _ca = _ca.replace(tzinfo=timezone.utc)
                _hdr_hrs = (datetime.now(timezone.utc) - _ca).total_seconds() / 3600
                _hdr_fecha = _ca.strftime("%Y-%m-%d %H:%M UTC")
            else:
                _hdr_hrs = 0.0
                _hdr_fecha = str(_ca)[:16] if _ca else "—"
            _hdr_n   = int(_hdr_ultima.get("senales_detectadas", 0))
            _hdr_dur = float(_hdr_ultima.get("duracion_seg", 0))
            _hdr_est = _hdr_ultima.get("estado", "—")
            _hdr_ok  = _hdr_est in ("ok", "ok_sin_senales")
            st.markdown(f"""
| | |
|---|---|
| **Fecha** | `{_hdr_fecha}` |
| **Estado** | {"✅" if _hdr_ok else "⚠️"} `{_hdr_est}` |
| **Señales detectadas** | `{_hdr_n}` |
| **Duración** | `{_hdr_dur:.0f} seg` |
| **Hace** | `{_hdr_hrs:.1f} horas` |
""")
        else:
            st.info("Sin ejecuciones registradas.")

with _hdr_right:
    if _hdr_admin:
        st.write("")
        if st.button("⚡ Forzar pipeline", type="primary", use_container_width=True):
            _hdr_out = st.empty()
            _hdr_logs: list[str] = []
            _hdr_proc = subprocess.Popen(
                [sys.executable, "main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path(__file__).parent),
            )
            with st.spinner("Ejecutando pipeline..."):
                for _hdr_line in _hdr_proc.stdout:  # type: ignore[union-attr]
                    _hdr_logs.append(_hdr_line.rstrip())
                    _hdr_out.code("\n".join(_hdr_logs[-20:]))
                _hdr_proc.wait()
            st.success("Pipeline completado.")

# ---------------------------------------------------------------------------
# Sidebar — Última ejecución + Panel de Control (admin)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("📊 Última ejecución")
    try:
        ultima = get_ultima_ejecucion()
    except Exception:
        ultima = None

    if ultima:
        created_at = ultima.get("created_at")
        if created_at is not None:
            if hasattr(created_at, "replace"):
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                horas_desde = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
                fecha_str = created_at.strftime("%Y-%m-%d %H:%M")
            else:
                horas_desde = 0.0
                fecha_str = str(created_at)[:16]
        else:
            horas_desde = 0.0
            fecha_str = "—"

        n_ej   = ultima.get("senales_detectadas", 0)
        dur_ej = float(ultima.get("duracion_seg", 0))
        est_ej = ultima.get("estado", "")

        if horas_desde > 25:
            st.error("⚠️ Pipeline no ejecutado en 25h")
        elif est_ej in ("ok", "ok_sin_senales"):
            st.success(f"✅ {fecha_str} — {n_ej} señales")
        else:
            st.warning(f"⚠️ {fecha_str} — estado: {est_ej}")

        st.caption(f"Duración: {dur_ej:.0f}s | Señales: {n_ej}")
    else:
        st.info("Sin ejecuciones registradas")

    # --- Admin Panel ---
    st.divider()
    if "admin_unlocked" not in st.session_state:
        st.session_state["admin_unlocked"] = False

    if not st.session_state["admin_unlocked"]:
        _pwd = st.text_input("🔐 Contraseña admin", type="password", key="admin_pwd_input")
        if _pwd:
            try:
                _correct = st.secrets.get("ADMIN_PASSWORD", "Casanova55-")
            except Exception:
                _correct = os.getenv("ADMIN_PASSWORD", "Casanova55-")
            if _pwd == _correct:
                st.session_state["admin_unlocked"] = True
                st.rerun()
            else:
                st.error("Contraseña incorrecta")

    if st.session_state["admin_unlocked"]:
        st.subheader("⚙️ Panel de Control")
        if st.button("🔒 Cerrar sesión admin", use_container_width=True):
            st.session_state["admin_unlocked"] = False
            st.rerun()

        if st.button("▶️ Ejecutar Pipeline Ahora", type="primary", use_container_width=True):
            output_box = st.empty()
            log_lines: list[str] = []
            proc = subprocess.Popen(
                [sys.executable, "main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path(__file__).parent),
            )
            with st.spinner("Analizando 75 tickers..."):
                for line in proc.stdout:  # type: ignore[union-attr]
                    log_lines.append(line.rstrip())
                    output_box.code("\n".join(log_lines[-20:]))
                proc.wait()

            try:
                ultima_post = get_ultima_ejecucion()
            except Exception:
                ultima_post = None

            if ultima_post:
                n_post   = ultima_post.get("senales_detectadas", 0)
                dur_post = float(ultima_post.get("duracion_seg", 0))
                est_post = ultima_post.get("estado", "")
                if est_post in ("ok", "ok_sin_senales"):
                    st.success(f"✅ Pipeline completado — {n_post} señales detectadas en {dur_post:.0f}s")
                else:
                    st.error(f"❌ Pipeline con errores — estado: {est_post}")
            else:
                st.warning("Pipeline finalizado — sin registro en log")

        if st.button("🔄 Actualizar Seguimiento", use_container_width=True):
            with st.spinner("Actualizando seguimiento..."):
                result = subprocess.run(
                    [sys.executable, "src/tracker.py"],
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).parent),
                )
            output = (result.stdout or "") + (result.stderr or "")
            updated = output.lower().count("updated")
            st.info(f"🔄 Seguimiento actualizado ({updated} señal(es) procesada(s))")

        st.divider()
        st.warning("⏱ Backtesting puede tardar 15-20 min para los 75 tickers.")
        if st.button("🔬 Ejecutar Backtesting Completo", use_container_width=True):
            _bt_output = st.empty()
            _bt_logs: list[str] = []
            _bt_proc = subprocess.Popen(
                [sys.executable, "-m", "src.backtester"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path(__file__).parent),
            )
            with st.spinner("Ejecutando backtest 2015-hoy..."):
                for _bt_line in _bt_proc.stdout:  # type: ignore[union-attr]
                    _bt_logs.append(_bt_line.rstrip())
                    _bt_output.code("\n".join(_bt_logs[-30:]))
                _bt_proc.wait()
            try:
                _bt_df   = get_backtest_df()
                _bt_st   = compute_stats(_bt_df)
                _bt_hr   = _bt_st["hit_rate_global"]
                _bt_rec  = (
                    "✅ OPERAR" if _bt_hr >= 58
                    else "⚠️ AJUSTAR FILTROS" if _bt_hr >= 50
                    else "🛑 NO OPERAR"
                )
                st.success(f"Backtest completado — Hit Rate: {_bt_hr:.1f}% · {_bt_rec}")
            except Exception:
                st.success("Backtest completado. Abre la pestaña 🔬 Backtesting para ver resultados.")

# ---------------------------------------------------------------------------
# Tab definitions
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📡 Radar Activo", "📊 Laboratorio", "📋 Bitácora", "🏆 Track Record", "📖 Manual", "🔬 Backtesting"]
)

# ===========================================================================
# TAB 1 — Radar Activo
# ===========================================================================

with tab1:
    st.header("📡 Radar Activo")

    ventana_label = st.radio(
        "Ventana temporal",
        ["Últimas 24h", "Últimas 48h", "Últimos 7 días"],
        horizontal=True,
    )
    horas_map = {"Últimas 24h": 24, "Últimas 48h": 48, "Últimos 7 días": 168}
    horas = horas_map[ventana_label]

    with st.spinner("Cargando señales..."):
        df_recientes = get_senales_recientes(horas=horas)

    if df_recientes.empty:
        st.markdown(f"""
        <div style="
            text-align:center; padding:52px 24px;
            background:#161b22; border:1px dashed #30363d;
            border-radius:14px; margin-top:16px;
        ">
            <div style="font-size:2.8em; margin-bottom:14px;">💤</div>
            <p style="font-size:1em; font-weight:600; color:#c9d1d9; margin:0 0 6px;">
                Sin señales en las últimas {horas}h
            </p>
            <p style="font-size:0.83em; color:#8b9cbc; margin:0;">
                El pipeline se ejecuta cada día a las 22:00 EST
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        gc = (df_recientes["tipo_evento"] == "golden_cross").sum()
        dc = (df_recientes["tipo_evento"] == "death_cross").sum()
        st.success(
            f"{len(df_recientes)} señal(es) detectada(s) — "
            f"🟢 {gc} Golden Cross · 🔴 {dc} Death Cross"
        )
        for _, row in df_recientes.iterrows():
            _render_signal_card(row.to_dict())

# ===========================================================================
# TAB 2 — Laboratorio de Gráficos
# ===========================================================================

with tab2:
    st.header("📊 Laboratorio de Gráficos")

    col_sel, col_per = st.columns([2, 1])
    with col_sel:
        ticker_sel = st.selectbox("Ticker", sorted(TICKERS))
    with col_per:
        periodo_sel = st.selectbox("Período", ["6M", "1Y", "2Y"])

    periodo_dias = {"6M": 180, "1Y": 365, "2Y": 730}
    dias = periodo_dias[periodo_sel]
    fecha_inicio = (date.today() - timedelta(days=dias)).strftime("%Y-%m-%d")

    with st.spinner(f"Descargando {ticker_sel}..."):
        df_chart = yf.download(
            ticker_sel, start=fecha_inicio, progress=False, auto_adjust=True
        )

    if df_chart.empty:
        st.error(f"No se pudieron obtener datos para {ticker_sel}")
    else:
        if isinstance(df_chart.columns, pd.MultiIndex):
            df_chart.columns = df_chart.columns.get_level_values(0)

        df_chart.index = pd.to_datetime(df_chart.index)

        df_chart["SMA50"]  = df_chart["Close"].rolling(50).mean()
        df_chart["SMA200"] = df_chart["Close"].rolling(200).mean()

        delta  = df_chart["Close"].diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / loss.replace(0, float("nan"))
        df_chart["RSI"] = 100 - (100 / (1 + rs))

        df_sig = get_historico(ticker=ticker_sel)

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.04,
        )

        fig.add_trace(
            go.Candlestick(
                x=df_chart.index,
                open=df_chart["Open"],
                high=df_chart["High"],
                low=df_chart["Low"],
                close=df_chart["Close"],
                name=ticker_sel,
                showlegend=False,
                increasing_line_color=COLORS["bullish"],
                decreasing_line_color=COLORS["bearish"],
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(x=df_chart.index, y=df_chart["SMA50"],
                       line=dict(color=COLORS["sma50"], width=1.5),
                       name="SMA 50"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df_chart.index, y=df_chart["SMA200"],
                       line=dict(color=COLORS["sma200"], width=1.5),
                       name="SMA 200"),
            row=1, col=1,
        )

        if not df_sig.empty:
            df_sig["fecha_evento"] = pd.to_datetime(df_sig["fecha_evento"])
            golden = df_sig[df_sig["tipo_evento"] == "golden_cross"]
            death  = df_sig[df_sig["tipo_evento"] == "death_cross"]

            if not golden.empty:
                fig.add_trace(
                    go.Scatter(
                        x=golden["fecha_evento"],
                        y=golden["precio_cierre"],
                        mode="markers",
                        marker=dict(symbol="triangle-up", color=COLORS["bullish"], size=13),
                        name="Golden Cross",
                    ),
                    row=1, col=1,
                )
            if not death.empty:
                fig.add_trace(
                    go.Scatter(
                        x=death["fecha_evento"],
                        y=death["precio_cierre"],
                        mode="markers",
                        marker=dict(symbol="triangle-down", color=COLORS["bearish"], size=13),
                        name="Death Cross",
                    ),
                    row=1, col=1,
                )

        fig.add_trace(
            go.Scatter(x=df_chart.index, y=df_chart["RSI"],
                       line=dict(color=COLORS["rsi"], width=1.2),
                       name="RSI-14"),
            row=2, col=1,
        )
        for level, color in [(70, COLORS["bearish"]), (30, COLORS["bullish"])]:
            fig.add_hline(y=level, line_dash="dot", line_color=color, row=2, col=1)

        fig.update_layout(
            title=dict(text=f"{ticker_sel} — {periodo_sel}", font=dict(color="#e6edf3")),
            height=620,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
        )
        fig.update_yaxes(title_text="RSI", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        if not df_sig.empty:
            st.subheader(f"Señales históricas — {ticker_sel}")
            cols_show = ["fecha_evento", "tipo_evento", "precio_cierre",
                         "scoring", "rsi_14", "adx_14", "retorno_7d", "retorno_30d"]
            cols_present = [c for c in cols_show if c in df_sig.columns]
            st.dataframe(
                df_sig[cols_present],
                column_config=_col_config_signals(),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Sin señales históricas registradas para este ticker.")

# ===========================================================================
# TAB 3 — Bitácora
# ===========================================================================

with tab3:
    st.header("📋 Bitácora")

    col1, col2, col3 = st.columns(3)
    with col1:
        tickers_filter = st.multiselect("Tickers", sorted(TICKERS))
    with col2:
        tipo_filter = st.selectbox("Tipo señal", ["Todos", "golden_cross", "death_cross"])
    with col3:
        scoring_min = st.slider("Scoring mínimo", 1, 3, 1)

    ticker_arg = tickers_filter[0] if len(tickers_filter) == 1 else None
    tipo_arg   = None if tipo_filter == "Todos" else tipo_filter

    with st.spinner("Cargando bitácora..."):
        df_hist = get_historico(ticker=ticker_arg, tipo=tipo_arg, scoring_min=scoring_min)

    if not df_hist.empty and tickers_filter and len(tickers_filter) > 1:
        df_hist = df_hist[df_hist["ticker"].isin(tickers_filter)]

    fecha_col = st.columns(2)
    with fecha_col[0]:
        fecha_desde = st.date_input("Desde", value=date.today() - timedelta(days=365))
    with fecha_col[1]:
        fecha_hasta = st.date_input("Hasta", value=date.today())

    if not df_hist.empty and "fecha_evento" in df_hist.columns:
        df_hist["fecha_evento"] = pd.to_datetime(df_hist["fecha_evento"])
        df_hist = df_hist[
            (df_hist["fecha_evento"].dt.date >= fecha_desde) &
            (df_hist["fecha_evento"].dt.date <= fecha_hasta)
        ]

    cols_bitacora = [
        "fecha_evento", "ticker", "tipo_evento", "precio_cierre",
        "scoring", "rsi_14", "adx_14", "news_sentiment", "analisis_gemini",
        "retorno_7d", "retorno_30d",
    ]
    cols_present = [c for c in cols_bitacora if c in df_hist.columns]

    if df_hist.empty:
        st.info("Sin señales para los filtros seleccionados.")
    else:
        st.dataframe(
            df_hist[cols_present],
            column_config=_col_config_signals(),
            hide_index=True,
            use_container_width=True,
        )

        csv_buf = io.StringIO()
        df_hist[cols_present].to_csv(csv_buf, index=False)
        st.download_button(
            label="⬇️ Exportar CSV",
            data=csv_buf.getvalue(),
            file_name=f"senales_{date.today()}.csv",
            mime="text/csv",
        )

# ===========================================================================
# TAB 4 — Track Record
# ===========================================================================

with tab4:
    st.header("🏆 Track Record")

    with st.spinner("Calculando métricas..."):
        stats = get_stats_rendimiento()

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total señales evaluadas", stats["total_senales"])
    col_m2.metric("Hit Rate", f"{stats['hit_rate']:.1f}%")
    col_m3.metric(
        "Retorno medio 30d",
        f"{stats['retorno_medio_30d']:+.2f}%",
        delta=f"{stats['retorno_medio_30d']:+.2f}%",
        delta_color="normal",
    )
    col_m4.metric("Mejor sector", stats["mejor_sector"])

    with st.spinner("Cargando datos de rendimiento..."):
        df_tr = get_historico(scoring_min=1)

    if df_tr.empty or "senal_ok" not in df_tr.columns:
        st.info("Sin datos suficientes todavía (las señales necesitan >30 días para evaluarse).")
    else:
        df_eval = df_tr[df_tr["senal_ok"].notna()].copy()

        if df_eval.empty:
            st.info("Aún no hay señales evaluadas (>30 días de antigüedad).")
        else:
            df_eval["fecha_evento"] = pd.to_datetime(df_eval["fecha_evento"])

            if "dist_sma_pct" in df_eval.columns and "retorno_30d" in df_eval.columns:
                st.subheader("Señal vs Retorno 30d")
                fig_scatter = go.Figure()
                for tipo, color in [
                    ("golden_cross", COLORS["bullish"]),
                    ("death_cross",  COLORS["tomato"]),
                ]:
                    sub = df_eval[df_eval["tipo_evento"] == tipo]
                    if not sub.empty:
                        fig_scatter.add_trace(go.Scatter(
                            x=sub["dist_sma_pct"],
                            y=sub["retorno_30d"],
                            mode="markers",
                            marker=dict(
                                color=color,
                                size=sub["adx_14"].clip(lower=8, upper=40) if "adx_14" in sub.columns else 10,
                                opacity=0.75,
                                line=dict(color="#0e1117", width=1),
                            ),
                            text=sub["ticker"],
                            name=tipo.replace("_", " ").title(),
                        ))
                fig_scatter.update_layout(
                    xaxis_title="Distancia SMA (%)",
                    yaxis_title="Retorno 30d (%)",
                    template="plotly_dark",
                    height=420,
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#161b22",
                )
                fig_scatter.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])
                st.plotly_chart(fig_scatter, use_container_width=True)

            if "sector" in df_eval.columns:
                st.subheader("Hit Rate por Sector")
                sector_stats = (
                    df_eval.groupby("sector")["senal_ok"]
                    .agg(hit_rate=lambda x: (x == True).mean() * 100, n="count")
                    .reset_index()
                    .sort_values("hit_rate", ascending=False)
                )
                sector_stats = sector_stats[
                    sector_stats["sector"].notna() & (sector_stats["sector"] != "N/A")
                ]
                if not sector_stats.empty:
                    fig_bar = go.Figure(go.Bar(
                        x=sector_stats["sector"],
                        y=sector_stats["hit_rate"],
                        marker_color=COLORS["steelblue"],
                        marker_line_color="#0e1117",
                        marker_line_width=1,
                        text=sector_stats["n"].apply(lambda n: f"n={n}"),
                        textposition="outside",
                        textfont=dict(color="#8b9cbc", size=11),
                    ))
                    fig_bar.update_layout(
                        yaxis_title="Hit Rate (%)",
                        template="plotly_dark",
                        height=370,
                        paper_bgcolor="#0e1117",
                        plot_bgcolor="#161b22",
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("Evolución del Hit Rate acumulado")
            df_sorted = df_eval.sort_values("fecha_evento")
            df_sorted["cum_ok"]    = (df_sorted["senal_ok"] == True).cumsum()
            df_sorted["cum_total"] = range(1, len(df_sorted) + 1)
            df_sorted["cum_hr"]    = df_sorted["cum_ok"] / df_sorted["cum_total"] * 100

            fig_line = go.Figure(go.Scatter(
                x=df_sorted["fecha_evento"],
                y=df_sorted["cum_hr"],
                mode="lines+markers",
                line=dict(color=COLORS["gold"], width=2.2),
                marker=dict(size=5, color=COLORS["gold"]),
                name="Hit Rate acumulado",
                fill="tozeroy",
                fillcolor="rgba(255,215,0,0.06)",
            ))
            fig_line.add_hline(
                y=50, line_dash="dot", line_color=COLORS["neutral"],
                annotation_text="50%", annotation_font_color=COLORS["neutral"],
            )
            fig_line.update_layout(
                yaxis_title="Hit Rate acumulado (%)",
                template="plotly_dark",
                height=370,
                paper_bgcolor="#0e1117",
                plot_bgcolor="#161b22",
            )
            st.plotly_chart(fig_line, use_container_width=True)

# ===========================================================================
# TAB 5 — Manual
# ===========================================================================

with tab5:
    st.header("📖 Manual de Usuario — DINERITO")
    st.caption("Guía completa del sistema de detección de señales bursátiles con IA")

    with st.expander("1. ¿Qué es DINERITO?", expanded=True):
        st.markdown("""
**DINERITO** es una plataforma de análisis bursátil automatizada que monitoriza **75 activos del mercado americano** (acciones, ETFs e índices) en busca de señales técnicas de alta probabilidad.

Cada día a las **22:00 EST** un pipeline automatizado:
1. Descarga precios históricos y calcula indicadores técnicos
2. Detecta cruces de medias móviles (Golden Cross / Death Cross)
3. Aplica un sistema de filtros para eliminar falsas señales
4. Enriquece cada señal con datos fundamentales y sentimiento de noticias
5. Genera un análisis de 3 puntos con Inteligencia Artificial (Gemini 2.5 Flash)
6. Guarda los resultados en la base de datos y envía alertas por Telegram

El objetivo es **encontrar señales de tendencia real** antes de que el movimiento ya esté descontado por el mercado.
        """)

    with st.expander("2. Las señales: Golden Cross y Death Cross"):
        col_gc, col_dc = st.columns(2)
        with col_gc:
            st.markdown("""
#### 🟢 Golden Cross
Se produce cuando la **SMA de 50 sesiones cruza al alza** la SMA de 200 sesiones.

**Significado:** el precio a corto plazo supera la tendencia de largo plazo → señal **alcista**.

**Condición exacta:**
- Ayer: SMA₅₀ ≤ SMA₂₀₀
- Hoy:  SMA₅₀ > SMA₂₀₀

**Lo que se busca:** inicio de un nuevo tramo alcista con momentum real.
            """)
        with col_dc:
            st.markdown("""
#### 🔴 Death Cross
Se produce cuando la **SMA de 50 sesiones cruza a la baja** la SMA de 200 sesiones.

**Significado:** el precio a corto plazo cae por debajo de la tendencia de largo plazo → señal **bajista**.

**Condición exacta:**
- Ayer: SMA₅₀ ≥ SMA₂₀₀
- Hoy:  SMA₅₀ < SMA₂₀₀

**Lo que se busca:** inicio de una corrección o tendencia bajista sostenida.
            """)

        st.info("""
**¿Por qué medias de 50 y 200?**
Son las referencias más seguidas por gestores institucionales y fondos. Cuando se cruzan, una parte del mercado reacciona automáticamente (stops, rebalanceos), lo que crea momentum adicional.
        """)

    with st.expander("3. Sistema de filtros anti-ruido (por qué no toda señal pasa)"):
        st.markdown("""
Un cruce de medias en solitario tiene una tasa de falsas señales muy alta. DINERITO aplica **3 filtros en cascada** para quedarse sólo con los cruces de alta convicción:

---
#### Filtro 1 — Fuerza de tendencia: ADX ≥ 25

| ADX | Interpretación |
|-----|---------------|
| < 20 | Mercado lateral (sin tendencia) |
| 20–25 | Tendencia débil |
| **≥ 25** | **Tendencia válida ✓** |
| ≥ 40 | Tendencia muy fuerte |

---
#### Filtro 2 — Confirmación de volumen: Vol ≥ 1.2× su media de 20 días

- **Volumen relativo < 1.2×** → señal descartada
- **Volumen relativo ≥ 1.2×** → señal con participación institucional potencial ✓

---
#### Filtro 3 — RSI no en zona extrema opuesta

- **Golden Cross:** RSI ≤ 75
- **Death Cross:** RSI ≥ 25

---
        """)
        st.success("Un cruce que pasa los 3 filtros tiene históricamente una tasa de acierto muy superior a un cruce sin filtrar.")

    with st.expander("4. Sistema de Scoring (● / ●● / ●●●)"):
        st.markdown("""
| Distancia |  Scoring | Interpretación |
|-----------|---------|---------------|
| dist < 0.5%  | ● (1/3) | Cruce reciente — señal débil |
| 0.5% ≤ dist < 1.5% | ●● (2/3) | Separación moderada — señal media |
| dist ≥ 1.5% | ●●● (3/3) | Medias bien separadas — señal fuerte |

> Consejo: filtra por scoring ≥ 2 en la Bitácora para concentrarte en las señales con más convicción.
        """)

    with st.expander("5. Análisis IA — cómo interpretar los 3 puntos"):
        st.markdown("""
Cada señal recibe un análisis de **Gemini 2.5 Flash** con exactamente **3 puntos**:

**1. TÉCNICO** — ¿el contexto de precio justifica el cruce?

**2. FUNDAMENTAL** — ¿hay un catalizador real? (usa Google Search en tiempo real)

**3. RIESGO** — principal amenaza a vigilar en las próximas 4 semanas.
        """)

    with st.expander("6. Guía de las 4 pestañas principales"):
        st.markdown("""
| Pestaña | Cuándo usarla |
|---------|--------------|
| 📡 **Radar Activo** | Primera consulta del día. Señales de las últimas 24h/48h/7d. |
| 📊 **Laboratorio** | Validar visualmente un cruce. Ver el gráfico de velas + RSI. |
| 📋 **Bitácora** | Análisis retrospectivo. Exportar datos. Filtrar por ticker/tipo/scoring. |
| 🏆 **Track Record** | Métricas de rendimiento histórico. Hit rate. Retorno medio. |
        """)

    with st.expander("7. Universo de activos monitorizados (75 tickers)"):
        st.markdown("""
- **Mega-cap (20):** AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSLA, JPM, V, MA, UNH, JNJ, PG, HD, BAC, WMT, XOM, CVX, LLY, AVGO
- **Mid-cap Growth (40):** COST, MRK, ABBV, CRM, ACN, AMD, NFLX, TMO, PEP, KO, ADBE, CSCO, MCD, ABT, WFC, TXN, NEE, LIN, PM, DHR, INTC, RTX, HON, UPS, IBM, CAT, SBUX, GS, BKNG, SPGI, AMGN, MDT, ISRG, NOW, PANW, UBER, SHOP, SQ, SNOW, ARM
- **ETFs sectoriales (10):** XLK, XLF, XLE, XLV, XLY, XLI, XLP, XLU, XLB, XLRE
- **Índices (5):** SPY, QQQ, DIA, IWM, VTI
        """)

    with st.expander("8. Glosario de indicadores técnicos"):
        st.markdown("""
| Indicador | Definición | Uso en DINERITO |
|-----------|-----------|-----------------|
| **SMA 50** | Media 50 cierres | Tendencia corto-medio plazo |
| **SMA 200** | Media 200 cierres | Tendencia largo plazo |
| **RSI-14** | Fuerza Relativa 14 sesiones (0–100) | Filtra zonas extremas |
| **ADX-14** | Fuerza de tendencia (0–100) | Confirma tendencia real |
| **Volumen relativo** | Vol hoy ÷ Vol MA20 | Convicción institucional |
| **dist_sma_pct** | (SMA₅₀−SMA₂₀₀)/SMA₂₀₀×100 | Base del scoring |
        """)

    with st.expander("9. Fuentes de datos y APIs externas"):
        st.markdown("""
| Fuente | Datos | Frecuencia |
|--------|-------|-----------|
| **Yahoo Finance** | Precios OHLCV | Diaria |
| **Alpha Vantage** | Sentimiento, fundamentales, Fed Rate, CPI | Por señal |
| **Gemini 2.5 Flash** | Análisis IA 3 puntos | Por señal |
| **Supabase** | Base de datos PostgreSQL | Tiempo real |
| **Telegram Bot** | Alertas push | Inmediata |
        """)

    with st.expander("10. Cómo leer una señal de principio a fin"):
        st.markdown("""
```
🟢 NVDA   ●●●
GOLDEN CROSS · 2026-05-08
💰 $890.40  |  RSI 62  |  ADX 32  |  🏢 Technology  |  📰 Bullish
```

1. **🟢 Golden Cross** → señal alcista
2. **●●● Scoring 3/3** → cruce bien consolidado (dist ≥ 1.5%)
3. **RSI 62** → momentum sin sobrecompra. Hay recorrido.
4. **ADX 32** → tendencia real confirmada
5. **📰 Bullish** → noticias favorables

**Siguiente paso:** Laboratorio → buscar NVDA → expandir análisis IA.
        """)

    with st.expander("11. Automatización: cómo funciona el pipeline diario"):
        st.markdown("""
Pipeline via **GitHub Actions** a las **03:00 UTC (22:00 EST)**:

```
1. Descarga OHLCV 75 tickers → calcula indicadores
2. Detección de cruces + 3 filtros (ADX / Volumen / RSI)
3. Contexto macro: Fed Rate + CPI (cacheado)
4. Por señal: noticias + fundamentales + análisis Gemini → Supabase + Telegram
5. Log de ejecución en Supabase
```
        """)

    with st.expander("12. Preguntas frecuentes (FAQ)"):
        st.markdown("""
**¿Esto es un consejo de inversión?**
No. DINERITO es análisis técnico cuantitativo. Las señales son puntos de partida, no recomendaciones.

---
**¿Por qué no hay señales hoy?**
El sistema no fuerza señales. Que no haya es un resultado válido y frecuente.

---
**¿Puedo añadir más tickers?**
Sí. Edita `TICKERS` en `src/config.py`. Requisito: ticker disponible en Yahoo Finance.

---
**¿El sistema funciona en mercados no americanos?**
No. Las SMA 50/200 y los fundamentales de Alpha Vantage están calibrados para el mercado americano.
        """)

    st.divider()
    st.caption("DINERITO · Pipeline automatizado de señales bursátiles · Actualizado diariamente a las 22:00 EST")

# ===========================================================================
# TAB 6 — Backtesting
# ===========================================================================

with tab6:
    st.header("🔬 Backtesting Histórico 2015–Hoy")

    # Check table exists
    _bt_has_data = False
    try:
        _bt_has_data = backtest_table_exists()
    except Exception:
        _bt_has_data = False

    if not _bt_has_data:
        st.info(
            "No hay datos de backtesting todavía. "
            "Ejecuta el backtest desde el Panel de Control del sidebar (admin) "
            "o con `python -m src.backtester`."
        )
        st.stop()

    # ── Sección 2: Filtros ────────────────────────────────────────────────
    with st.expander("⚙️ Filtros", expanded=True):
        _f_col1, _f_col2, _f_col3, _f_col4 = st.columns(4)

        _all_years   = get_all_years()
        _all_sectors = get_all_sectors()

        if len(_all_years) < 3:
            st.warning(
                f"**Base de datos incompleta** — solo hay {len(_all_years)} año(s) en `backtest_resultados`. "
                "Para ver métricas fiables ejecuta el backtest completo desde el sidebar (admin) "
                "o con `python -m src.backtester`.",
                icon="⚠️",
            )

        _yr_min = int(_all_years[0])  if _all_years else 2015
        _yr_max = int(_all_years[-1]) if _all_years else 2025
        if _yr_max <= _yr_min:
            _yr_max = _yr_min + 1  # slider requires min < max

        _year_range = _f_col1.slider(
            "Rango de años",
            min_value=_yr_min,
            max_value=_yr_max,
            value=(_yr_min, _yr_max),
            key="bt_year_range",
        )
        _sel_sectors = _f_col2.multiselect("Sectores", _all_sectors, key="bt_sectores")
        _tipo_filter = _f_col3.radio(
            "Tipo señal",
            ["Todas", "golden_cross", "death_cross"],
            horizontal=True,
            key="bt_tipo",
        )
        _scoring_min = _f_col4.slider("Scoring mínimo", 1, 3, 1, key="bt_scoring_min")

    # Build filter args
    _años_sel = list(range(_year_range[0], _year_range[1] + 1))
    _tipo_arg = None if _tipo_filter == "Todas" else _tipo_filter
    _sect_arg = _sel_sectors if _sel_sectors else None

    with st.spinner("Cargando datos..."):
        _bt_df = get_backtest_df(
            años=_años_sel,
            sectores=_sect_arg,
            tipo=_tipo_arg,
            scoring_min=_scoring_min,
        )

    _bt_stats = compute_stats(_bt_df)

    # ── Sección 1: Métricas principales ──────────────────────────────────
    st.subheader("Métricas globales")
    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("Hit Rate Global",      f"{_bt_stats['hit_rate_global']:.1f}%")
    _m2.metric("Retorno Medio 30d",    f"{_bt_stats['retorno_medio_30d']:+.2f}%")
    _m3.metric("Total Señales",        _bt_stats["total_senales"])
    _m4.metric("Señales / Año (media)", f"{_bt_stats['senales_por_año_media']:.1f}")

    _m5, _m6, _m7, _m8 = st.columns(4)
    _m5.metric("Hit Rate Golden Cross", f"{_bt_stats['hit_rate_golden']:.1f}%")
    _m6.metric("Hit Rate Death Cross",  f"{_bt_stats['hit_rate_death']:.1f}%")
    _m7.metric("Mejor Ticker",          _bt_stats["mejor_ticker"])
    _m8.metric("Peor Ticker",           _bt_stats["peor_ticker"])

    st.divider()

    # ── Sección 3: Gráficos ───────────────────────────────────────────────

    # A) Equity Curve
    _equity = _bt_stats["equity_curve"]
    if _equity:
        st.subheader("A) Equity Curve — $1.000 invertido por señal (compuesto)")
        _eq_df = pd.DataFrame(_equity)
        _fig_eq = go.Figure()
        _fig_eq.add_trace(go.Scatter(
            x=_eq_df["fecha"],
            y=_eq_df["capital"],
            mode="lines",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.07)",
            name="Capital acumulado",
        ))
        _fig_eq.add_hline(
            y=1000,
            line_dash="dot",
            line_color=COLORS["neutral"],
            annotation_text="$1.000 inicial",
            annotation_font_color=COLORS["neutral"],
        )
        _fig_eq.update_layout(
            yaxis_title="Capital ($)",
            template="plotly_dark",
            height=380,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161b22",
        )
        st.plotly_chart(_fig_eq, use_container_width=True)

    # B) Heatmap año × sector
    _hr_año  = _bt_stats["hit_rate_por_año"]
    _hr_sect = _bt_stats["hit_rate_por_sector"]
    if _hr_año and _hr_sect and not _bt_df.empty:
        st.subheader("B) Heatmap Hit Rate — Año × Sector")
        _hmap_ev = _bt_df[_bt_df["exito_30d"].notna()].copy()
        if not _hmap_ev.empty and "sector" in _hmap_ev.columns and "año" in _hmap_ev.columns:
            _hmap_ev["exito_num"] = _hmap_ev["exito_30d"].astype(float)
            _pivot = (
                _hmap_ev.groupby(["año", "sector"])["exito_num"]
                .mean()
                .mul(100)
                .round(1)
                .unstack(fill_value=None)
            )
            _pivot = _pivot[[c for c in _pivot.columns if c not in ("nan", "N/A", "Unknown", "")]]
            if not _pivot.empty:
                _fig_hm = go.Figure(go.Heatmap(
                    z=_pivot.values,
                    x=_pivot.columns.tolist(),
                    y=_pivot.index.tolist(),
                    colorscale=[[0, "#7f1d1d"], [0.5, "#78350f"], [1, "#14532d"]],
                    text=[[f"{v:.0f}%" if v is not None else "" for v in row] for row in _pivot.values],
                    texttemplate="%{text}",
                    colorbar=dict(title="Hit Rate %"),
                    zmin=0,
                    zmax=100,
                ))
                _fig_hm.update_layout(
                    xaxis_title="Sector",
                    yaxis_title="Año",
                    template="plotly_dark",
                    height=400,
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#161b22",
                )
                st.plotly_chart(_fig_hm, use_container_width=True)

    # C) Hit Rate por Scoring
    st.subheader("C) Hit Rate por Scoring")
    _hr_sc = _bt_stats["hit_rate_por_scoring"]
    if _hr_sc:
        _fig_sc = go.Figure(go.Bar(
            x=[f"Scoring {k}" for k in sorted(_hr_sc)],
            y=[_hr_sc[k] for k in sorted(_hr_sc)],
            marker_color=[COLORS["steelblue"], COLORS["primary"], COLORS["gold"]],
            text=[f"{_hr_sc[k]:.1f}%" for k in sorted(_hr_sc)],
            textposition="outside",
            textfont=dict(color="#c9d1d9"),
        ))
        _fig_sc.add_hline(y=50, line_dash="dot", line_color=COLORS["neutral"])
        _fig_sc.update_layout(
            yaxis_title="Hit Rate (%)",
            yaxis_range=[0, 100],
            template="plotly_dark",
            height=340,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161b22",
        )
        st.plotly_chart(_fig_sc, use_container_width=True)

    # D) Scatter dist_sma_pct vs retorno_30d
    if not _bt_df.empty and "dist_sma_pct" in _bt_df.columns and "retorno_30d" in _bt_df.columns:
        _sc_df = _bt_df.dropna(subset=["dist_sma_pct", "retorno_30d"])
        if not _sc_df.empty:
            st.subheader("D) Distancia SMA vs Retorno 30d")
            _fig_sc2 = go.Figure()
            for _tipo, _col in [("golden_cross", COLORS["bullish"]), ("death_cross", COLORS["tomato"])]:
                _sub = _sc_df[_sc_df["tipo_evento"] == _tipo]
                if not _sub.empty:
                    _fig_sc2.add_trace(go.Scatter(
                        x=_sub["dist_sma_pct"],
                        y=_sub["retorno_30d"],
                        mode="markers",
                        marker=dict(color=_col, size=6, opacity=0.55, line=dict(color="#0e1117", width=0.5)),
                        text=_sub["ticker"],
                        name=_tipo.replace("_", " ").title(),
                    ))
            _fig_sc2.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])
            _fig_sc2.update_layout(
                xaxis_title="Distancia SMA (%)",
                yaxis_title="Retorno 30d (%)",
                template="plotly_dark",
                height=400,
                paper_bgcolor="#0e1117",
                plot_bgcolor="#161b22",
            )
            st.plotly_chart(_fig_sc2, use_container_width=True)

    st.divider()

    # ── Sección 4: Tabla completa ─────────────────────────────────────────
    st.subheader("Tabla de señales históricas")

    _show_cols = [
        "fecha_senal", "ticker", "sector", "tipo_evento",
        "precio_entrada", "scoring", "adx_valor", "rsi_valor", "vol_ratio",
        "retorno_7d", "retorno_15d", "retorno_30d", "exito_30d", "año",
    ]
    _show_cols = [c for c in _show_cols if c in _bt_df.columns]

    if not _bt_df.empty:
        _disp_df = _bt_df[_show_cols].copy()
        _disp_df["resultado"] = _disp_df["exito_30d"].map(
            {True: "✅ Éxito", False: "❌ Fallo", None: "—"}
        ).fillna("—")

        _col_cfg_bt = {
            "fecha_senal":    st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
            "ticker":         st.column_config.TextColumn("Ticker"),
            "sector":         st.column_config.TextColumn("Sector"),
            "tipo_evento":    st.column_config.TextColumn("Tipo"),
            "precio_entrada": st.column_config.NumberColumn("Precio", format="$%.2f"),
            "scoring":        st.column_config.NumberColumn("★", format="%d"),
            "adx_valor":      st.column_config.NumberColumn("ADX", format="%.1f"),
            "rsi_valor":      st.column_config.NumberColumn("RSI", format="%.1f"),
            "vol_ratio":      st.column_config.NumberColumn("Vol ×", format="%.2f"),
            "retorno_7d":     st.column_config.NumberColumn("Ret 7d %", format="%.2f%%"),
            "retorno_15d":    st.column_config.NumberColumn("Ret 15d %", format="%.2f%%"),
            "retorno_30d":    st.column_config.NumberColumn("Ret 30d %", format="%.2f%%"),
            "exito_30d":      st.column_config.CheckboxColumn("Éxito 30d"),
            "año":            st.column_config.NumberColumn("Año", format="%d"),
            "resultado":      st.column_config.TextColumn("Resultado"),
        }

        st.dataframe(
            _disp_df,
            column_config=_col_cfg_bt,
            hide_index=True,
            use_container_width=True,
        )

        _csv_bt = io.StringIO()
        _disp_df.to_csv(_csv_bt, index=False)
        st.download_button(
            label="⬇️ Exportar CSV",
            data=_csv_bt.getvalue(),
            file_name=f"backtest_{date.today()}.csv",
            mime="text/csv",
        )
    else:
        st.info("Sin datos para los filtros seleccionados.")

    st.divider()

    # ── Sección 5: Conclusión automática ──────────────────────────────────
    st.subheader("Conclusión automática del sistema")

    _hr_g = _bt_stats["hit_rate_global"]
    _recomendacion = (
        "✅ OPERAR" if _hr_g >= 58
        else "⚠️ AJUSTAR FILTROS" if _hr_g >= 50
        else "🛑 NO OPERAR"
    )

    _mejor_sector = (
        max(_bt_stats["hit_rate_por_sector"], key=_bt_stats["hit_rate_por_sector"].get)
        if _bt_stats["hit_rate_por_sector"] else "N/A"
    )
    _peor_año = (
        min(_bt_stats["hit_rate_por_año"], key=_bt_stats["hit_rate_por_año"].get)
        if _bt_stats["hit_rate_por_año"] else "N/A"
    )
    _peor_año_hr = (
        _bt_stats["hit_rate_por_año"].get(_peor_año, 0.0)
        if isinstance(_peor_año, int) else 0.0
    )
    _mejor_sector_hr = _bt_stats["hit_rate_por_sector"].get(_mejor_sector, 0.0)
    _hr_sc3 = _bt_stats["hit_rate_por_scoring"].get(3, 0.0)

    _año_min = _year_range[0]
    _año_max = _year_range[1]

    st.info(f"""
**El sistema generó {_bt_stats['total_senales']} señales entre {_año_min}–{_año_max}.**

- Hit rate global: **{_hr_g:.1f}%** | Golden Cross: **{_bt_stats['hit_rate_golden']:.1f}%** | Death Cross: **{_bt_stats['hit_rate_death']:.1f}%**
- Las señales de scoring 3 tienen un hit rate de **{_hr_sc3:.1f}%**
- Mejor sector: **{_mejor_sector}** ({_mejor_sector_hr:.1f}% acierto)
- Peor año: **{_peor_año}** ({_peor_año_hr:.1f}% acierto)
- Retorno medio a 30 días: **{_bt_stats['retorno_medio_30d']:+.2f}%**

**Recomendación: {_recomendacion}**
""")
