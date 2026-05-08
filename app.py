"""DINERITO — Dashboard Streamlit con 4 pestañas.

Pestaña 1: Radar Activo    — señales recientes con análisis IA
Pestaña 2: Laboratorio     — gráfico de velas + SMA + RSI por ticker
Pestaña 3: Bitácora        — tabla histórica filtrable + exportar CSV
Pestaña 4: Track Record    — métricas de rendimiento y hit-rate
"""

import io
from datetime import date, timedelta

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
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Plataforma Bursátil IA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

STAR_MAP = {1: "⭐", 2: "⭐⭐", 3: "⭐⭐⭐🔥"}


def _safe_str(val, default: str = "N/A") -> str:
    """Return str(val) unless val is None/NaN/empty — then return default.

    pandas reads NULL columns as float('nan'), which is truthy, so
    the plain `val or default` pattern fails for those values.
    """
    if val is None:
        return default
    if isinstance(val, float) and pd.isna(val):
        return default
    s = str(val).strip()
    return s if s and s != "nan" else default


def _scoring_badge(scoring: int) -> str:
    return STAR_MAP.get(int(scoring), "⭐")


def _card_color(tipo: str) -> str:
    return "#1a472a" if tipo == "golden_cross" else "#4a1c1c"


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
    emoji    = "🟢" if tipo == "golden_cross" else "🔴"
    label    = tipo.replace("_", " ").upper()
    color    = _card_color(tipo)

    st.markdown(
        f"""
        <div style="background:{color};border-radius:10px;padding:16px;margin-bottom:12px;">
          <h3 style="margin:0;color:white;">{emoji} {ticker} &nbsp; {_scoring_badge(scoring)}</h3>
          <p style="color:#ccc;margin:4px 0;">{label} · {fecha}</p>
          <p style="color:white;margin:4px 0;">
            💰 <b>${precio:.2f}</b> &nbsp;|&nbsp;
            RSI <b>{rsi:.0f}</b> &nbsp;|&nbsp;
            ADX <b>{adx:.0f}</b> &nbsp;|&nbsp;
            🏢 {sector} &nbsp;|&nbsp;
            📰 {sentiment}
          </p>
          <hr style="border-color:#555;margin:8px 0;">
          <p style="color:#ddd;font-size:0.85em;white-space:pre-wrap;">{analisis[:500]}{"..." if len(analisis) > 500 else ""}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab definitions
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["📡 Radar Activo", "📊 Laboratorio", "📋 Bitácora", "🏆 Track Record"]
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

    df_recientes = get_senales_recientes(horas=horas)

    if df_recientes.empty:
        st.info(
            f"Sin señales en las últimas {horas} horas. "
            "El pipeline se ejecuta diariamente a las 22:00 EST.",
            icon="💤",
        )
    else:
        st.success(f"{len(df_recientes)} señal(es) detectada(s)")
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
        # Flatten MultiIndex if present
        if isinstance(df_chart.columns, pd.MultiIndex):
            df_chart.columns = df_chart.columns.get_level_values(0)

        df_chart.index = pd.to_datetime(df_chart.index)

        # Compute SMAs
        df_chart["SMA50"]  = df_chart["Close"].rolling(50).mean()
        df_chart["SMA200"] = df_chart["Close"].rolling(200).mean()

        # Compute RSI-14
        delta  = df_chart["Close"].diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / loss.replace(0, float("nan"))
        df_chart["RSI"] = 100 - (100 / (1 + rs))

        # Historical signals for this ticker
        df_sig = get_historico(ticker=ticker_sel)

        # Build chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.04,
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df_chart.index,
                open=df_chart["Open"],
                high=df_chart["High"],
                low=df_chart["Low"],
                close=df_chart["Close"],
                name=ticker_sel,
                showlegend=False,
            ),
            row=1, col=1,
        )

        # SMA lines
        fig.add_trace(
            go.Scatter(x=df_chart.index, y=df_chart["SMA50"],
                       line=dict(color="royalblue", width=1.5),
                       name="SMA 50"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df_chart.index, y=df_chart["SMA200"],
                       line=dict(color="orange", width=1.5),
                       name="SMA 200"),
            row=1, col=1,
        )

        # Signal markers
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
                        marker=dict(symbol="triangle-up", color="lime", size=12),
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
                        marker=dict(symbol="triangle-down", color="red", size=12),
                        name="Death Cross",
                    ),
                    row=1, col=1,
                )

        # RSI subplot
        fig.add_trace(
            go.Scatter(x=df_chart.index, y=df_chart["RSI"],
                       line=dict(color="purple", width=1),
                       name="RSI-14"),
            row=2, col=1,
        )
        for level, color in [(70, "red"), (30, "green")]:
            fig.add_hline(y=level, line_dash="dot", line_color=color,
                          row=2, col=1)

        fig.update_layout(
            title=f"{ticker_sel} — {periodo_sel}",
            height=600,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_yaxes(title_text="RSI", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Historical signals table for this ticker
        if not df_sig.empty:
            st.subheader(f"Señales históricas — {ticker_sel}")
            cols_show = ["fecha_evento", "tipo_evento", "precio_cierre",
                         "scoring", "rsi_14", "adx_14", "retorno_7d", "retorno_30d"]
            cols_present = [c for c in cols_show if c in df_sig.columns]
            st.dataframe(df_sig[cols_present], use_container_width=True)
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
        st.dataframe(df_hist[cols_present], use_container_width=True)

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

    stats = get_stats_rendimiento()

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total señales evaluadas", stats["total_senales"])
    col_m2.metric("Hit Rate", f"{stats['hit_rate']:.1f}%")
    col_m3.metric("Retorno medio 30d", f"{stats['retorno_medio_30d']:+.2f}%")
    col_m4.metric("Mejor sector", stats["mejor_sector"])

    df_tr = get_historico(scoring_min=1)

    if df_tr.empty or "senal_ok" not in df_tr.columns:
        st.info("Sin datos suficientes todavía (las señales necesitan >30 días para evaluarse).")
        st.stop()

    df_eval = df_tr[df_tr["senal_ok"].notna()].copy()

    if df_eval.empty:
        st.info("Aún no hay señales evaluadas (>30 días de antigüedad).")
        st.stop()

    df_eval["fecha_evento"] = pd.to_datetime(df_eval["fecha_evento"])

    # --- Scatter: dist_sma_pct vs retorno_30d ---
    if "dist_sma_pct" in df_eval.columns and "retorno_30d" in df_eval.columns:
        st.subheader("Señal vs Retorno 30d")
        fig_scatter = go.Figure()
        for tipo, color in [("golden_cross", "lime"), ("death_cross", "tomato")]:
            sub = df_eval[df_eval["tipo_evento"] == tipo]
            if not sub.empty:
                fig_scatter.add_trace(go.Scatter(
                    x=sub["dist_sma_pct"],
                    y=sub["retorno_30d"],
                    mode="markers",
                    marker=dict(
                        color=color,
                        size=sub["adx_14"].clip(lower=8, upper=40) if "adx_14" in sub.columns else 10,
                        opacity=0.7,
                    ),
                    text=sub["ticker"],
                    name=tipo.replace("_", " ").title(),
                ))
        fig_scatter.update_layout(
            xaxis_title="Distancia SMA (%)",
            yaxis_title="Retorno 30d (%)",
            template="plotly_dark",
            height=400,
        )
        fig_scatter.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Bar: hit rate por sector ---
    if "sector" in df_eval.columns:
        st.subheader("Hit Rate por Sector")
        sector_stats = (
            df_eval.groupby("sector")["senal_ok"]
            .agg(hit_rate=lambda x: (x == True).mean() * 100, n="count")
            .reset_index()
            .sort_values("hit_rate", ascending=False)
        )
        sector_stats = sector_stats[sector_stats["sector"].notna() & (sector_stats["sector"] != "N/A")]
        if not sector_stats.empty:
            fig_bar = go.Figure(go.Bar(
                x=sector_stats["sector"],
                y=sector_stats["hit_rate"],
                marker_color="steelblue",
                text=sector_stats["n"].apply(lambda n: f"n={n}"),
                textposition="outside",
            ))
            fig_bar.update_layout(
                yaxis_title="Hit Rate (%)",
                template="plotly_dark",
                height=350,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- Line: cumulative hit rate over time ---
    st.subheader("Evolución del Hit Rate acumulado")
    df_sorted = df_eval.sort_values("fecha_evento")
    df_sorted["cum_ok"]    = (df_sorted["senal_ok"] == True).cumsum()
    df_sorted["cum_total"] = range(1, len(df_sorted) + 1)
    df_sorted["cum_hr"]    = df_sorted["cum_ok"] / df_sorted["cum_total"] * 100

    fig_line = go.Figure(go.Scatter(
        x=df_sorted["fecha_evento"],
        y=df_sorted["cum_hr"],
        mode="lines+markers",
        line=dict(color="gold", width=2),
        name="Hit Rate acumulado",
    ))
    fig_line.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="50%")
    fig_line.update_layout(
        yaxis_title="Hit Rate acumulado (%)",
        template="plotly_dark",
        height=350,
    )
    st.plotly_chart(fig_line, use_container_width=True)
