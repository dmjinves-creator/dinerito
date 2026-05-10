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

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📡 Radar Activo", "📊 Laboratorio", "📋 Bitácora", "🏆 Track Record", "📖 Manual"]
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

# ===========================================================================
# TAB 5 — Manual
# ===========================================================================

with tab5:
    st.header("📖 Manual de Usuario — DINERITO")
    st.caption("Guía completa del sistema de detección de señales bursátiles con IA")

    # -----------------------------------------------------------------------
    # 1. Qué es DINERITO
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 2. Las señales: Golden Cross y Death Cross
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 3. Sistema de filtros anti-ruido
    # -----------------------------------------------------------------------
    with st.expander("3. Sistema de filtros anti-ruido (por qué no toda señal pasa)"):
        st.markdown("""
Un cruce de medias en solitario tiene una tasa de falsas señales muy alta. DINERITO aplica **3 filtros en cascada** para quedarse sólo con los cruces de alta convicción:
        """)

        st.markdown("""
---
#### Filtro 1 — Fuerza de tendencia: ADX ≥ 25
El **ADX (Average Directional Index)** mide la fuerza de una tendencia, independientemente de su dirección.

| ADX | Interpretación |
|-----|---------------|
| < 20 | Mercado lateral (sin tendencia) |
| 20–25 | Tendencia débil |
| **≥ 25** | **Tendencia válida ✓** |
| ≥ 40 | Tendencia muy fuerte |

**¿Por qué?** Un cruce en mercado lateral produce muchas señales falsas (whipsaws). El ADX ≥ 25 garantiza que hay una tendencia real detrás del cruce.

---
#### Filtro 2 — Confirmación de volumen: Vol ≥ 1.2× su media de 20 días
El precio se mueve donde va el dinero. Si el cruce ocurre con **volumen inferior a la media**, hay pocas probabilidades de continuación.

- **Volumen relativo < 1.2×** → señal descartada (sin convicción)
- **Volumen relativo ≥ 1.2×** → señal con participación institucional potencial ✓

---
#### Filtro 3 — RSI no en zona extrema opuesta
Evita entrar en el peor momento del ciclo:

- **Golden Cross:** RSI ≤ 75 (si el RSI ya es > 75 el activo está sobrecomprado; el cruce llega tarde)
- **Death Cross:** RSI ≥ 25 (si el RSI ya es < 25 el activo está sobrevendido; el cruce llega tarde)

**Resultado:** sólo pasan los cruces donde todavía hay recorrido en la dirección de la señal.

---
        """)

        st.success("Un cruce que pasa los 3 filtros tiene históricamente una tasa de acierto muy superior a un cruce sin filtrar.")

    # -----------------------------------------------------------------------
    # 4. Sistema de scoring (1-3 estrellas)
    # -----------------------------------------------------------------------
    with st.expander("4. Sistema de Scoring (⭐ / ⭐⭐ / ⭐⭐⭐🔥)"):
        st.markdown("""
Una vez que una señal pasa los 3 filtros, recibe una **puntuación de 1 a 3** basada en la **distancia porcentual entre las dos medias**:

```
dist_sma_pct = ((SMA₅₀ - SMA₂₀₀) / SMA₂₀₀) × 100
```

| Distancia |  Scoring | Interpretación |
|-----------|---------|---------------|
| dist < 0.5%  | ⭐ (1/3) | Cruce reciente, medias casi en contacto — señal débil |
| 0.5% ≤ dist < 1.5% | ⭐⭐ (2/3) | Separación moderada — señal media |
| dist ≥ 1.5% | ⭐⭐⭐🔥 (3/3) | Medias bien separadas — señal fuerte |

**Cuanto mayor es la separación, más confirmado está el cambio de tendencia.** Un scoring 3 indica que las medias llevan varios días divergiendo, lo que reduce la probabilidad de una reversión inmediata.

> Consejo: filtra por scoring ≥ 2 en la Bitácora para concentrarte en las señales con más convicción.
        """)

    # -----------------------------------------------------------------------
    # 5. Análisis IA: los 3 puntos
    # -----------------------------------------------------------------------
    with st.expander("5. Análisis IA — cómo interpretar los 3 puntos"):
        st.markdown("""
Cada señal que supera los filtros recibe un análisis generado por **Gemini 2.5 Flash** (Google). El modelo recibe todos los datos técnicos, fundamentales y macroeconómicos y devuelve exactamente **3 puntos**:

---
#### 1. TÉCNICO
¿El contexto de precio justifica o contradice el cruce?

El modelo evalúa si el cruce es coherente con la acción del precio reciente: ¿lleva semanas subiendo? ¿hay resistencias cercanas? ¿el RSI acompaña?

---
#### 2. FUNDAMENTAL
¿Hay un catalizador real detrás del movimiento?

El modelo usa búsqueda web en tiempo real (Google Search grounding) para identificar si hay noticias recientes, resultados de earnings, cambios en la dirección o eventos corporativos que expliquen el movimiento.

---
#### 3. RIESGO
Principal amenaza a vigilar en las próximas 4 semanas.

Puede ser un earnings date cercano, un sector en corrección, tipo de interés, riesgo regulatorio, o simplemente que la valoración ya descuenta mucho optimismo.

---

**Datos que recibe el modelo:**
- Precio de cierre, SMA 50/200, RSI-14, ADX-14, volumen relativo
- Scoring y distancia entre medias
- Sector, P/E ratio, EPS, capitalización bursátil
- Sentimiento de las últimas 5 noticias (Alpha Vantage)
- Fed Funds Rate y CPI del mes en curso
        """)

    # -----------------------------------------------------------------------
    # 6. Guía de las 4 pestañas
    # -----------------------------------------------------------------------
    with st.expander("6. Guía de las 4 pestañas principales"):
        st.markdown("""
### 📡 Radar Activo
Muestra las señales detectadas en el último periodo seleccionado (24h / 48h / 7 días).

**Cuándo usarlo:** primera cosa que consultar cada mañana. Si hay señales nuevas, aparecen aquí con su análisis IA completo.

**Tip:** si no hay señales, es buena señal — el sistema no fuerza entradas. El mercado no siempre da oportunidades de alta calidad.

---
### 📊 Laboratorio de Gráficos
Gráfico interactivo de velas japonesas para cualquier ticker del universo, con:
- **SMA 50** (azul) y **SMA 200** (naranja) superpuestas
- **RSI-14** en panel inferior con niveles 30/70
- **Marcadores de señales históricas** (triángulos verdes = Golden Cross, rojos = Death Cross)
- Tabla de señales históricas del ticker seleccionado

**Cuándo usarlo:** para validar visualmente una señal o estudiar el historial de cruces de un activo concreto.

---
### 📋 Bitácora
Histórico completo y filtrable de todas las señales detectadas. Filtros disponibles:
- **Tickers** (uno o varios)
- **Tipo de señal** (Golden Cross / Death Cross / Todos)
- **Scoring mínimo** (1, 2 o 3)
- **Rango de fechas** (desde / hasta)

**Exportación:** botón CSV para descargar el conjunto filtrado.

**Cuándo usarlo:** análisis retrospectivo, backtesting manual, o para compartir señales.

---
### 🏆 Track Record
Métricas de rendimiento histórico del sistema:

| Métrica | Descripción |
|---------|-------------|
| **Total señales evaluadas** | Señales con más de 30 días de antigüedad |
| **Hit Rate** | % de señales con retorno positivo a 30 días |
| **Retorno medio 30d** | Retorno medio de todas las señales evaluadas |
| **Mejor sector** | Sector con mayor hit rate histórico |

Incluye tres gráficos:
1. **Scatter dist_sma vs retorno 30d** — correlación entre la fuerza del cruce y el retorno
2. **Bar hit rate por sector** — qué sectores responden mejor a estas señales
3. **Evolución del hit rate acumulado** — cómo mejora (o empeora) la tasa de acierto con el tiempo

> Las señales necesitan **más de 30 días** desde su emisión para aparecer en el Track Record.
        """)

    # -----------------------------------------------------------------------
    # 7. Universo de activos
    # -----------------------------------------------------------------------
    with st.expander("7. Universo de activos monitorizados (75 tickers)"):
        st.markdown("""
El sistema analiza diariamente 75 activos divididos en 3 grupos:

#### Mega-cap (20 acciones)
Las 20 mayores empresas del S&P 500 por capitalización: AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSLA, JPM, V, MA, UNH, JNJ, PG, HD, BAC, WMT, XOM, CVX, LLY, AVGO.

#### Mid-cap Growth (40 acciones)
Empresas de crecimiento de mediana capitalización con alta liquidez: COST, MRK, ABBV, CRM, ACN, AMD, NFLX, TMO, PEP, KO, ADBE, CSCO, MCD, ABT, WFC, TXN, NEE, LIN, PM, DHR, INTC, RTX, HON, UPS, IBM, CAT, SBUX, GS, BKNG, SPGI, AMGN, MDT, ISRG, NOW, PANW, UBER, SHOP, SQ, SNOW, ARM.

#### ETFs sectoriales (10)
XLK (Tecnología), XLF (Financiero), XLE (Energía), XLV (Salud), XLY (Consumo discrecional), XLI (Industrial), XLP (Consumo básico), XLU (Utilities), XLB (Materiales), XLRE (Inmobiliario).

#### Índices de mercado (5)
SPY (S&P 500), QQQ (Nasdaq 100), DIA (Dow Jones), IWM (Russell 2000), VTI (Total Market).

---
**¿Por qué ETFs e índices?** Los cruces en ETFs sectoriales señalan rotaciones de capital entre sectores, que a menudo preceden movimientos en las acciones individuales del sector.
        """)

    # -----------------------------------------------------------------------
    # 8. Indicadores técnicos — glosario
    # -----------------------------------------------------------------------
    with st.expander("8. Glosario de indicadores técnicos"):
        st.markdown("""
| Indicador | Fórmula / Definición | Uso en DINERITO |
|-----------|---------------------|-----------------|
| **SMA 50** | Media aritmética de los últimos 50 cierres | Tendencia a corto-medio plazo |
| **SMA 200** | Media aritmética de los últimos 200 cierres | Tendencia de largo plazo |
| **RSI-14** | Índice de Fuerza Relativa (14 sesiones). 0-100. | Filtra entradas en zonas extremas |
| **ADX-14** | Average Directional Index (14 sesiones). 0-100. | Confirma que hay tendencia real |
| **VOL MA20** | Media de volumen de 20 días | Referencia para el filtro de volumen |
| **Volumen relativo** | Volumen hoy ÷ VOL MA20 | Mide la convicción institucional |
| **dist_sma_pct** | (SMA₅₀ - SMA₂₀₀) / SMA₂₀₀ × 100 | Base del sistema de scoring |

---

**RSI — niveles clave:**
- **> 70:** Sobrecomprado (cuidado con Golden Cross en esta zona)
- **30–70:** Zona neutra (señales más fiables)
- **< 30:** Sobrevendido (cuidado con Death Cross en esta zona)

**ADX — niveles clave:**
- **< 20:** Sin tendencia (señales ignoradas)
- **25–40:** Tendencia válida ✓
- **> 40:** Tendencia muy fuerte (momentum alto)
        """)

    # -----------------------------------------------------------------------
    # 9. Fuentes de datos y APIs
    # -----------------------------------------------------------------------
    with st.expander("9. Fuentes de datos y APIs externas"):
        st.markdown("""
| Fuente | Datos obtenidos | Frecuencia |
|--------|----------------|-----------|
| **Yahoo Finance** (yfinance) | Precios históricos OHLCV para todos los tickers | Diaria |
| **Alpha Vantage** | Sentimiento de noticias, fundamentales (P/E, EPS, sector, market cap), Fed Funds Rate, CPI | Por señal detectada |
| **Gemini 2.5 Flash** (Google AI) | Análisis narrativo de 3 puntos con Google Search grounding | Por señal detectada |
| **Supabase** | Base de datos PostgreSQL — almacenamiento de señales y ejecuciones | Lectura en tiempo real |
| **Telegram Bot** | Alertas push cuando se detecta una nueva señal | Inmediata tras detección |

---

**Limitaciones conocidas:**
- Alpha Vantage (plan gratuito): máximo 5 peticiones/minuto → el pipeline añade pausa de 12 segundos entre llamadas.
- Gemini: si el Google Search grounding no está disponible en la región, el análisis se genera sin búsqueda web en tiempo real (se indica en el análisis).
- Yahoo Finance: los datos de preapertura / cierre pueden tardar hasta 15–30 minutos en actualizarse tras el cierre del mercado.
        """)

    # -----------------------------------------------------------------------
    # 10. Cómo interpretar una señal completa
    # -----------------------------------------------------------------------
    with st.expander("10. Cómo leer una señal de principio a fin"):
        st.markdown("""
### Ejemplo paso a paso

Imagina que aparece esta señal en el Radar Activo:

```
🟢 NVDA   ⭐⭐⭐🔥
GOLDEN CROSS · 2026-05-08
💰 $890.40  |  RSI 62  |  ADX 32  |  🏢 Technology  |  📰 Bullish
```

**Cómo leerlo:**

1. **🟢 Golden Cross** → SMA 50 acaba de cruzar al alza la SMA 200: señal alcista.
2. **⭐⭐⭐🔥 Scoring 3/3** → las medias están separadas más de 1.5%, el cruce está bien consolidado.
3. **RSI 62** → momentum alcista sin estar sobrecomprado. Todavía hay recorrido.
4. **ADX 32** → hay una tendencia real detrás, no es un movimiento lateral.
5. **📰 Bullish** → el sentimiento de las últimas 5 noticias es favorable.

**Lo que habría que mirar después:**
- Ir al **Laboratorio** y buscar NVDA para ver el gráfico de velas con el cruce marcado.
- Leer el análisis IA completo (los 3 puntos: TÉCNICO, FUNDAMENTAL, RIESGO).
- Consultar la **Bitácora** para ver cómo se comportaron los cruces anteriores de NVDA.
- Ver en el **Track Record** si el sector Tecnología tiene buen hit rate histórico.

---

### Señales de alerta (cuándo ser más cautos)

- **Scoring 1 + RSI > 65** en Golden Cross → el cruce puede ser prematuro.
- **ADX entre 25 y 27** → justo en el límite; tendencia débil.
- **Volumen relativo entre 1.2× y 1.4×** → confirmación justa, no entusiasta.
- **Sentimiento Bearish** con Golden Cross → el mercado no cree en el cruce.
- **Earnings en menos de 2 semanas** → alta incertidumbre binaria.
        """)

    # -----------------------------------------------------------------------
    # 11. Automatización y pipeline
    # -----------------------------------------------------------------------
    with st.expander("11. Automatización: cómo funciona el pipeline diario"):
        st.markdown("""
El pipeline se ejecuta automáticamente cada día via **GitHub Actions** a las **03:00 UTC (22:00 EST)**, justo después del cierre del mercado americano.

### Pasos del pipeline (en orden)

```
1. Descarga OHLCV para los 75 tickers (Yahoo Finance)
   └── Calcula SMA 50, SMA 200, RSI-14, ADX-14, VOL MA20

2. Detección de cruces con sistema de filtros
   ├── Filtro ADX ≥ 25
   ├── Filtro Volumen ≥ 1.2× media 20d
   └── Filtro RSI no extremo

3. Si hay señales → obtiene contexto macro (Fed Rate, CPI)
   └── Solo una vez, cacheado para todo el pipeline

4. Por cada señal detectada:
   ├── Obtiene sentimiento de noticias (Alpha Vantage)
   ├── Obtiene fundamentales de la empresa (Alpha Vantage)
   ├── Genera análisis IA (Gemini 2.5 Flash + Google Search)
   ├── Guarda en Supabase
   └── Envía alerta por Telegram

5. Registra resumen de ejecución en Supabase
   └── (tickers analizados, señales, errores, duración)
```

### Logs
Cada ejecución genera un log en `logs/pipeline_YYYYMMDD.log` con el detalle completo de cada paso, incluyendo qué señales se descartaron y por qué filtro.

### Si no hay señales
El pipeline envía igualmente un **resumen diario por Telegram** indicando el número de tickers analizados y que no hubo señales ese día.
        """)

    # -----------------------------------------------------------------------
    # 12. Preguntas frecuentes
    # -----------------------------------------------------------------------
    with st.expander("12. Preguntas frecuentes (FAQ)"):
        st.markdown("""
**¿Esto es un consejo de inversión?**
No. DINERITO es una herramienta de análisis técnico cuantitativo. Las señales son puntos de partida para tu propio análisis, no recomendaciones de compra o venta.

---

**¿Por qué no hay señales hoy?**
El sistema no fuerza señales. En un día normal, la mayoría de los 75 activos no producen ningún cruce que pase los 3 filtros. Que no haya señales es un resultado válido y frecuente.

---

**¿Qué significa "senal_ok" en la base de datos?**
Es el campo de evaluación ex-post: se marca como `True` si el activo subió más de 0% en los 30 días siguientes a una Golden Cross (o bajó más de 0% en un Death Cross). Se calcula automáticamente cuando han pasado más de 30 días desde la señal.

---

**¿Por qué el análisis IA dice "sin acceso a búsqueda web"?**
Significa que Gemini generó el análisis sin Google Search grounding (puede ocurrir por limitaciones regionales del plan de API). El análisis sigue siendo válido pero no incluye noticias en tiempo real de ese día.

---

**¿Cuántos tickers se analizan realmente cada día?**
Los 75 configurados, salvo que se use la variable de entorno `TICKERS_TEST` para limitar el análisis a un subconjunto (útil para pruebas).

---

**¿Puedo añadir más tickers?**
Sí. Edita la lista `TICKERS` en `src/config.py`. El único requisito es que el ticker esté disponible en Yahoo Finance.

---

**¿El sistema funciona en mercados no americanos?**
No está diseñado para ello. Las medias SMA 50/200 están calibradas para el mercado americano (sesiones de lunes a viernes). Los datos fundamentales de Alpha Vantage sólo cubren acciones de EE.UU.
        """)

    st.divider()
    st.caption("DINERITO · Pipeline automatizado de señales bursátiles · Actualizado diariamente a las 22:00 EST")
