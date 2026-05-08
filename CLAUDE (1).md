# PLATAFORMA CUANTITATIVA DE ANÁLISIS BURSÁTIL CON IA
## Documento Maestro de Contexto para Claude Code
**Versión:** 2.0 — Plan Completo Corregido
**Uso:** Pon este archivo en la raíz del proyecto. Claude lo leerá antes de generar cualquier código.

---

## 1. VISIÓN GENERAL DEL PROYECTO

Sistema privado de análisis cuantitativo bursátil que:
- Analiza 75 tickers del S&P 500 / Nasdaq cada día a las 22:00 EST (tras el cierre del mercado)
- Detecta cambios de tendencia mediante Golden Cross / Death Cross (SMA 50 vs SMA 200)
- Aplica un sistema de filtros para eliminar señales falsas (ADX, RSI, Volumen)
- Enriquece cada señal con datos reales via Alpha Vantage MCP (noticias, fundamentales, macro)
- Analiza cada señal con Gemini 1.5 Pro + grounding de búsqueda web
- Almacena todo en Supabase PostgreSQL (accesible desde CronJob y frontend)
- Envía alertas push por Telegram Bot inmediatamente al detectar señales
- Muestra todo en un dashboard Streamlit con 4 pestañas

**El sistema NO es un sistema de trading automático. Es una herramienta de detección y análisis para toma de decisiones manual.**

---

## 2. STACK TECNOLÓGICO COMPLETO

```
Lenguaje:         Python 3.11+
Datos de precio:  yfinance (descarga masiva 75 tickers, sin API key)
Datos enriquec.:  Alpha Vantage API (señales detectadas únicamente, 25 req/día free)
MCP en VS Code:   Alpha Vantage MCP Server (desarrollo y testing con datos reales)
Indicadores:      pandas_ta (SMA, RSI, ADX calculados localmente)
IA:               Google Gemini 1.5 Pro con Google Search Retrieval (grounding)
Base de datos:    Supabase PostgreSQL (cloud, free tier 500MB)
Frontend:         Streamlit + Plotly (gráficos interactivos)
Alertas:          Telegram Bot API
Hosting CronJob:  PythonAnywhere (free tier, 1 tarea programada)
Hosting Frontend: Streamlit Community Cloud (free tier)
Control versión:  GitHub (repo privado)
Entorno local:    VS Code + extensión Claude + Python venv
```

---

## 3. ESTRUCTURA DE ARCHIVOS

```
plataforma-bursatil/
├── CLAUDE.md                   ← este archivo (contexto maestro)
├── .env                        ← API keys (NUNCA subir a GitHub)
├── .env.example                ← plantilla sin valores
├── .gitignore                  ← excluye .env, venv, __pycache__, logs
├── requirements.txt            ← dependencias Python
├── main.py                     ← pipeline principal (ejecuta el CronJob)
├── app.py                      ← frontend Streamlit
├── src/
│   ├── __init__.py
│   ├── config.py               ← variables de entorno + constantes + lista tickers
│   ├── data_fetcher.py         ← descarga yfinance + validación + procesamiento paralelo
│   ├── signal_detector.py      ← lógica Golden/Death Cross + filtros ADX/RSI/volumen
│   ├── enricher.py             ← Alpha Vantage: noticias, overview, macro
│   ├── ai_engine.py            ← Gemini 1.5 Pro + grounding + prompt estructurado
│   ├── database.py             ← Supabase: insertar señales, consultar, actualizar seguimiento
│   ├── notifier.py             ← Telegram Bot: enviar alertas formateadas
│   └── tracker.py              ← script semanal: rellena precio_30d y señal_ok
├── tests/
│   ├── test_signal_detector.py
│   ├── test_data_fetcher.py
│   └── test_database.py
├── logs/                       ← archivos de log (excluidos de git)
└── .streamlit/
    └── secrets.toml            ← secrets Streamlit (excluido de git)
```

---

## 4. VARIABLES DE ENTORNO (.env)

Crear archivo `.env` en la raíz con exactamente estas variables:

```env
# Supabase PostgreSQL
SUPABASE_URL=https://xxxxxxxxxxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_DB_PASSWORD=tu_password_de_supabase

# Google Gemini
GEMINI_API_KEY=AIzaSy...

# Alpha Vantage (datos fundamentales y noticias)
ALPHA_VANTAGE_KEY=XXXX1234YYYY5678

# Telegram Bot
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNO...
TELEGRAM_CHAT_ID=987654321
```

Crear también `.env.example` con los mismos campos pero sin valores reales (este SÍ sube a GitHub).

---

## 5. CONFIGURACIÓN MCP — ALPHA VANTAGE EN VS CODE

Crear archivo `.vscode/mcp.json` en la raíz del proyecto:

```json
{
  "servers": {
    "alphavantage": {
      "type": "http",
      "url": "https://mcp.alphavantage.co/mcp?apikey=TU_ALPHA_VANTAGE_KEY"
    }
  }
}
```

Reemplazar TU_ALPHA_VANTAGE_KEY con la key real del .env.
Este archivo permite a Claude en VS Code consultar datos financieros reales durante el desarrollo.

---

## 6. BASE DE DATOS — SCHEMA SUPABASE

Ejecutar este SQL en el SQL Editor de Supabase al crear el proyecto:

```sql
-- Tabla principal de señales detectadas
CREATE TABLE IF NOT EXISTS senales (
    id               SERIAL PRIMARY KEY,
    created_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    fecha_evento     DATE NOT NULL,
    ticker           VARCHAR(10) NOT NULL,
    tipo_evento      VARCHAR(20) NOT NULL,     -- golden_cross / death_cross
    precio_cierre    DECIMAL(10,2),
    sma_50           DECIMAL(10,2),
    sma_200          DECIMAL(10,2),
    rsi_14           DECIMAL(5,2),
    adx_14           DECIMAL(5,2),
    volumen_relativo DECIMAL(5,2),
    dist_sma_pct     DECIMAL(5,2),
    scoring          INTEGER DEFAULT 0,         -- 1=débil, 2=medio, 3=fuerte
    -- Datos fundamentales (Alpha Vantage)
    sector           VARCHAR(100),
    pe_ratio         VARCHAR(20),
    eps              VARCHAR(20),
    market_cap       VARCHAR(30),
    earnings_date    VARCHAR(30),
    -- Sentimiento de noticias (Alpha Vantage)
    news_sentiment   VARCHAR(20),               -- Bullish/Bearish/Neutral
    news_score       DECIMAL(5,4),
    -- Análisis IA
    analisis_gemini  TEXT,
    -- Contexto macro en el momento de la señal
    fed_funds_rate   DECIMAL(5,2),
    cpi_inflacion    DECIMAL(5,2),
    -- Seguimiento posterior (rellena tracker.py semanalmente)
    precio_7d        DECIMAL(10,2),
    precio_30d       DECIMAL(10,2),
    retorno_7d       DECIMAL(6,2),
    retorno_30d      DECIMAL(6,2),
    senal_ok         BOOLEAN,
    UNIQUE(fecha_evento, ticker, tipo_evento)
);

-- Log de ejecuciones del CronJob
CREATE TABLE IF NOT EXISTS ejecuciones_log (
    id                  SERIAL PRIMARY KEY,
    fecha               TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tickers_analizados  INTEGER,
    senales_detectadas  INTEGER,
    errores             INTEGER,
    duracion_seg        DECIMAL(6,2),
    estado              VARCHAR(20)              -- ok / error / parcial
);

-- Índices para consultas rápidas en Streamlit
CREATE INDEX idx_senales_fecha    ON senales(fecha_evento DESC);
CREATE INDEX idx_senales_ticker   ON senales(ticker);
CREATE INDEX idx_senales_tipo     ON senales(tipo_evento);
CREATE INDEX idx_senales_scoring  ON senales(scoring DESC);
```

---

## 7. MÓDULO: src/config.py

**Propósito:** Carga y valida todas las variables de entorno. Define constantes del sistema y la lista de 75 tickers.

**Debe incluir:**
- Función `_req(key)` que lanza ValueError si la variable no está definida
- Todas las variables de entorno cargadas con python-dotenv
- Constantes del sistema: ADX_MIN=25, RSI_MAX_GOLDEN=75, RSI_MIN_DEATH=25, VOL_MIN_RATIO=1.2
- Lista TICKERS con exactamente estos 75 activos:
  ```
  Mega-cap (20): AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSLA, JPM, V, MA,
                 UNH, JNJ, PG, HD, BAC, WMT, XOM, CVX, LLY, AVGO
  Mid-cap growth (30): COST, MRK, ABBV, CRM, ACN, AMD, NFLX, TMO, PEP, KO,
                       ADBE, CSCO, MCD, ABT, WFC, TXN, NEE, LIN, PM, DHR,
                       INTC, RTX, HON, UPS, IBM, CAT, SBUX, GS, BKNG, SPGI
  ETFs sectoriales (10): XLK, XLF, XLE, XLV, XLY, XLI, XLP, XLU, XLB, XLRE
  Índices (5): SPY, QQQ, DIA, IWM, VTI
  ```
- Función `calcular_scoring(dist_pct)` que retorna 1/2/3 según umbrales [0.5, 1.5]

---

## 8. MÓDULO: src/data_fetcher.py

**Propósito:** Descarga datos de precio de yfinance, valida integridad, calcula indicadores técnicos con pandas_ta, y procesa los 75 tickers en paralelo.

**Funciones requeridas:**

`descargar_y_validar(ticker: str) -> pd.DataFrame | None`
- Descarga 2 años de datos diarios con yfinance
- Valida: mínimo 220 filas, sin NaN en Close/High/Low/Volume, sin gaps > 7 días
- Si hay NaN pequeños: rellena con forward fill + backward fill
- Si falla cualquier validación: retorna None y registra en log
- Manejo de excepciones: captura cualquier error de yfinance y retorna None

`calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame`
- Añade columnas: SMA_50, SMA_200, RSI_14, ADX_14, VOL_MA20
- Usa pandas_ta para todos los cálculos
- Elimina filas con NaN resultantes del cálculo (primeras 200 velas)
- Retorna el DataFrame con las columnas añadidas

`procesar_todos_tickers(tickers: list) -> list[dict]`
- Usa ThreadPoolExecutor con max_workers=5
- Llama descargar_y_validar + calcular_indicadores para cada ticker
- Añade delay aleatorio entre 0.3-0.8s entre peticiones para no saturar Yahoo
- Retorna lista de dicts con {ticker, df} para los que tuvieron éxito
- Registra en log cuántos fallaron y por qué

---

## 9. MÓDULO: src/signal_detector.py

**Propósito:** Detecta Golden Cross y Death Cross con sistema de filtros en cascada para eliminar señales falsas.

**Lógica completa del detector:**

```
CONDICIÓN GOLDEN CROSS:
  ayer.SMA_50 <= ayer.SMA_200  AND  hoy.SMA_50 > hoy.SMA_200

CONDICIÓN DEATH CROSS:
  ayer.SMA_50 >= ayer.SMA_200  AND  hoy.SMA_50 < hoy.SMA_200

FILTRO 1 — Solo en tendencia real (anti-whipsaw):
  Si ADX_14 < 25 → descartar (mercado lateral, señal no fiable)

FILTRO 2 — Volumen confirma el movimiento:
  Si Volume_hoy < VOL_MA20 * 1.2 → descartar (sin convicción)

FILTRO 3 — RSI no en zona extrema opuesta:
  Golden Cross: Si RSI_14 > 75 → descartar (sobrecomprado, trampa alcista)
  Death Cross:  Si RSI_14 < 25 → descartar (sobrevendido, trampa bajista)

CÁLCULO DE SCORING (fuerza del cruce):
  dist_pct = ((SMA_50 - SMA_200) / SMA_200) * 100
  Si |dist_pct| < 0.5% → scoring = 1 (débil)
  Si |dist_pct| < 1.5% → scoring = 2 (medio)
  Si |dist_pct| >= 1.5% → scoring = 3 (fuerte)
```

`detectar_cruce(df: pd.DataFrame, ticker: str) -> dict | None`
- Aplica la lógica completa descrita arriba
- Si supera todos los filtros, retorna dict con todos los campos para la BD
- Si no hay cruce o falla algún filtro, retorna None

`detectar_en_todos(datos_tickers: list) -> list[dict]`
- Itera sobre la lista de DataFrames procesados
- Retorna lista de señales detectadas (puede ser vacía)

---

## 10. MÓDULO: src/enricher.py

**Propósito:** Enriquece cada señal detectada con datos de Alpha Vantage.
Solo se llama cuando hay señales — máximo 3 peticiones por señal.

**Funciones requeridas:**

`obtener_noticias_sentimiento(ticker: str) -> dict`
- Endpoint: NEWS_SENTIMENT, limit=5, sort=LATEST
- Retorna: sentiment_label (Bullish/Bearish/Neutral), overall_sentiment_score
- Si falla: retorna {'news_sentiment': 'N/A', 'news_score': 0.0}

`obtener_overview_empresa(ticker: str) -> dict`
- Endpoint: COMPANY_OVERVIEW
- Retorna: sector, PERatio, EPS, MarketCapitalization, NextEarningsDate
- Si falla: retorna dict con todos los campos en 'N/A'

`obtener_contexto_macro() -> dict`
- Endpoint: FEDERAL_FUNDS_RATE (interval=monthly) y CPI (interval=monthly)
- Solo llama una vez por ejecución del pipeline (no por ticker)
- Retorna: {'fed_rate': float, 'cpi': float}
- Cachear resultado en variable de módulo para evitar llamadas repetidas

`enriquecer_senal(senal: dict) -> dict`
- Llama a las tres funciones anteriores
- Merge de todos los datos en el dict de la señal
- Retorna señal enriquecida lista para insertar en BD

**IMPORTANTE:** Añadir delay de 12s entre llamadas a Alpha Vantage
para respetar el límite de 5 peticiones/minuto del tier gratuito.

---

## 11. MÓDULO: src/ai_engine.py

**Propósito:** Genera análisis en lenguaje natural de cada señal usando Gemini 1.5 Pro con Google Search Retrieval activado.

**Prompt del sistema (usar exactamente este):**
```
Eres un analista cuantitativo senior especializado en renta variable americana.
Recibes datos técnicos y fundamentales de una señal de mercado y debes proporcionar
un análisis conciso, factual y accionable. Basa tu respuesta en datos verificables.
Evita especulación sin respaldo. Usa terminología financiera precisa.
```

**Prompt de usuario (construir dinámicamente):**
```
SEÑAL DETECTADA:
- Ticker: {ticker} ({sector})
- Tipo: {tipo_evento.replace('_',' ').upper()}
- Fecha: {fecha_evento}
- Precio cierre: ${precio_cierre:.2f}

DATOS TÉCNICOS:
- SMA 50: ${sma_50:.2f} | SMA 200: ${sma_200:.2f}
- Distancia entre medias: {dist_sma_pct:+.2f}%
- RSI-14: {rsi_14:.1f} | ADX-14: {adx_14:.1f} (fuerza tendencia)
- Volumen relativo: {volumen_relativo:.1f}x sobre media 20 días
- Scoring señal: {scoring}/3

DATOS FUNDAMENTALES:
- P/E Ratio: {pe_ratio} | EPS: {eps}
- Cap. bursátil: {market_cap}
- Próximo earnings: {earnings_date}

SENTIMIENTO NOTICIAS (últimos 7 días):
- Sentimiento: {news_sentiment} (score: {news_score:.3f})

CONTEXTO MACROECONÓMICO:
- Fed Funds Rate: {fed_funds_rate:.2f}%
- CPI Inflación: {cpi_inflacion:.2f}%

Busca noticias recientes sobre {ticker} y analiza en exactamente 3 puntos
(máximo 30 palabras cada uno):
1. TÉCNICO: ¿el contexto de precio justifica o contradice el cruce?
2. FUNDAMENTAL: ¿hay catalizador real o es movimiento de mercado general?
3. RIESGO: principal amenaza a vigilar en las próximas 4 semanas.
```

**Funciones requeridas:**

`analizar_senal(senal: dict) -> str`
- Configura Gemini con google_search_retrieval si está disponible
- Fallback: si google_search_retrieval no está en la región, usa el modelo sin grounding
- Maneja RateLimitError con retry exponencial (3 intentos, espera 60s entre intentos)
- Retorna el texto del análisis o mensaje de error descriptivo

**NOTA:** La key de Gemini la carga desde config.py. Inicializar genai una sola vez al importar el módulo.

---

## 12. MÓDULO: src/database.py

**Propósito:** Todas las operaciones con Supabase PostgreSQL.

**Conexión:** Usar psycopg2 con connection string de Supabase.
La URL de conexión se construye como:
`postgresql://postgres:{SUPABASE_DB_PASSWORD}@db.{SUPABASE_PROJECT_REF}.supabase.co:5432/postgres`

El SUPABASE_PROJECT_REF se extrae de SUPABASE_URL (la parte entre https:// y .supabase.co).

**Funciones requeridas:**

`insertar_senal(senal: dict) -> bool`
- INSERT con ON CONFLICT DO NOTHING (para evitar duplicados por la constraint UNIQUE)
- Retorna True si insertó, False si ya existía o hubo error

`get_senales_recientes(horas: int = 48) -> pd.DataFrame`
- SELECT de señales de las últimas N horas, ordenadas por fecha DESC
- Retorna DataFrame para Streamlit

`get_historico(ticker: str = None, tipo: str = None, scoring_min: int = 1) -> pd.DataFrame`
- SELECT con filtros opcionales
- Retorna DataFrame para la pestaña Bitácora

`get_stats_rendimiento() -> dict`
- SELECT agregado: hit_rate, retorno_medio_30d, total_senales, mejor_sector
- Solo considera señales con senal_ok IS NOT NULL
- Retorna dict con métricas para el Track Record

`registrar_ejecucion(stats: dict) -> None`
- INSERT en ejecuciones_log con los resultados del pipeline

`get_senales_para_seguimiento() -> pd.DataFrame`
- SELECT señales con precio_30d IS NULL y fecha_evento < NOW() - 30 days

`actualizar_seguimiento(id: int, precio_7d: float, precio_30d: float) -> None`
- UPDATE de precio_7d, precio_30d, retorno_7d, retorno_30d y senal_ok
- senal_ok = True si retorno_30d > 0 para golden_cross, o < 0 para death_cross

---

## 13. MÓDULO: src/notifier.py

**Propósito:** Envía alertas por Telegram cuando se detecta una señal.

**Formato del mensaje Telegram (usar Markdown):**
```
{emoji} *{TIPO CRUCE}* — ${ticker}

💰 *Precio:* ${precio:.2f}
⭐ *Scoring:* {scoring}/3
📊 *RSI:* {rsi:.0f} | *ADX:* {adx:.0f}
🏢 *Sector:* {sector}
📰 *Noticias:* {news_sentiment}

🤖 *Análisis IA:*
{analisis_gemini[:300]}...

🔗 [Ver gráfico en dashboard](URL_STREAMLIT?ticker={ticker})
```
- Golden Cross → emoji 🟢
- Death Cross → emoji 🔴
- Scoring 3 → añadir 🔥 al título

**Funciones:**
`enviar_alerta(senal: dict) -> bool`
`enviar_resumen_diario(n_senales: int, n_tickers: int, duracion: float) -> None`

---

## 14. MÓDULO: main.py

**Propósito:** Pipeline principal. Lo ejecuta el CronJob de PythonAnywhere cada día.

**Flujo de ejecución:**
```
1. Iniciar timer y logging
2. config.py → cargar configuración
3. data_fetcher.procesar_todos_tickers(TICKERS) → 75 DataFrames
4. signal_detector.detectar_en_todos(datos) → lista de señales
5. Si no hay señales: registrar en log y terminar
6. enricher.obtener_contexto_macro() → datos macro (1 sola vez)
7. Para cada señal:
   a. enricher.enriquecer_senal(senal) → añade noticias + fundamentales
   b. ai_engine.analizar_senal(senal) → texto Gemini
   c. database.insertar_senal(senal) → guarda en Supabase
   d. notifier.enviar_alerta(senal) → Telegram
8. notifier.enviar_resumen_diario(stats)
9. database.registrar_ejecucion(stats)
10. Log de finalización con duración total
```

**Sistema de logging:**
- Usar Python logging con FileHandler (logs/pipeline_YYYYMMDD.log) y StreamHandler
- Nivel: INFO en producción, DEBUG en desarrollo
- Formato: `%(asctime)s | %(levelname)s | %(message)s`

---

## 15. MÓDULO: app.py (Frontend Streamlit)

**Propósito:** Dashboard web con 4 pestañas.

**Configuración página:**
```python
st.set_page_config(
    page_title="Plataforma Bursátil IA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)
```

**Conexión a Supabase:** usar `st.secrets` (funciona tanto local como en Streamlit Cloud).

**PESTAÑA 1 — Radar Activo:**
- Selector de ventana: últimas 24h / 48h / 7 días
- Tarjetas por señal con: ticker, precio, scoring (estrellas), RSI, ADX, sector, análisis Gemini
- Color card: verde para golden_cross, rojo para death_cross
- Badge de scoring: ⭐ / ⭐⭐ / ⭐⭐⭐🔥
- Si no hay señales: mensaje informativo con la última ejecución del CronJob

**PESTAÑA 2 — Laboratorio de Gráficos:**
- Selector de ticker (dropdown con los 75 de la lista)
- Selector de período: 6M / 1Y / 2Y
- Gráfico Plotly: velas japonesas + línea SMA 50 (azul) + línea SMA 200 (naranja)
- Markers en el gráfico en cada señal histórica: triángulo verde (golden) / rojo (death)
- Subgráfico inferior: RSI con líneas en 30 y 70
- Tabla debajo del gráfico: señales históricas de ese ticker

**PESTAÑA 3 — Bitácora:**
- Filtros: ticker (multiselect), tipo señal, scoring mínimo, rango de fechas
- Tabla interactiva con st.dataframe
- Botón exportar CSV
- Columnas: fecha, ticker, tipo, precio, scoring, RSI, ADX, sentimiento, análisis IA

**PESTAÑA 4 — Track Record:**
- Métricas principales en st.metric: total señales, hit rate %, retorno medio 30d, mejor sector
- Gráfico scatter: eje X = dist_sma_pct, eje Y = retorno_30d, color = tipo_evento, tamaño = adx_14
- Gráfico barras: hit rate por sector
- Gráfico línea: evolución del hit rate acumulado en el tiempo
- Nota: solo muestra señales con senal_ok IS NOT NULL (>30 días de antigüedad)

---

## 16. MÓDULO: src/tracker.py

**Propósito:** Script semanal que actualiza el seguimiento de señales antiguas.
Ejecutar manualmente o como segunda tarea en PythonAnywhere (domingo 09:00 UTC).

**Flujo:**
1. `database.get_senales_para_seguimiento()` → señales con >30 días sin precio_30d
2. Para cada señal: descargar precio actual y precio hace 7 y 30 días con yfinance
3. Calcular retornos: `((precio_Nd - precio_cierre) / precio_cierre) * 100`
4. `database.actualizar_seguimiento(id, precio_7d, precio_30d)`
5. Log de cuántas señales se actualizaron

---

## 17. REQUIREMENTS.TXT

Generar con exactamente estas dependencias (sin fijar versiones menores):

```
yfinance>=0.2.40
pandas>=2.0.0
pandas-ta>=0.3.14b
google-generativeai>=0.8.0
psycopg2-binary>=2.9.0
python-dotenv>=1.0.0
requests>=2.31.0
streamlit>=1.35.0
plotly>=5.20.0
sqlalchemy>=2.0.0
supabase>=2.4.0
```

---

## 18. ARCHIVOS DE CONFIGURACIÓN

**.gitignore completo:**
```
.env
venv/
__pycache__/
*.pyc
*.pyo
.DS_Store
logs/
data/
*.db
.streamlit/secrets.toml
.vscode/mcp.json
*.egg-info/
dist/
build/
.pytest_cache/
```

**.streamlit/secrets.toml (local, NO subir a GitHub):**
```toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "eyJ..."
SUPABASE_DB_PASSWORD = "tu_password"
```

---

## 19. ORDEN DE CONSTRUCCIÓN (para Claude en VS Code)

Construir en este orden exacto. Cada módulo es testeable antes de avanzar al siguiente.

```
FASE 1 — Base:
  1. config.py          → testar: python -m src.config (debe imprimir lista tickers)
  2. data_fetcher.py    → testar con 3 tickers: AAPL, MSFT, NVDA
  3. signal_detector.py → testar con datos históricos de NVDA (tuvo golden cross en 2023)

FASE 2 — Enriquecimiento:
  4. enricher.py        → testar con AAPL (llamadas reales a Alpha Vantage)
  5. database.py        → testar: insertar señal de prueba y leer de Supabase
  6. notifier.py        → testar: enviar mensaje de prueba al bot de Telegram

FASE 3 — Integración:
  7. ai_engine.py       → testar con señal de prueba completa
  8. main.py            → testar con lista reducida de 5 tickers
  9. tracker.py         → testar con señales de prueba en BD

FASE 4 — Frontend:
  10. app.py            → testar: streamlit run app.py (debe cargar con datos de BD)
```

---

## 20. COMANDOS DE DESARROLLO

```bash
# Activar entorno virtual (siempre al empezar)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo (test local)
python main.py

# Ejecutar solo el tracker de seguimiento
python -m src.tracker

# Lanzar frontend local
streamlit run app.py

# Ejecutar tests
python -m pytest tests/ -v

# Actualizar requirements.txt tras instalar nueva librería
pip freeze > requirements.txt

# Subir cambios a GitHub
git add .
git commit -m "descripción del cambio"
git push origin main
```

---

## 21. DESPLIEGUE EN PRODUCCIÓN

### PythonAnywhere (CronJob)
1. Crear cuenta en pythonanywhere.com (free tier)
2. Consola Bash: `git clone https://github.com/USUARIO/plataforma-bursatil.git`
3. Crear venv e instalar requirements.txt
4. Crear .env con las variables de producción
5. Tasks → Nueva tarea programada:
   - Hora: 03:00 UTC (= 22:00 EST / 23:00 hora España invierno)
   - Comando: `/home/USUARIO/plataforma-bursatil/venv/bin/python /home/USUARIO/plataforma-bursatil/main.py`
6. Para actualizar: `cd plataforma-bursatil && git pull`

### Streamlit Community Cloud (Frontend)
1. share.streamlit.io → conectar repo GitHub
2. Main file: app.py | Branch: main
3. Settings → Secrets: pegar contenido de secrets.toml
4. Deploy → URL pública automática

---

## 22. NOTAS IMPORTANTES PARA CLAUDE

- Todo el código debe ser Python 3.11+
- Usar type hints en todas las funciones
- Cada función debe tener docstring con descripción, args y returns
- Manejar TODAS las excepciones explícitamente — nunca usar `except: pass`
- Usar logging en lugar de print() en todos los módulos excepto app.py
- Los módulos en src/ no deben importarse entre sí circularmente
- database.py NO importa a ningún otro módulo de src/
- config.py es el único que importan TODOS los demás módulos
- Las llamadas a APIs externas siempre en bloques try/except con logging del error
- El pipeline main.py debe terminar correctamente aunque haya 0 señales
- Streamlit: usar st.cache_data para cachear las queries a Supabase (TTL: 300 segundos)

---

*Fin del documento CLAUDE.md — versión 2.0*
*Actualizado: Mayo 2026*
