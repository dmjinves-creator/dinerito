# DINERITO — Plataforma Bursátil IA

Detecta Golden Cross / Death Cross en 75 tickers del S&P 500 y Nasdaq. Genera análisis con Gemini 2.5 Flash, guarda en Supabase y envía alertas por Telegram.

## Setup local

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # rellena con tus credenciales
python main.py                # ejecuta el pipeline
streamlit run app.py          # abre el dashboard
```

## Variables de entorno (.env)

| Variable | Descripción |
|---|---|
| `SUPABASE_URL` | URL de tu proyecto Supabase |
| `SUPABASE_KEY` | Clave pública (publishable) |
| `SUPABASE_DB_PASSWORD` | Contraseña de la BD |
| `SUPABASE_POOLER_REGION` | Región del pooler (ej. `eu-west-1`) |
| `GEMINI_API_KEY` | API key de Google AI Studio |
| `ALPHA_VANTAGE_KEY` | API key de Alpha Vantage (free tier) |
| `TELEGRAM_BOT_TOKEN` | Token del bot de Telegram |
| `TELEGRAM_CHAT_ID` | Chat ID donde llegan las alertas |

## Estructura

```
main.py          # pipeline diario (cron 22:00 EST)
app.py           # dashboard Streamlit (4 pestañas)
src/
  config.py      # constantes y lista de 75 tickers
  data_fetcher.py  # descarga yfinance + indicadores
  signal_detector.py  # detección Golden/Death Cross
  enricher.py    # Alpha Vantage: noticias + fundamentales
  ai_engine.py   # análisis Gemini 2.5 Flash
  database.py    # Supabase PostgreSQL (psycopg2)
  notifier.py    # alertas Telegram
  tracker.py     # seguimiento semanal de señales
```

## Despliegue

- **Pipeline**: PythonAnywhere — tarea cron diaria a las 22:00 EST  
- **Dashboard**: Streamlit Cloud — conecta el repo y configura secrets
