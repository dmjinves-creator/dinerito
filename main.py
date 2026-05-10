"""Pipeline principal de DINERITO.

Ejecuta el ciclo completo de análisis diario:
  1. Descarga datos y calcula indicadores técnicos (75 tickers)
  2. Detecta cruces de medias (golden / death cross) con filtros ADX/RSI/volumen
  3. Enriquece cada señal con datos fundamentales, noticias y contexto macro
  4. Genera análisis IA con Gemini 2.5 Flash
  5. Guarda en Supabase y envía alertas por Telegram
  6. Registra resumen de ejecución
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup — FileHandler + StreamHandler
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    log_file = logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    fmt = "%(asctime)s | %(levelname)s | %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    _setup_logging()
    logger = logging.getLogger(__name__)

    start = time.time()
    logger.info("=== DINERITO pipeline iniciado ===")

    # ------------------------------------------------------------------
    # Imports after logging is configured so module-level loggers work
    # ------------------------------------------------------------------
    from src.config import TICKERS
    _tickers_env = os.getenv("TICKERS_TEST", "")
    if _tickers_env:
        TICKERS = [t.strip().upper() for t in _tickers_env.split(",") if t.strip()]
        logger.info("TICKERS_TEST override: %s", TICKERS)
    from src.data_fetcher import procesar_todos_tickers
    from src.signal_detector import detectar_en_todos
    from src.enricher import enriquecer_senal, obtener_contexto_macro
    from src.ai_engine import analizar_senal
    from src.database import insertar_senal, registrar_ejecucion
    from src.notifier import enviar_alerta, enviar_resumen_diario

    # ------------------------------------------------------------------
    # Step 1 — Download price data and compute indicators
    # ------------------------------------------------------------------
    logger.info("Descargando datos para %d tickers...", len(TICKERS))
    datos_tickers = procesar_todos_tickers(TICKERS)
    logger.info("%d tickers procesados correctamente", len(datos_tickers))

    # ------------------------------------------------------------------
    # Step 2 — Detect crossover signals
    # ------------------------------------------------------------------
    senales = detectar_en_todos(datos_tickers)
    logger.info("%d señal(es) detectada(s) tras los filtros", len(senales))

    n_tickers   = len(datos_tickers)
    n_senales   = 0
    n_errores   = 0

    if not senales:
        logger.info("Sin señales hoy — pipeline finaliza sin alertas")
        duracion = time.time() - start
        enviar_resumen_diario(0, n_tickers, duracion)
        registrar_ejecucion({
            "tickers_analizados":  n_tickers,
            "senales_detectadas":  0,
            "errores":             0,
            "duracion_seg":        round(duracion, 2),
            "estado":              "ok_sin_senales",
        })
        logger.info("=== Pipeline finalizado en %.1f s ===", duracion)
        return

    # ------------------------------------------------------------------
    # Step 3 — Macro context (fetched once for all signals)
    # ------------------------------------------------------------------
    logger.info("Obteniendo contexto macroeconómico...")
    macro = obtener_contexto_macro()
    logger.info("Macro: Fed %.2f%% | CPI %.2f%%",
                macro.get("fed_funds_rate", 0), macro.get("cpi_inflacion", 0))

    # ------------------------------------------------------------------
    # Steps 4-7 — Enrich, analyse, store, notify each signal
    # ------------------------------------------------------------------
    for senal in senales:
        ticker = senal.get("ticker", "?")
        try:
            # 4a. Fundamentals + news sentiment
            logger.info("[%s] Enriqueciendo señal...", ticker)
            senal = enriquecer_senal(senal)

            # 4b. Gemini AI analysis — returns dict with analisis + trading levels
            logger.info("[%s] Generando análisis IA...", ticker)
            ai_result = analizar_senal(senal)
            senal.update(ai_result)

            # 4c. Persist to Supabase
            inserted = insertar_senal(senal)
            if inserted:
                n_senales += 1

            # 4d. Telegram alert
            enviar_alerta(senal)

        except Exception as exc:
            logger.error("[%s] Error procesando señal: %s", ticker, exc)
            n_errores += 1

    # ------------------------------------------------------------------
    # Step 8-9 — Daily summary and execution log
    # ------------------------------------------------------------------
    duracion = time.time() - start
    estado   = "ok" if n_errores == 0 else "parcial"

    enviar_resumen_diario(n_senales, n_tickers, duracion)
    registrar_ejecucion({
        "tickers_analizados":  n_tickers,
        "senales_detectadas":  n_senales,
        "errores":             n_errores,
        "duracion_seg":        round(duracion, 2),
        "estado":              estado,
    })

    logger.info(
        "=== Pipeline finalizado en %.1f s | señales=%d | errores=%d ===",
        duracion, n_senales, n_errores,
    )


if __name__ == "__main__":
    run()
