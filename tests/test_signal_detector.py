"""Tests for src/signal_detector.py — all branches of the cascade filter system."""

import pandas as pd
import pytest

from src.signal_detector import detectar_cruce, detectar_en_todos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(sma50: float, sma200: float, *,
         rsi: float = 50.0,
         adx: float = 30.0,
         volume: float = 2_000_000,
         vol_ma20: float = 1_000_000,
         close: float | None = None) -> dict:
    """Build a single-row dict that mimics the DataFrame produced by data_fetcher."""
    price = close if close is not None else sma50
    return {
        "Close":    price,
        "High":     price * 1.01,
        "Low":      price * 0.99,
        "Volume":   volume,
        "VOL_MA20": vol_ma20,
        "SMA_50":   sma50,
        "SMA_200":  sma200,
        "RSI_14":   rsi,
        "ADX_14":   adx,
    }


def _df(*rows: dict) -> pd.DataFrame:
    """Build a DatetimeIndex DataFrame from one or more row dicts."""
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="B")
    return pd.DataFrame(list(rows), index=idx)


# ---------------------------------------------------------------------------
# 1. Golden Cross válido — supera los 3 filtros
# ---------------------------------------------------------------------------

class TestGoldenCrossValido:
    def test_tipo_evento(self):
        """Golden Cross is labelled correctly."""
        df = _df(_row(99, 100), _row(101, 100))
        r = detectar_cruce(df, "GC")
        assert r is not None
        assert r["tipo_evento"] == "golden_cross"

    def test_campos_requeridos(self):
        """Result contains all fields needed for the database insert."""
        df = _df(_row(99, 100), _row(101, 100))
        r = detectar_cruce(df, "GC")
        required = {
            "fecha_evento", "ticker", "tipo_evento", "precio_cierre",
            "sma_50", "sma_200", "rsi_14", "adx_14",
            "volumen_relativo", "dist_sma_pct", "scoring",
        }
        assert required.issubset(r.keys())

    def test_dist_y_scoring(self):
        """dist_sma_pct and scoring are computed correctly for a 1 % cross."""
        # sma50=101, sma200=100 → dist = +1.0 % → scoring 2
        df = _df(_row(99, 100), _row(101, 100))
        r = detectar_cruce(df, "GC")
        assert abs(r["dist_sma_pct"] - 1.0) < 0.01
        assert r["scoring"] == 2

    def test_volumen_relativo(self):
        """volumen_relativo is volume / vol_ma20."""
        df = _df(_row(99, 100), _row(101, 100, volume=3_000_000, vol_ma20=1_000_000))
        r = detectar_cruce(df, "GC")
        assert r["volumen_relativo"] == pytest.approx(3.0, abs=0.01)


# ---------------------------------------------------------------------------
# 2. Golden Cross descartado por ADX bajo (Filtro 1)
# ---------------------------------------------------------------------------

class TestGoldenCrossDescartadoADX:
    def test_adx_justo_por_debajo(self):
        """ADX = 24 (< 25 threshold) must suppress the signal."""
        df = _df(_row(99, 100), _row(101, 100, adx=24.0))
        assert detectar_cruce(df, "GC_ADX") is None

    def test_adx_en_umbral_exacto_pasa(self):
        """ADX exactly at 25 must pass filter 1."""
        df = _df(_row(99, 100), _row(101, 100, adx=25.0))
        assert detectar_cruce(df, "GC_ADX_OK") is not None

    def test_adx_muy_bajo(self):
        """Very low ADX (flat market) also rejected."""
        df = _df(_row(99, 100), _row(101, 100, adx=10.0))
        assert detectar_cruce(df, "GC_ADX_LOW") is None


# ---------------------------------------------------------------------------
# 3. Death Cross válido — supera los 3 filtros
# ---------------------------------------------------------------------------

class TestDeathCrossValido:
    def test_tipo_evento(self):
        """Death Cross is labelled correctly."""
        df = _df(_row(101, 100), _row(99, 100))
        r = detectar_cruce(df, "DC")
        assert r is not None
        assert r["tipo_evento"] == "death_cross"

    def test_dist_negativa(self):
        """dist_sma_pct must be negative for a Death Cross."""
        df = _df(_row(101, 100), _row(99, 100))
        r = detectar_cruce(df, "DC")
        assert r["dist_sma_pct"] < 0

    def test_scoring_fuerte(self):
        """A 2 % cross (sma50=98, sma200=100) earns scoring 3."""
        df = _df(_row(103, 100), _row(98, 100))
        r = detectar_cruce(df, "DC_SC3")
        assert r is not None
        assert r["scoring"] == 3


# ---------------------------------------------------------------------------
# 4. Sin cruce detectado
# ---------------------------------------------------------------------------

class TestSinCruce:
    def test_sma50_siempre_encima(self):
        """No cross when SMA_50 remains above SMA_200 both days."""
        df = _df(_row(105, 100), _row(107, 100))
        assert detectar_cruce(df, "NC_ABOVE") is None

    def test_sma50_siempre_debajo(self):
        """No cross when SMA_50 remains below SMA_200 both days."""
        df = _df(_row(95, 100), _row(94, 100))
        assert detectar_cruce(df, "NC_BELOW") is None

    def test_smas_iguales(self):
        """No cross when SMAs are equal both days (flat)."""
        df = _df(_row(100, 100), _row(100, 100))
        assert detectar_cruce(df, "NC_FLAT") is None

    def test_menos_de_dos_filas(self):
        """Returns None and does not raise when DataFrame has only 1 row."""
        df = _df(_row(101, 100))
        assert detectar_cruce(df, "NC_SHORT") is None


# ---------------------------------------------------------------------------
# 5. Filtro 2 — volumen insuficiente
# ---------------------------------------------------------------------------

class TestFiltroVolumen:
    def test_volumen_bajo_rechaza_golden(self):
        """volume / vol_ma20 = 0.5 < 1.2 → Golden Cross rejected."""
        df = _df(_row(99, 100), _row(101, 100, volume=500_000, vol_ma20=1_000_000))
        assert detectar_cruce(df, "VOL_GC") is None

    def test_volumen_bajo_rechaza_death(self):
        """Low volume also rejects Death Cross."""
        df = _df(_row(101, 100), _row(99, 100, volume=500_000, vol_ma20=1_000_000))
        assert detectar_cruce(df, "VOL_DC") is None

    def test_volumen_en_umbral_exacto_pasa(self):
        """volume = vol_ma20 * 1.2 exactly must pass filter 2."""
        df = _df(_row(99, 100), _row(101, 100, volume=1_200_000, vol_ma20=1_000_000))
        assert detectar_cruce(df, "VOL_OK") is not None


# ---------------------------------------------------------------------------
# 6. Filtro 3 — RSI en zona extrema
# ---------------------------------------------------------------------------

class TestFiltroRSI:
    def test_rsi_sobrecomprado_rechaza_golden(self):
        """RSI 78 > 75 → Golden Cross is an overbought trap, must be rejected."""
        df = _df(_row(99, 100), _row(101, 100, rsi=78.0))
        assert detectar_cruce(df, "RSI_GC") is None

    def test_rsi_75_exacto_pasa_golden(self):
        """RSI exactly at 75 is the boundary: Golden Cross must pass."""
        df = _df(_row(99, 100), _row(101, 100, rsi=75.0))
        assert detectar_cruce(df, "RSI_GC_OK") is not None

    def test_rsi_sobrevendido_rechaza_death(self):
        """RSI 20 < 25 → Death Cross is an oversold trap, must be rejected."""
        df = _df(_row(101, 100), _row(99, 100, rsi=20.0))
        assert detectar_cruce(df, "RSI_DC") is None

    def test_rsi_25_exacto_pasa_death(self):
        """RSI exactly at 25 is the boundary: Death Cross must pass."""
        df = _df(_row(101, 100), _row(99, 100, rsi=25.0))
        assert detectar_cruce(df, "RSI_DC_OK") is not None


# ---------------------------------------------------------------------------
# 7. Scoring thresholds
# ---------------------------------------------------------------------------

class TestScoring:
    def test_scoring_1_dist_menor_05(self):
        """dist = 0.2 % → scoring 1 (weak)."""
        df = _df(_row(99.8, 100), _row(100.2, 100))
        r = detectar_cruce(df, "SC1")
        assert r is not None and r["scoring"] == 1

    def test_scoring_2_dist_entre_05_y_15(self):
        """dist = 1.0 % → scoring 2 (medium)."""
        df = _df(_row(99, 100), _row(101, 100))
        r = detectar_cruce(df, "SC2")
        assert r is not None and r["scoring"] == 2

    def test_scoring_3_dist_mayor_15(self):
        """dist = 2.0 % → scoring 3 (strong)."""
        df = _df(_row(97, 100), _row(102, 100))
        r = detectar_cruce(df, "SC3")
        assert r is not None and r["scoring"] == 3

    def test_scoring_death_cross_usa_abs(self):
        """Scoring uses absolute value of dist: -2 % also gives scoring 3."""
        df = _df(_row(103, 100), _row(98, 100))
        r = detectar_cruce(df, "SC3_DC")
        assert r is not None and r["scoring"] == 3


# ---------------------------------------------------------------------------
# 8. detectar_en_todos
# ---------------------------------------------------------------------------

class TestDetectarEnTodos:
    def test_lista_vacia(self):
        """Empty input returns empty list without raising."""
        assert detectar_en_todos([]) == []

    def test_un_ticker_con_senal(self):
        """Single ticker with a valid cross returns one signal."""
        df = _df(_row(99, 100), _row(101, 100))
        result = detectar_en_todos([{"ticker": "GC", "df": df}])
        assert len(result) == 1
        assert result[0]["ticker"] == "GC"

    def test_mezcla_con_y_sin_senal(self):
        """Only tickers with qualifying crosses are returned."""
        gc_df = _df(_row(99, 100), _row(101, 100))
        no_df = _df(_row(105, 100), _row(106, 100))
        result = detectar_en_todos([
            {"ticker": "GC", "df": gc_df},
            {"ticker": "NONE", "df": no_df},
        ])
        assert len(result) == 1
        assert result[0]["ticker"] == "GC"

    def test_multiples_senales(self):
        """Multiple qualifying crosses are all returned."""
        gc_df = _df(_row(99, 100), _row(101, 100))
        dc_df = _df(_row(101, 100), _row(99, 100))
        result = detectar_en_todos([
            {"ticker": "GC", "df": gc_df},
            {"ticker": "DC", "df": dc_df},
        ])
        tipos = {r["ticker"]: r["tipo_evento"] for r in result}
        assert tipos == {"GC": "golden_cross", "DC": "death_cross"}

    def test_df_invalido_no_detiene_pipeline(self):
        """A corrupt DataFrame for one ticker must not prevent others from being processed."""
        bad_df = pd.DataFrame()   # empty — will log a warning and skip
        gc_df  = _df(_row(99, 100), _row(101, 100))
        result = detectar_en_todos([
            {"ticker": "BAD", "df": bad_df},
            {"ticker": "GC",  "df": gc_df},
        ])
        tickers = [r["ticker"] for r in result]
        assert "GC" in tickers
        assert "BAD" not in tickers
