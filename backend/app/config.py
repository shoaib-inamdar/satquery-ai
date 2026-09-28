"""
config.py — centralised configuration for SatQuery AI backend.
All paths, model names and tunable constants live here.
"""
from pathlib import Path
import os

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]          # satquery-ai/
BACKEND_DIR = ROOT / "backend"
SAMPLES_DIR = ROOT / "samples"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"

ADAPTER_PATH = MODELS_DIR / "adapters" / "bigearthnet_lora"
ADAPTER_EXISTS = (ADAPTER_PATH / "adapter_config.json").exists()

# ── VLM ───────────────────────────────────────────────────────────────────────
VLM_MODEL_ID = os.environ.get("SATQUERY_VLM", "HuggingFaceTB/SmolVLM-256M-Instruct")
VLM_MAX_NEW_TOKENS = 256
VLM_DEVICE = os.environ.get("SATQUERY_DEVICE", "cpu")   # "cuda" if GPU available

# ── Router ────────────────────────────────────────────────────────────────────
LAYA_CONFIDENCE_THRESHOLD = 0.55   # below this → fallback to RuleRouter

# ── Change detection ──────────────────────────────────────────────────────────
CHANGE_UNCHANGED_TOLERANCE = 0.01  # |Δ| < 1 % of AOI → "unchanged"

# ── API ───────────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path(os.environ.get("SATQUERY_UPLOAD_DIR", "/tmp/satquery_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 500 * 1024 * 1024   # 500 MB

# ── Overlaps ──────────────────────────────────────────────────────────────────
MIN_OVERLAP_FRACTION = 0.60    # reject pairs below this

# ── Confidence weights ────────────────────────────────────────────────────────
CONF_WEIGHT_INPUT   = 0.30
CONF_WEIGHT_SEP     = 0.40
CONF_WEIGHT_COREG   = 0.15
CONF_WEIGHT_ROUTER  = 0.15
