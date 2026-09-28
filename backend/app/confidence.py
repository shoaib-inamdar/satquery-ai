"""
confidence.py — Computed confidence with auditable breakdown.
Formula: weighted geometric mean of components. Clamped to [0.05, 0.95].
Never presented as a calibrated probability.
"""
from __future__ import annotations
import math
import numpy as np
from .schemas import ConfidenceBreakdown
from .config import (
    CONF_WEIGHT_INPUT, CONF_WEIGHT_SEP,
    CONF_WEIGHT_COREG, CONF_WEIGHT_ROUTER,
)


def _clamp(v: float) -> float:
    return max(0.05, min(0.95, float(v)))


def _weighted_geometric_mean(components: dict[str, tuple[float, float]]) -> float:
    """
    components: {name: (value, weight)}
    Returns weighted geometric mean.
    """
    log_sum = 0.0
    total_w = 0.0
    for name, (v, w) in components.items():
        v = max(v, 1e-6)
        log_sum += w * math.log(v)
        total_w += w
    if total_w == 0:
        return 0.5
    return math.exp(log_sum / total_w)


# ── Component scorers ─────────────────────────────────────────────────────────

def score_input_quality(band_availability: float, nodata_fraction: float, resolution_ok: bool) -> float:
    """
    band_availability: fraction of expected bands present [0,1].
    nodata_fraction: fraction of nodata pixels [0,1].
    resolution_ok: True if resolution is suitable.
    """
    avail = _clamp(band_availability)
    nodata_penalty = _clamp(1.0 - nodata_fraction * 2.0)
    res_bonus = 1.0 if resolution_ok else 0.7
    return _clamp(avail * nodata_penalty * res_bonus)


def score_otsu_separation(band: np.ndarray, threshold: float) -> float:
    """
    Otsu between-class variance ratio as a separation quality score.
    Returns value in [0, 1].
    """
    flat = band[np.isfinite(band)].astype(np.float64)
    if flat.size < 10:
        return 0.1
    total_var = float(np.var(flat))
    if total_var < 1e-12:
        return 0.1

    below = flat[flat <= threshold]
    above = flat[flat > threshold]
    if below.size == 0 or above.size == 0:
        return 0.1

    w0 = len(below) / len(flat)
    w1 = len(above) / len(flat)
    between_var = w0 * w1 * (float(np.mean(above)) - float(np.mean(below))) ** 2
    ratio = between_var / total_var
    return _clamp(ratio)


def score_coreg_residual(shift_px: float, max_acceptable_px: float = 5.0) -> float:
    """Lower residual → higher score. shift_px ≥ max_acceptable_px → score ≈ 0.1."""
    return _clamp(1.0 - min(shift_px / max_acceptable_px, 1.0) * 0.85)


def score_cross_modal_agreement(agreement_fraction: float) -> float:
    """fraction of pixels where optical and SAR agree → score."""
    return _clamp(agreement_fraction)


# ── Main assembler ────────────────────────────────────────────────────────────

def compute_confidence(
    input_quality: float,
    separation_quality: float,
    router_confidence: float,
    coreg_residual: float | None = None,
    cross_modal_agreement: float | None = None,
) -> ConfidenceBreakdown:
    components: dict[str, tuple[float, float]] = {
        "input_quality": (input_quality, CONF_WEIGHT_INPUT),
        "separation_quality": (separation_quality, CONF_WEIGHT_SEP),
        "router_confidence": (router_confidence, CONF_WEIGHT_ROUTER),
    }
    if coreg_residual is not None:
        components["coreg_residual"] = (coreg_residual, CONF_WEIGHT_COREG)
    if cross_modal_agreement is not None:
        components["cross_modal_agreement"] = (cross_modal_agreement, CONF_WEIGHT_COREG)

    overall = _clamp(_weighted_geometric_mean(components))

    return ConfidenceBreakdown(
        overall=round(overall, 3),
        input_quality=round(input_quality, 3),
        separation_quality=round(separation_quality, 3),
        coreg_residual=round(coreg_residual, 3) if coreg_residual is not None else None,
        cross_modal_agreement=round(cross_modal_agreement, 3) if cross_modal_agreement is not None else None,
        router_confidence=round(router_confidence, 3),
        formula=(
            "weighted_geometric_mean("
            f"input={CONF_WEIGHT_INPUT}, sep={CONF_WEIGHT_SEP}, "
            f"coreg/agree={CONF_WEIGHT_COREG}, router={CONF_WEIGHT_ROUTER})"
        ),
    )
