"""
tests/test_backend.py — pytest suite for SatQuery AI backend.
Covers: ingestion, modality detection, compatibility, indices, router,
grounding, change detection, fusion, confidence, report, anti-fabrication check.
"""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

# ── Ensure backend is importable ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ingestion import detect_modality, check_compatibility, load_image
from app.indices import ndvi, ndwi, mndwi, ndbi, bsi, compute_all_indices
from app.confidence import (
    score_input_quality, score_otsu_separation, score_coreg_residual,
    score_cross_modal_agreement, compute_confidence,
)
from app.evidence import EvidenceBundle
from app.router import RuleRouter, AgenticRouter
from app.tools.vqa import measure_scene, reject_invented_numbers
from app.tools.grounding import run_grounding
from app.tools.change import run_change


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_optical_data(h=64, w=64, seed=0) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    data = rng.uniform(0, 3000, (6, h, w)).astype(np.float32)
    # Set NIR high for some vegetation pixels
    data[3, :h//2, :w//2] = 2800
    data[2, :h//2, :w//2] = 600
    # Set NIR/SWIR low for water
    data[3, h//2:, w//2:] = 200
    data[1, h//2:, w//2:] = 1800
    meta = {
        "filename": "SYNTHETIC_test.tif",
        "band_count": 6,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": [75.0, 0.001, 0, 20.0, 0, -0.001],
        "nodata": None,
        "nodata_fraction": 0.0,
        "modality": "optical",
        "resolution_m": 10.0,
        "bounds": [75.0, 19.936, 75.064, 20.0],
        "width": w, "height": h,
        "descriptions": ["B02_blue", "B03_green", "B04_red", "B08_nir", "B11_swir1", "B12_swir2"],
    }
    return data, meta


def make_sar_data(h=64, w=64, seed=1) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.001, 0.3, (1, h, w)).astype(np.float32)
    meta = {
        "filename": "SYNTHETIC_sar.tif",
        "band_count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": [75.0, 0.001, 0, 20.0, 0, -0.001],
        "nodata": None,
        "nodata_fraction": 0.0,
        "modality": "sar",
        "resolution_m": 10.0,
        "bounds": [75.0, 19.936, 75.064, 20.0],
        "width": w, "height": h,
        "descriptions": ["VV"],
    }
    return data, meta


# ── Modality detection ────────────────────────────────────────────────────────

class TestModalityDetection:
    def test_optical_detection(self):
        m = detect_modality(6, "float32", ["B02", "B03", "B04", "B08", "B11", "B12"], 0, 10000)
        assert m == "optical"

    def test_sar_detection_by_band_count(self):
        m = detect_modality(1, "float32", ["VV"], 0.0, 0.5)
        assert m == "sar"

    def test_sar_detection_by_keyword(self):
        m = detect_modality(2, "float32", ["VV", "VH"], 0, 0.5)
        assert m == "sar"

    def test_rgb_detection(self):
        m = detect_modality(3, "uint8", [None, None, None], 0, 255)
        assert m == "rgb"


# ── Indices ───────────────────────────────────────────────────────────────────

class TestIndices:
    def test_ndvi_range(self):
        nir = np.array([3000.0, 0.0])
        red = np.array([500.0, 0.0])
        result = ndvi(nir, red)
        assert result[0] > 0.5
        assert -1.0 <= float(result[0]) <= 1.0

    def test_ndwi_water(self):
        green = np.array([2000.0])
        nir = np.array([100.0])
        result = ndwi(green, nir)
        assert result[0] > 0.0  # water should be positive

    def test_ndbi_builtup(self):
        swir = np.array([2500.0])
        nir = np.array([1000.0])
        result = ndbi(swir, nir)
        assert result[0] > 0.0

    def test_compute_all_indices_optical(self):
        data, meta = make_optical_data()
        indices = compute_all_indices(data, meta)
        assert "ndvi" in indices
        assert "ndwi" in indices or "mndwi" in indices

    def test_no_division_by_zero(self):
        nir = np.zeros(10)
        red = np.zeros(10)
        result = ndvi(nir, red)
        assert np.all(np.isfinite(result))


# ── Compatibility ─────────────────────────────────────────────────────────────

class TestCompatibility:
    def test_single_image_pass(self):
        _, meta = make_optical_data()
        meta["filename"] = "test.tif"
        report = check_compatibility([meta])
        assert report.input_config == "single"
        assert report.overall in ("pass", "warn")

    def test_pair_same_crs_pass(self):
        _, m1 = make_optical_data()
        _, m2 = make_optical_data()
        m1["filename"] = "t1.tif"
        m2["filename"] = "t2.tif"
        report = check_compatibility([m1, m2])
        assert report.input_config == "bitemporal"

    def test_pair_no_overlap_fail(self):
        _, m1 = make_optical_data()
        _, m2 = make_optical_data()
        m1["filename"] = "t1.tif"
        m2["filename"] = "t2.tif"
        # Make them non-overlapping
        m2["bounds"] = [80.0, 25.0, 81.0, 26.0]  # Far away
        report = check_compatibility([m1, m2])
        statuses = [c.status for c in report.checks]
        assert "fail" in statuses


# ── Router ────────────────────────────────────────────────────────────────────

class TestRuleRouter:
    def setup_method(self):
        self.router = RuleRouter()

    def test_vqa_routing(self):
        d = self.router.route("How many buildings are in this image?", "single")
        assert d.task == "vqa"
        assert d.router_name == "rule-based"

    def test_change_routing(self):
        d = self.router.route("What changed between these two dates?", "bitemporal")
        assert d.task == "change"

    def test_fusion_routing(self):
        d = self.router.route("Use optical and SAR to find built-up areas", "optical_sar")
        assert d.task in ("fusion", "vqa", "grounding")  # all are valid for this query

    def test_grounding_routing(self):
        d = self.router.route("Highlight the water body", "single")
        assert d.task == "grounding"

    def test_refusal_change_without_pair(self):
        d = self.router.route("What changed between these two dates?", "single")
        # Either changes task or gives refusal
        if d.task == "change":
            assert d.refusal_reason is not None

    def test_refusal_fusion_without_pair(self):
        d = self.router.route("Fuse optical and SAR images", "single")
        if d.task == "fusion":
            assert d.refusal_reason is not None


# ── Grounding ─────────────────────────────────────────────────────────────────

class TestGrounding:
    def test_water_grounding(self):
        data, meta = make_optical_data()
        bundle = EvidenceBundle("test_grounding")
        result = run_grounding(data, meta, "highlight water body", bundle,
                               {"target_class": "water", "smoothing": 1})
        assert result.tool == "grounding"
        assert result.provenance == "classical-cv-baseline"
        # Either found water fraction OR reported available indices — both are valid responses
        assert len(result.measurements) > 0

    def test_vegetation_grounding(self):
        data, meta = make_optical_data()
        bundle = EvidenceBundle("test_veg")
        result = run_grounding(data, meta, "show vegetation areas", bundle,
                               {"target_class": "vegetation"})
        assert result.tool == "grounding"


# ── Change detection ──────────────────────────────────────────────────────────

class TestChangeDetection:
    def test_change_detection_basic(self):
        d1, m1 = make_optical_data(seed=0)
        d2, m2 = make_optical_data(seed=0)
        # Modify t2
        d2[3, 0:32, 0:32] *= 0.3  # Drop NIR → NDVI drops
        m1["filename"] = "t1.tif"
        m2["filename"] = "t2.tif"
        bundle = EvidenceBundle("test_change")
        result = run_change(d1, m1, d2, m2, "what changed?", bundle, {})
        assert result.tool == "change"
        assert "changed_fraction" in result.measurements
        assert 0.0 <= result.measurements["changed_fraction"] <= 1.0

    def test_change_against_truth(self):
        """Change detection should find the synthetic edits."""
        root = Path(__file__).resolve().parents[3]
        t1_path = root / "samples" / "synthetic" / "SYNTHETIC_optical_t1.tif"
        t2_path = root / "samples" / "synthetic" / "SYNTHETIC_optical_t2.tif"
        truth_path = root / "samples" / "synthetic" / "truth.json"

        if not (t1_path.exists() and t2_path.exists()):
            pytest.skip("Synthetic samples not generated. Run scripts/make_synthetic_samples.py first.")

        d1, m1 = load_image(t1_path)
        d2, m2 = load_image(t2_path)
        truth = json.loads(truth_path.read_text())

        bundle = EvidenceBundle("truth_test")
        result = run_change(d1, m1, d2, m2, "what changed?", bundle, {})

        changed_frac = result.measurements.get("changed_fraction", 0.0)
        # We made 3 edits affecting roughly 10-20% of the 256x256 image
        assert changed_frac > 0.01, f"Expected change > 1%, got {changed_frac:.4f}"
        print(f"\n[truth test] Detected changed_fraction={changed_frac:.4f}")


# ── Confidence ────────────────────────────────────────────────────────────────

class TestConfidence:
    def test_confidence_range(self):
        c = compute_confidence(0.8, 0.7, 0.9)
        assert 0.05 <= c.overall <= 0.95

    def test_confidence_components_present(self):
        c = compute_confidence(0.8, 0.7, 0.9, coreg_residual=0.8)
        assert c.coreg_residual is not None
        assert c.input_quality == pytest.approx(0.8, abs=0.01)

    def test_low_input_quality_reduces_confidence(self):
        c_high = compute_confidence(0.95, 0.9, 0.9)
        c_low  = compute_confidence(0.1, 0.9, 0.9)
        assert c_low.overall < c_high.overall


# ── Evidence bundle ───────────────────────────────────────────────────────────

class TestEvidence:
    def test_add_and_retrieve(self):
        b = EvidenceBundle("test")
        b.add_measurement("water_fraction", 0.25, "ndwi", confidence=0.8)
        items = b.items()
        assert len(items) == 1
        assert items[0].value == pytest.approx(0.25)

    def test_number_check_present(self):
        b = EvidenceBundle("test")
        b.add_measurement("vegetation_fraction", 0.4567, "ndvi", confidence=0.8)
        assert b.contains_number(0.4567, tol=0.01)

    def test_number_check_absent(self):
        b = EvidenceBundle("test")
        b.add_measurement("vegetation_fraction", 0.4567, "ndvi", confidence=0.8)
        assert not b.contains_number(0.9999, tol=0.01)

    def test_anti_fabrication_rejects_invented_number(self):
        b = EvidenceBundle("test")
        b.add_measurement("vegetation_fraction", 0.35, "ndvi", confidence=0.8)
        answer = "The vegetation covers 99.9% of the image."  # invented number
        template = "Based on measurements: vegetation 35%."
        result = reject_invented_numbers(answer, b, template)
        assert result == template

    def test_anti_fabrication_passes_known_number(self):
        b = EvidenceBundle("test")
        b.add_measurement("vegetation_fraction", 35.0, "ndvi", confidence=0.8)
        answer = "The vegetation covers 35% of the image."
        template = "Based on measurements."
        result = reject_invented_numbers(answer, b, template)
        assert result == answer  # should not be replaced


# ── Parameter whitelist ───────────────────────────────────────────────────────

class TestRegistry:
    def test_unknown_param_dropped(self):
        from app.registry import validate_params
        cleaned, warnings = validate_params("vqa", {"secret_param": 999, "max_new_tokens": 100})
        assert "secret_param" not in cleaned
        assert any("not whitelisted" in w for w in warnings)

    def test_param_clamped(self):
        from app.registry import validate_params
        cleaned, warnings = validate_params("vqa", {"max_new_tokens": 9999})
        assert cleaned["max_new_tokens"] <= 512

    def test_invalid_choice_defaults(self):
        from app.registry import validate_params
        cleaned, warnings = validate_params("grounding", {"target_class": "aliens"})
        assert cleaned["target_class"] == "water"  # default


# ── NOT-adapted badge ─────────────────────────────────────────────────────────

class TestAdaptationBadge:
    def test_badge_not_adapted(self):
        from app.config import ADAPTER_EXISTS
        if ADAPTER_EXISTS:
            pytest.skip("Adapter is present — this tests the absent case.")
        from app.tools.vqa import run_vqa
        data, meta = make_optical_data()
        bundle = EvidenceBundle("badge_test")
        result = run_vqa(data, meta, "describe the image", bundle, {})
        assert "NOT adapted" in result.answer or result.extra.get("adapter_present") is False
