"""
tools/vqa.py — Visual Question Answering tool.
Evidence-first: measurements from pixels are computed first,
then the VLM rephrases them. Numbers not in the evidence bundle are rejected.
"""
from __future__ import annotations
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..config import VLM_MODEL_ID, VLM_MAX_NEW_TOKENS, VLM_DEVICE, ADAPTER_PATH, ADAPTER_EXISTS
from ..evidence import EvidenceBundle
from ..indices import compute_all_indices, extract_bands
from ..schemas import ToolResult


# ── Scene measurement ─────────────────────────────────────────────────────────

def measure_scene(data: np.ndarray, meta: dict) -> dict[str, Any]:
    """Compute land-cover fractions and index statistics from pixels."""
    indices = compute_all_indices(data, meta)
    measurements: dict[str, Any] = {}

    def frac_above(arr: np.ndarray, thr: float) -> float:
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return 0.0
        return float((valid > thr).sum() / valid.size)

    def stat(arr: np.ndarray, name: str):
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return
        measurements[f"{name}_mean"] = round(float(valid.mean()), 4)
        measurements[f"{name}_std"] = round(float(valid.std()), 4)
        measurements[f"{name}_min"] = round(float(valid.min()), 4)
        measurements[f"{name}_max"] = round(float(valid.max()), 4)

    for idx_name, arr in indices.items():
        stat(arr, idx_name)

    # Land-cover fractions
    if "ndvi" in indices:
        measurements["vegetation_fraction"] = round(frac_above(indices["ndvi"], 0.3), 4)
        measurements["dense_vegetation_fraction"] = round(frac_above(indices["ndvi"], 0.5), 4)
    if "ndwi" in indices:
        measurements["water_fraction_ndwi"] = round(frac_above(indices["ndwi"], 0.0), 4)
    if "mndwi" in indices:
        measurements["water_fraction_mndwi"] = round(frac_above(indices["mndwi"], 0.0), 4)
    if "ndbi" in indices:
        measurements["builtup_fraction"] = round(frac_above(indices["ndbi"], 0.0), 4)
    if "bsi" in indices:
        measurements["bare_soil_fraction"] = round(frac_above(indices["bsi"], 0.0), 4)

    measurements["band_count"] = int(data.shape[0])
    measurements["height_px"] = int(data.shape[1])
    measurements["width_px"] = int(data.shape[2])

    return measurements


def measurements_to_summary(m: dict) -> str:
    """Turn measurements dict into a compact text summary for the VLM prompt."""
    lines = []
    if "vegetation_fraction" in m:
        lines.append(f"Vegetation cover: {m['vegetation_fraction']*100:.1f}%")
    if "water_fraction_mndwi" in m:
        lines.append(f"Water fraction: {m['water_fraction_mndwi']*100:.1f}%")
    elif "water_fraction_ndwi" in m:
        lines.append(f"Water fraction: {m['water_fraction_ndwi']*100:.1f}%")
    if "builtup_fraction" in m:
        lines.append(f"Built-up fraction: {m['builtup_fraction']*100:.1f}%")
    if "bare_soil_fraction" in m:
        lines.append(f"Bare soil fraction: {m['bare_soil_fraction']*100:.1f}%")
    if "ndvi_mean" in m:
        lines.append(f"Mean NDVI: {m['ndvi_mean']:.3f}")
    return "; ".join(lines) if lines else "Band statistics computed."


# ── Anti-fabrication number check ─────────────────────────────────────────────

def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"\d+\.?\d*", text)]


def reject_invented_numbers(answer: str, bundle: EvidenceBundle, template: str) -> str:
    """
    If any number in `answer` is not in the evidence bundle,
    replace answer with `template`.
    """
    numbers_in_answer = _extract_numbers(answer)
    for n in numbers_in_answer:
        if not bundle.contains_number(n, tol=0.5):
            return template
    return answer


# ── VLM loading (lazy, thread-safe) ──────────────────────────────────────────
_vlm_lock = threading.Lock()
_vlm_model = None
_vlm_processor = None
_vlm_loaded = False
_vlm_load_error: str | None = None


def _load_vlm():
    global _vlm_model, _vlm_processor, _vlm_loaded, _vlm_load_error
    with _vlm_lock:
        if _vlm_loaded:
            return
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
            _vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
            _vlm_model = AutoModelForVision2Seq.from_pretrained(
                VLM_MODEL_ID,
                torch_dtype=torch.float16 if VLM_DEVICE != "cpu" else torch.float32,
            ).to(VLM_DEVICE)
            # Load LoRA adapter if present
            if ADAPTER_EXISTS:
                from peft import PeftModel
                _vlm_model = PeftModel.from_pretrained(
                    _vlm_model,
                    str(ADAPTER_PATH),
                )
            _vlm_model.eval()
            _vlm_loaded = True
        except Exception as e:
            _vlm_load_error = str(e)
            _vlm_loaded = True  # prevent re-attempts


def vlm_answer(question: str, image_rgb: np.ndarray | None, context: str) -> tuple[str, str]:
    """
    Run the VLM. Returns (answer_text, provenance).
    Falls back to template if the VLM cannot load.
    """
    _load_vlm()
    if _vlm_load_error or _vlm_model is None:
        return (
            f"[classical-cv-baseline] Based on pixel measurements: {context}",
            "classical-cv-baseline",
        )
    try:
        import torch
        prompt_text = (
            f"You are a remote-sensing analyst. "
            f"Measured scene statistics: {context}. "
            f"Question: {question} "
            f"Answer using only the statistics provided. Be concise."
        )
        if image_rgb is not None:
            from PIL import Image as PILImage  # lazy import — not available without Pillow
            pil_img = PILImage.fromarray(image_rgb)
            inputs = _vlm_processor(
                text=[{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ]}],
                images=[pil_img],
                return_tensors="pt",
            ).to(VLM_DEVICE)
        else:
            inputs = _vlm_processor(
                text=[{"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                ]}],
                return_tensors="pt",
            ).to(VLM_DEVICE)

        with torch.no_grad():
            out_ids = _vlm_model.generate(
                **inputs,
                max_new_tokens=VLM_MAX_NEW_TOKENS,
                do_sample=False,
            )
        answer = _vlm_processor.decode(out_ids[0], skip_special_tokens=True)
        # Strip the prompt echo if model repeats it
        if prompt_text in answer:
            answer = answer.split(prompt_text)[-1].strip()
        provenance = "real-model"
        if ADAPTER_EXISTS:
            provenance = "real-model"  # RS-adapted
        return answer, provenance
    except Exception as e:
        return (
            f"[classical-cv-baseline] Measurements: {context}",
            "classical-cv-baseline",
        )


# ── Public tool entry point ───────────────────────────────────────────────────

def run_vqa(
    data: np.ndarray,
    meta: dict,
    query: str,
    bundle: EvidenceBundle,
    params: dict,
    image_rgb: np.ndarray | None = None,
) -> ToolResult:
    """
    1. Compute scene measurements from pixels.
    2. Add to evidence bundle.
    3. Call VLM with measurements as context.
    4. Reject any numbers in the answer not in the bundle.
    """
    measurements = measure_scene(data, meta)
    summary = measurements_to_summary(measurements)

    # Record measurements in evidence bundle
    for k, v in measurements.items():
        if isinstance(v, (int, float)):
            bundle.add_measurement(
                name=k, value=v,
                method="spectral-indices" if "fraction" in k or k.startswith("ndv") else "band-stats",
                provenance="classical-cv-baseline",
                confidence=0.75,
            )

    answer, provenance = vlm_answer(query, image_rgb, summary)

    template = f"Based on pixel measurements: {summary}. Unable to answer further from available bands."
    answer = reject_invented_numbers(answer, bundle, template)

    adaptation_note = ""
    if not ADAPTER_EXISTS:
        adaptation_note = " [NOT adapted — generic VLM baseline]"

    return ToolResult(
        tool="vqa",
        provenance=provenance,
        answer=answer + adaptation_note,
        measurements=measurements,
        extra={
            "adapter_present": ADAPTER_EXISTS,
            "vlm_model_id": VLM_MODEL_ID,
        },
    )
