"""
tools/caption.py — Scene description from measured scene graph.
Optionally rephrased by the VLM under the same number check.
"""
from __future__ import annotations
import numpy as np
from ..evidence import EvidenceBundle
from ..indices import compute_all_indices
from ..schemas import ToolResult
from .vqa import measure_scene, vlm_answer, reject_invented_numbers
from ..config import ADAPTER_EXISTS, VLM_MODEL_ID


def build_scene_graph(measurements: dict, meta: dict) -> dict:
    """Build a simple scene graph from measurements."""
    classes = {}
    if "vegetation_fraction" in measurements:
        classes["vegetation"] = measurements["vegetation_fraction"]
    if "water_fraction_mndwi" in measurements:
        classes["water"] = measurements["water_fraction_mndwi"]
    elif "water_fraction_ndwi" in measurements:
        classes["water"] = measurements["water_fraction_ndwi"]
    if "builtup_fraction" in measurements:
        classes["built-up"] = measurements["builtup_fraction"]
    if "bare_soil_fraction" in measurements:
        classes["bare-soil"] = measurements["bare_soil_fraction"]
    # Sort by dominance
    return dict(sorted(classes.items(), key=lambda x: -x[1]))


def caption_from_scene_graph(scene_graph: dict, meta: dict, max_classes: int = 5) -> str:
    """Generate a deterministic scene description from the scene graph."""
    modality = meta.get("modality", "unknown")
    parts = []
    top = list(scene_graph.items())[:max_classes]
    for cls, frac in top:
        parts.append(f"{cls} ({frac*100:.1f}%)")

    if not parts:
        return f"This {modality} image contains insufficient spectral bands for land-cover estimation."

    dominant = top[0][0] if top else "unknown"
    description = (
        f"This {modality} image is dominated by {dominant}. "
        f"Land-cover composition: {', '.join(parts)}."
    )
    res = meta.get("resolution_m")
    if res:
        description += f" Image resolution: approximately {res:.0f} m/pixel."
    return description


def run_caption(
    data: np.ndarray,
    meta: dict,
    bundle: EvidenceBundle,
    params: dict,
    image_rgb: np.ndarray | None = None,
) -> ToolResult:
    max_classes = int(params.get("max_classes", 5))
    measurements = measure_scene(data, meta)
    scene_graph = build_scene_graph(measurements, meta)

    for k, v in measurements.items():
        if isinstance(v, (int, float)):
            bundle.add_measurement(
                name=k, value=v, method="spectral-indices",
                provenance="classical-cv-baseline", confidence=0.75,
            )

    base_caption = caption_from_scene_graph(scene_graph, meta, max_classes)

    # Optionally rephrase with VLM
    query = f"Write a one-paragraph remote-sensing scene description. {base_caption}"
    answer, provenance = vlm_answer(query, image_rgb, base_caption)
    answer = reject_invented_numbers(answer, bundle, base_caption)

    adaptation_note = "" if ADAPTER_EXISTS else " [NOT adapted — generic VLM baseline]"

    return ToolResult(
        tool="caption",
        provenance=provenance,
        answer=answer + adaptation_note,
        measurements={**measurements, "scene_graph": scene_graph},
        extra={"adapter_present": ADAPTER_EXISTS, "vlm_model_id": VLM_MODEL_ID},
    )
