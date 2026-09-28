"""
registry.py — Tool registry with whitelisted parameters.
The agentic controller may only pass parameters declared here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ParamSpec:
    name: str
    type: str                       # 'float' | 'int' | 'str' | 'bool'
    default: Any
    min: Any = None
    max: Any = None
    choices: list[str] | None = None
    description: str = ""


@dataclass
class ToolSpec:
    name: str
    version: str
    provenance: str                  # 'real-model' | 'classical-cv-baseline' | 'rule-based'
    accepted_input_configs: list[str]  # 'single' | 'bitemporal' | 'optical_sar'
    permitted_params: list[ParamSpec] = field(default_factory=list)
    description: str = ""


REGISTRY: dict[str, ToolSpec] = {
    "vqa": ToolSpec(
        name="vqa",
        version="1.0.0",
        provenance="real-model",
        accepted_input_configs=["single", "bitemporal", "optical_sar"],
        permitted_params=[
            ParamSpec("max_new_tokens", "int", 256, 32, 512,
                      description="Max tokens for VLM answer."),
            ParamSpec("evidence_only", "bool", True,
                      description="Reject numbers not in evidence bundle."),
        ],
        description="Vision-language model VQA with evidence-first grounding.",
    ),
    "caption": ToolSpec(
        name="caption",
        version="1.0.0",
        provenance="classical-cv-baseline",
        accepted_input_configs=["single"],
        permitted_params=[
            ParamSpec("max_classes", "int", 5, 1, 10,
                      description="Max land-cover classes to include."),
        ],
        description="Scene description from measured scene graph.",
    ),
    "grounding": ToolSpec(
        name="grounding",
        version="1.0.0",
        provenance="classical-cv-baseline",
        accepted_input_configs=["single", "optical_sar"],
        permitted_params=[
            ParamSpec("target_class", "str", "water",
                      choices=["water", "built-up", "vegetation", "bare-soil", "agriculture"],
                      description="Land-cover class to highlight."),
            ParamSpec("smoothing", "int", 3, 1, 9,
                      description="Morphological kernel size for mask cleaning."),
            ParamSpec("gsd_m", "float", None,
                      description="Ground sampling distance in metres (required if image has no CRS)."),
        ],
        description="Text-guided region grounding: mask → polygons → area.",
    ),
    "change": ToolSpec(
        name="change",
        version="1.0.0",
        provenance="classical-cv-baseline",
        accepted_input_configs=["bitemporal"],
        permitted_params=[
            ParamSpec("threshold_method", "str", "otsu",
                      choices=["otsu", "mean", "percentile95"],
                      description="Change-magnitude thresholding method."),
            ParamSpec("smoothing", "int", 3, 1, 9,
                      description="Morphological kernel size."),
        ],
        description="Bi-temporal change map, transition matrix, and change-VQA.",
    ),
    "fusion": ToolSpec(
        name="fusion",
        version="1.0.0",
        provenance="rule-based",
        accepted_input_configs=["optical_sar"],
        permitted_params=[
            ParamSpec("agreement_threshold", "float", 0.5, 0.0, 1.0,
                      description="Minimum agreement score to call a class 'confirmed'."),
        ],
        description="Optical–SAR fusion: confirmed/optical-only/SAR-only agreement map.",
    ),
    "flood_extent": ToolSpec(
        name="flood_extent",
        version="1.0.0",
        provenance="classical-cv-baseline",
        accepted_input_configs=["bitemporal", "optical_sar"],
        permitted_params=[
            ParamSpec("water_index", "str", "auto",
                      choices=["auto", "ndwi", "mndwi", "sar_threshold"],
                      description="Which index to use for water detection."),
        ],
        description="Flood extent: water mask before/after, flooded area in km².",
    ),
    "land_disturbance": ToolSpec(
        name="land_disturbance",
        version="1.0.0",
        provenance="classical-cv-baseline",
        accepted_input_configs=["bitemporal"],
        permitted_params=[
            ParamSpec("ndvi_drop_threshold", "float", 0.15, 0.05, 0.5,
                      description="Minimum NDVI drop to flag as candidate."),
        ],
        description="Land-disturbance candidates: NDVI drop + bare-soil gain (human review required).",
    ),
}


def get_tool(name: str) -> ToolSpec | None:
    return REGISTRY.get(name)


def validate_params(tool_name: str, params: dict) -> tuple[dict, list[str]]:
    """
    Validate and coerce params for a tool. Returns (cleaned_params, warnings).
    Drops any param not in the whitelist.
    """
    spec = REGISTRY.get(tool_name)
    if spec is None:
        return {}, [f"Unknown tool '{tool_name}'."]

    allowed = {p.name: p for p in spec.permitted_params}
    cleaned: dict = {}
    warnings: list[str] = []

    # Apply whitelisted params
    for ps in spec.permitted_params:
        if ps.name in params:
            v = params[ps.name]
            # Type coercion
            try:
                if ps.type == "int":
                    v = int(v)
                elif ps.type == "float":
                    v = float(v) if v is not None else None
                elif ps.type == "bool":
                    v = bool(v)
                elif ps.type == "str":
                    v = str(v)
            except (ValueError, TypeError):
                warnings.append(f"Param '{ps.name}': could not coerce to {ps.type}. Using default.")
                v = ps.default
            # Range check
            if v is not None and ps.min is not None and v < ps.min:
                warnings.append(f"Param '{ps.name}' clamped from {v} to min {ps.min}.")
                v = ps.min
            if v is not None and ps.max is not None and v > ps.max:
                warnings.append(f"Param '{ps.name}' clamped from {v} to max {ps.max}.")
                v = ps.max
            # Choices check
            if ps.choices and v not in ps.choices:
                warnings.append(f"Param '{ps.name}' = '{v}' not in {ps.choices}. Using default.")
                v = ps.default
            cleaned[ps.name] = v
        elif ps.default is not None:
            cleaned[ps.name] = ps.default

    # Warn about unknown params
    for k in params:
        if k not in allowed:
            warnings.append(f"Param '{k}' is not whitelisted for tool '{tool_name}' — ignored.")

    return cleaned, warnings
