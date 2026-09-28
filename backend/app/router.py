"""
router.py — Agentic task router.
Implements TaskRouter interface, LayaRouter (with Laya typed-decision engine),
and RuleRouter fallback.
"""
from __future__ import annotations
import re
from typing import Protocol
from .schemas import RoutingDecision
from .config import LAYA_CONFIDENCE_THRESHOLD


# ── Keyword taxonomy for RuleRouter ──────────────────────────────────────────
# Each entry is (pattern, is_regex). Literal strings use re.escape internally.
TASK_KEYWORDS: dict[str, list[tuple[str, bool]]] = {
    "grounding": [
        ("highlight", False), ("locate", False), ("show", False), ("mark", False),
        ("where is", False), ("find", False), ("identify", False),
        ("segment", False), ("outline", False), ("delineate", False),
        ("draw", False), ("map the", False), ("region", False),
    ],
    "change": [
        ("changed", False), ("change", False), ("before and after", False),
        ("temporal", False), ("increase", False), ("decrease", False),
        ("grew", False), ("reduced", False), ("loss", False), ("gain", False),
        ("different", False), ("compare", False), ("evolution", False),
        (r"has.*increased", True), (r"has.*decreased", True), ("remained unchanged", False),
    ],
    "fusion": [
        (r"optical.*sar", True), (r"sar.*optical", True), (r"radar.*optical", True),
        ("fuse", False), ("fusion", False), ("cross-modal", False),
        ("multimodal", False), ("both images", False), ("combine", False),
    ],
    "flood_extent": [
        ("flood", False), ("inundation", False), ("submerged", False),
        ("flooded area", False), ("water extent", False),
        ("flood map", False), ("extent of flood", False),
    ],
    "land_disturbance": [
        ("landslide", False), ("disturbance", False), ("land disturbance", False),
        ("clearance", False), ("deforestation", False),
        ("bare soil", False), ("erosion", False),
    ],
    "caption": [
        ("describe", False), ("description", False), ("explain", False),
        ("summarise", False), ("summarize", False), ("scene description", False),
        ("what is in", False), ("scene", False), ("overview", False), ("general", False),
    ],
    "vqa": [
        ("how many", False), ("what is", False), ("is there", False),
        ("are there", False), ("what type", False), ("what kind", False),
        ("what percentage", False), ("count", False), ("question", False),
    ],
}

# Required input configs per task
TASK_INPUT_CONFIG: dict[str, list[str]] = {
    "vqa":             ["single", "bitemporal", "optical_sar"],
    "caption":         ["single"],
    "grounding":       ["single", "optical_sar"],
    "change":          ["bitemporal"],
    "fusion":          ["optical_sar"],
    "flood_extent":    ["bitemporal", "optical_sar"],
    "land_disturbance": ["bitemporal"],
}


# ── Protocol ──────────────────────────────────────────────────────────────────
class TaskRouter(Protocol):
    def route(self, query: str, input_config: str) -> RoutingDecision: ...


# ── Rule-based router ─────────────────────────────────────────────────────────
class RuleRouter:
    name = "rule-based"

    def route(self, query: str, input_config: str) -> RoutingDecision:
        q = query.lower()
        scores: dict[str, float] = {t: 0.0 for t in TASK_KEYWORDS}

        for task, kw_list in TASK_KEYWORDS.items():
            for kw, is_regex in kw_list:
                pattern = kw if is_regex else re.escape(kw)
                try:
                    if re.search(pattern, q):
                        scores[task] += 1.0
                except re.PatternError:
                    pass  # skip malformed patterns

        # Normalise
        total = sum(scores.values()) or 1.0
        for t in scores:
            scores[t] /= total

        # Filter by compatible input config
        compatible = {
            t: s for t, s in scores.items()
            if input_config in TASK_INPUT_CONFIG.get(t, [])
        }
        if not compatible:
            # fallback: vqa for single, change for bitemporal, fusion for optical_sar
            fallback = {"single": "vqa", "bitemporal": "change", "optical_sar": "fusion"}.get(
                input_config, "vqa"
            )
            compatible = {fallback: 0.3}

        best_task = max(compatible, key=lambda t: compatible[t])
        best_score = compatible[best_task]

        alternatives = sorted(
            [t for t in compatible if t != best_task],
            key=lambda t: -compatible[t],
        )[:2]

        tool_chain = _default_tool_chain(best_task, input_config)

        # Refusal check
        refusal = _check_refusal(best_task, input_config)

        return RoutingDecision(
            task=best_task,
            tool_chain=tool_chain,
            params={},
            confidence=round(best_score, 3),
            router_name="rule-based",
            alternatives=alternatives,
            refusal_reason=refusal,
        )


def _default_tool_chain(task: str, input_config: str) -> list[str]:
    chains = {
        "vqa":             ["vqa"],
        "caption":         ["caption"],
        "grounding":       ["grounding"],
        "change":          ["change"],
        "fusion":          ["fusion"],
        "flood_extent":    ["flood_extent"],
        "land_disturbance": ["land_disturbance"],
    }
    return chains.get(task, ["vqa"])


def _check_refusal(task: str, input_config: str) -> str | None:
    required = TASK_INPUT_CONFIG.get(task, [])
    if input_config not in required:
        needed = " or ".join(required)
        return (
            f"Task '{task}' requires {needed} input, "
            f"but '{input_config}' was provided. "
            f"Please upload the appropriate image(s)."
        )
    return None


# ── Laya router ───────────────────────────────────────────────────────────────
class LayaRouter:
    """
    Wraps the open-source Laya typed-decision engine.
    If Laya is not installed or errors, raises ImportError / RuntimeError.
    The orchestrator catches these and falls back to RuleRouter.
    """
    name = "laya"
    _engine = None

    def __init__(self):
        try:
            import laya  # noqa: F401
            self._laya = laya
        except ImportError:
            raise ImportError("Laya is not installed.")

    def route(self, query: str, input_config: str) -> RoutingDecision:
        try:
            # Attempt to use Laya's classification API.
            # We read its API from the installed package; if it changes, we fall back.
            result = self._laya.classify(
                text=query,
                labels=list(TASK_KEYWORDS.keys()),
            )
            task = result.label
            conf = float(result.score)
        except Exception as exc:
            raise RuntimeError(f"Laya inference failed: {exc}") from exc

        if conf < LAYA_CONFIDENCE_THRESHOLD:
            raise RuntimeError(
                f"Laya confidence {conf:.2f} below threshold {LAYA_CONFIDENCE_THRESHOLD}."
            )

        refusal = _check_refusal(task, input_config)
        tool_chain = _default_tool_chain(task, input_config)
        return RoutingDecision(
            task=task,
            tool_chain=tool_chain,
            params={},
            confidence=round(conf, 3),
            router_name="laya",
            alternatives=[],
            refusal_reason=refusal,
        )


# ── Composite router: tries Laya, falls back to Rule ─────────────────────────
class AgenticRouter:
    def __init__(self):
        self._laya: LayaRouter | None = None
        self._rule = RuleRouter()
        try:
            self._laya = LayaRouter()
        except ImportError:
            pass

    @property
    def router_in_use(self) -> str:
        return "laya" if self._laya else "rule-based"

    def route(self, query: str, input_config: str) -> RoutingDecision:
        if self._laya:
            try:
                return self._laya.route(query, input_config)
            except RuntimeError:
                pass  # fall through to rule-based
        return self._rule.route(query, input_config)


# Singleton
_router_instance: AgenticRouter | None = None


def get_router() -> AgenticRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = AgenticRouter()
    return _router_instance
