"""
chat.py — Follow-up Q&A constrained to the session evidence bundle.
Does not call the LLM to invent facts. Cites evidence IDs.
"""
from __future__ import annotations
import re
from .evidence import EvidenceBundle
from .schemas import ChatResponse


HELP_OFFER = (
    "I can only answer from the measurements already computed. "
    "To get more information, please run a new analysis."
)


def answer_from_bundle(message: str, bundle: EvidenceBundle) -> ChatResponse:
    """
    Try to answer `message` from the evidence bundle.
    Only uses values stored in the bundle — no LLM inference.
    """
    msg = message.lower()
    items = bundle.items()

    if not items:
        return ChatResponse(
            reply=f"No analysis has been run for this session yet. {HELP_OFFER}",
            cited_evidence_ids=[],
        )

    cited_ids: list[str] = []
    replies: list[str] = []

    # ── Search evidence items for relevant values ──────────────────────────────
    for item in items:
        type_lower = item.type.lower()
        value = item.value
        ev_id = item.evidence_id

        # Match question keywords to evidence types
        relevant = False
        if any(kw in msg for kw in ("water", "flood")):
            if "water" in type_lower or "flood" in type_lower or "mndwi" in type_lower or "ndwi" in type_lower:
                relevant = True
        if any(kw in msg for kw in ("vegetation", "ndvi", "plant", "forest")):
            if "vegetation" in type_lower or "ndvi" in type_lower:
                relevant = True
        if any(kw in msg for kw in ("built", "urban", "city", "building")):
            if "builtup" in type_lower or "built" in type_lower or "ndbi" in type_lower:
                relevant = True
        if any(kw in msg for kw in ("change", "changed", "increase", "decrease")):
            if "change" in type_lower or "delta" in type_lower:
                relevant = True
        if any(kw in msg for kw in ("area", "fraction", "percent", "coverage")):
            if "fraction" in type_lower or "area" in type_lower:
                relevant = True
        if any(kw in msg for kw in ("confidence", "score", "quality")):
            if "separation" in type_lower or "confidence" in type_lower:
                relevant = True

        if relevant:
            cited_ids.append(ev_id)
            if isinstance(value, (int, float)):
                replies.append(
                    f"[{ev_id}] {item.type}: {value:.4g} (method: {item.method}, "
                    f"provenance: {item.provenance}, confidence: {item.confidence:.3f})"
                )
            elif isinstance(value, dict):
                parts = ", ".join(f"{k}={v}" for k, v in list(value.items())[:5])
                replies.append(f"[{ev_id}] {item.type}: {parts}")
            else:
                replies.append(f"[{ev_id}] {item.type}: {str(value)[:200]}")

    if replies:
        reply_text = "From the computed evidence:\n" + "\n".join(replies[:8])
        if len(replies) > 8:
            reply_text += f"\n...and {len(replies)-8} more evidence items."
    else:
        reply_text = (
            f"I could not find relevant evidence in the analysis results for your question. {HELP_OFFER}"
        )

    return ChatResponse(reply=reply_text, cited_evidence_ids=cited_ids[:8])
