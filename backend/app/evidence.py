"""
evidence.py — Evidence bundle: structured, ID'd results with provenance.
"""
from __future__ import annotations
import uuid
from .schemas import EvidenceItem, Provenance


def make_evidence_id() -> str:
    return "ev-" + str(uuid.uuid4())[:8]


def evidence_item(
    type: str,
    value,
    method: str,
    provenance: Provenance,
    confidence: float,
    evidence_id: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id or make_evidence_id(),
        type=type,
        value=value,
        method=method,
        provenance=provenance,
        confidence=confidence,
    )


class EvidenceBundle:
    """
    Accumulates evidence items for a session.
    The chat module is constrained to answers from this bundle only.
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._items: list[EvidenceItem] = []
        self._number_set: set[float] = set()

    def add(self, item: EvidenceItem) -> EvidenceItem:
        self._items.append(item)
        # Track numeric values for the anti-fabrication check
        v = item.value
        if isinstance(v, (int, float)):
            self._number_set.add(round(float(v), 4))
        elif isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, (int, float)):
                    self._number_set.add(round(float(vv), 4))
        return item

    def add_measurement(
        self, name: str, value, method: str,
        provenance: Provenance = "classical-cv-baseline",
        confidence: float = 0.8,
    ) -> EvidenceItem:
        ei = evidence_item(
            type=f"measurement:{name}",
            value=value,
            method=method,
            provenance=provenance,
            confidence=confidence,
        )
        return self.add(ei)

    def items(self) -> list[EvidenceItem]:
        return list(self._items)

    def numbers(self) -> set[float]:
        return set(self._number_set)

    def contains_number(self, n: float, tol: float = 0.01) -> bool:
        """True if the number n (within tolerance) appears in the bundle."""
        for v in self._number_set:
            if abs(v - round(float(n), 4)) <= tol:
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "items": [i.model_dump() for i in self._items],
        }
