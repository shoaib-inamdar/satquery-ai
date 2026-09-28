"""
schemas.py — Pydantic models for all API request/response contracts.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
import datetime


# ── Provenance literals ───────────────────────────────────────────────────────
Provenance = Literal["real-model", "classical-cv-baseline", "rule-based", "not-configured"]


# ── Ingestion / compatibility ─────────────────────────────────────────────────
class CheckItem(BaseModel):
    key: str
    status: Literal["pass", "warn", "fail"]
    message: str


class ImageMeta(BaseModel):
    filename: str
    modality: Literal["optical", "sar", "rgb", "unknown"]
    band_count: int
    dtype: str
    crs: Optional[str] = None
    transform: Optional[List[float]] = None   # 6-element Affine
    resolution_m: Optional[float] = None
    bounds: Optional[List[float]] = None      # left, bottom, right, top
    nodata_fraction: float = 0.0
    date_hint: Optional[str] = None


class CompatibilityReport(BaseModel):
    input_config: Literal["single", "bitemporal", "optical_sar"]
    checks: List[CheckItem]
    overall: Literal["pass", "warn", "fail"]
    images: List[ImageMeta]


class UploadResponse(BaseModel):
    session_id: str
    compatibility: CompatibilityReport


# ── Router ────────────────────────────────────────────────────────────────────
class RoutingDecision(BaseModel):
    task: str
    tool_chain: List[str]
    params: Dict[str, Any] = {}
    confidence: float
    router_name: Literal["laya", "rule-based"]
    alternatives: List[str] = []
    refusal_reason: Optional[str] = None


# ── Evidence ──────────────────────────────────────────────────────────────────
class EvidenceItem(BaseModel):
    evidence_id: str
    type: str
    value: Any
    method: str
    provenance: Provenance
    confidence: float


# ── Confidence breakdown ──────────────────────────────────────────────────────
class ConfidenceBreakdown(BaseModel):
    overall: float
    input_quality: float
    separation_quality: float
    coreg_residual: Optional[float] = None
    cross_modal_agreement: Optional[float] = None
    router_confidence: float
    formula: str = "weighted_geometric_mean"


# ── Trace ─────────────────────────────────────────────────────────────────────
class TraceStage(BaseModel):
    stage: str
    started_at: str
    ended_at: Optional[str] = None
    duration_ms: Optional[float] = None
    details: Dict[str, Any] = {}


class ExecutionTrace(BaseModel):
    session_id: str
    task: str
    router: str
    tool_chain: List[str]
    params: Dict[str, Any]
    warnings: List[str] = []
    stages: List[TraceStage] = []


# ── Tool result ───────────────────────────────────────────────────────────────
class ToolResult(BaseModel):
    tool: str
    provenance: Provenance
    answer: str
    measurements: Dict[str, Any] = {}
    overlay_url: Optional[str] = None
    geojson: Optional[Dict] = None
    extra: Dict[str, Any] = {}


# ── Full analysis response ────────────────────────────────────────────────────
class AnalyzeResponse(BaseModel):
    session_id: str
    query: str
    routing: RoutingDecision
    tool_results: List[ToolResult]
    evidence: List[EvidenceItem]
    confidence: ConfidenceBreakdown
    trace: ExecutionTrace
    adapter_present: bool
    rs_adaptation_label: str


# ── Chat ──────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    cited_evidence_ids: List[str] = []


# ── Analyze request ───────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    session_id: str
    query: str
    params: Dict[str, Any] = Field(default_factory=dict)
