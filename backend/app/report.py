"""
report.py — Downloadable PDF, JSON, and GeoJSON zip reports.
Uses reportlab for PDF generation.
"""
from __future__ import annotations
import io
import json
import zipfile
import datetime
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

from .schemas import AnalyzeResponse, EvidenceItem


def _safe_str(v: Any, max_len: int = 300) -> str:
    s = str(v)
    return s[:max_len] + "…" if len(s) > max_len else s


def build_pdf(response: AnalyzeResponse, query: str) -> bytes:
    """Generate a PDF report from the analysis response."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    def h1(text):
        story.append(Paragraph(text, styles["h1"]))

    def h2(text):
        story.append(Paragraph(text, styles["h2"]))

    def body(text):
        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 0.3*cm))

    def spacer():
        story.append(Spacer(1, 0.5*cm))

    # Header
    h1("SatQuery AI — Analysis Report")
    body(f"Generated: {datetime.datetime.utcnow().isoformat()}Z")
    body(f"Session ID: {response.session_id}")
    spacer()

    # Query
    h2("1. Query")
    body(f"<b>{query}</b>")
    spacer()

    # RS adaptation
    h2("2. Remote-Sensing Adaptation Status")
    badge = response.rs_adaptation_label
    colour = "#22aa22" if "adapted" in badge.lower() and "NOT" not in badge else "#cc7700"
    body(f'<font color="{colour}"><b>{badge}</b></font>')
    spacer()

    # Routing
    h2("3. Routing Decision")
    r = response.routing
    body(f"Task: <b>{r.task}</b> | Router: {r.router_name} | Confidence: {r.confidence:.3f}")
    body(f"Tool chain: {' → '.join(r.tool_chain)}")
    if r.refusal_reason:
        body(f"⚠️ Refusal: {r.refusal_reason}")
    spacer()

    # Tool results
    h2("4. Tool Results")
    for tr in response.tool_results:
        body(f"<b>Tool:</b> {tr.tool} | <b>Provenance:</b> {tr.provenance}")
        body(f"<b>Answer:</b> {_safe_str(tr.answer)}")
        if tr.measurements:
            meas_rows = [["Measurement", "Value"]]
            for k, v in list(tr.measurements.items())[:15]:
                meas_rows.append([str(k), _safe_str(v, 60)])
            t = Table(meas_rows, colWidths=[8*cm, 8*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTSIZE",   (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            story.append(t)
        spacer()

    # Confidence
    h2("5. Confidence Breakdown")
    c = response.confidence
    body(f"Overall: <b>{c.overall:.3f}</b> (NOT a calibrated probability)")
    body(f"Input quality: {c.input_quality:.3f}")
    body(f"Separation quality: {c.separation_quality:.3f}")
    body(f"Router confidence: {c.router_confidence:.3f}")
    if c.coreg_residual is not None:
        body(f"Co-registration residual: {c.coreg_residual:.3f}")
    if c.cross_modal_agreement is not None:
        body(f"Cross-modal agreement: {c.cross_modal_agreement:.3f}")
    body(f"Formula: {c.formula}")
    spacer()

    # Evidence
    h2("6. Evidence Items")
    ev_rows = [["ID", "Type", "Value", "Method", "Provenance", "Conf."]]
    for ei in response.evidence[:20]:
        ev_rows.append([
            ei.evidence_id, _safe_str(ei.type, 25), _safe_str(ei.value, 30),
            _safe_str(ei.method, 25), ei.provenance, f"{ei.confidence:.3f}",
        ])
    if len(response.evidence) > 20:
        ev_rows.append([f"… +{len(response.evidence)-20} more", "", "", "", "", ""])
    t = Table(ev_rows, colWidths=[2*cm, 3*cm, 3.5*cm, 2.5*cm, 3*cm, 1.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.black),
        ("FONTSIZE",   (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightblue]),
    ]))
    story.append(t)
    spacer()

    # Trace
    h2("7. Execution Trace")
    tr = response.trace
    for stage in tr.stages:
        dur = f"{stage.duration_ms:.1f}ms" if stage.duration_ms else ""
        body(f"• <b>{stage.stage}</b> {dur} — {json.dumps(stage.details)[:120]}")
    if tr.warnings:
        body("Warnings: " + "; ".join(tr.warnings))
    spacer()

    # Limitations
    h2("8. Limitations")
    body("• Classical indices are physics-based; turbid water and bare soil may be confused in RGB-only input.")
    body("• SAR-only detections are reported as ambiguous candidates.")
    body("• Change detection depends on co-registration quality; residual is reported.")
    body("• Confidence is a documented heuristic, not a calibrated probability.")
    body("• A small VLM may be weak on complex questions; evidence-first check limits fabrication.")

    doc.build(story)
    return buf.getvalue()


def build_json(response: AnalyzeResponse) -> bytes:
    return response.model_dump_json(indent=2).encode()


def build_geojson_zip(response: AnalyzeResponse) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, tr in enumerate(response.tool_results):
            if tr.geojson:
                zf.writestr(
                    f"{response.session_id}_{tr.tool}_{i}.geojson",
                    json.dumps(tr.geojson, indent=2),
                )
    return buf.getvalue()
