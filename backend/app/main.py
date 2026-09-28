"""
main.py — FastAPI application: uploads, analysis, SSE streaming, chat, reports, benchmarks.
Model inference runs in a thread pool (not directly in async handlers).
"""
from __future__ import annotations
import asyncio
import json
import uuid
import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import (
    UPLOAD_DIR, ADAPTER_EXISTS, VLM_MODEL_ID,
    ADAPTER_PATH, RESULTS_DIR, SAMPLES_DIR,
)
from .ingestion import load_image, check_compatibility
from .preprocessing import make_rgb_preview
from .router import get_router
from .registry import validate_params, get_tool
from .evidence import EvidenceBundle
from .trace import Tracer
from .confidence import (
    compute_confidence, score_input_quality,
    score_otsu_separation, score_coreg_residual, score_cross_modal_agreement,
)
from .schemas import (
    AnalyzeRequest, AnalyzeResponse, ChatRequest, ChatResponse,
    UploadResponse, CompatibilityReport, ToolResult,
)
from .chat import answer_from_bundle
from .report import build_pdf, build_json, build_geojson_zip

# ── Thread pool for blocking inference ───────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="SatQuery AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ───────────────────────────────────────────────────
_sessions: dict[str, dict] = {}   # session_id → {metas, datas, compat, responses, bundles}


def _session(sid: str) -> dict:
    if sid not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{sid}' not found.")
    return _sessions[sid]


# ── Health / status ───────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat() + "Z"}


@app.get("/api/status")
async def status():
    router = get_router()
    return {
        "router_in_use": router.router_in_use,
        "vlm_model_id": VLM_MODEL_ID,
        "adapter_present": ADAPTER_EXISTS,
        "rs_adaptation_label": (
            "RS-adapted (BigEarthNet LoRA)" if ADAPTER_EXISTS
            else "NOT adapted — generic VLM baseline"
        ),
        "version": "1.0.0",
    }


# ── Samples ───────────────────────────────────────────────────────────────────

@app.get("/api/samples")
async def list_samples():
    samples = []
    for folder in ["synthetic", "real"]:
        p = SAMPLES_DIR / folder
        if p.exists():
            for f in p.iterdir():
                if f.suffix.lower() in (".tif", ".tiff", ".geotiff", ".png", ".jpg"):
                    samples.append({
                        "name": f.name,
                        "label": "SYNTHETIC" if folder == "synthetic" else "REAL",
                        "path": str(f),
                    })
    return {"samples": samples}


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/api/upload", response_model=UploadResponse)
async def upload(
    files: list[UploadFile] = File(...),
    input_config: Optional[str] = Query(None),
):
    if not 1 <= len(files) <= 2:
        raise HTTPException(400, "Upload 1 or 2 images.")

    session_id = str(uuid.uuid4())[:8]
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    datas, metas = [], []
    for f in files:
        dest = session_dir / f.filename
        content = await f.read()
        dest.write_bytes(content)
        try:
            data, meta = load_image(dest)
        except Exception as e:
            raise HTTPException(400, f"Could not read '{f.filename}': {e}")
        datas.append(data)
        metas.append(meta)

    compat = check_compatibility(metas, declared_config=input_config)

    _sessions[session_id] = {
        "datas": datas,
        "metas": metas,
        "compat": compat,
        "paths": [session_dir / f.filename for f in files],
        "responses": {},
        "bundles": {},
    }

    return UploadResponse(session_id=session_id, compatibility=compat)


# ── Analyze (blocking, returns full result) ───────────────────────────────────

def _run_analysis(session_id: str, query: str, params: dict) -> AnalyzeResponse:
    """Blocking analysis — runs in a thread pool."""
    sess = _sessions[session_id]
    datas: list[np.ndarray] = sess["datas"]
    metas: list[dict] = sess["metas"]
    compat: CompatibilityReport = sess["compat"]

    bundle = EvidenceBundle(session_id)
    router = get_router()
    routing = router.route(query, compat.input_config)

    tracer = Tracer(
        session_id=session_id,
        task=routing.task,
        router=routing.router_name,
        tool_chain=routing.tool_chain,
        params=routing.params,
    )

    # Refusal
    if routing.refusal_reason:
        return AnalyzeResponse(
            session_id=session_id,
            query=query,
            routing=routing,
            tool_results=[ToolResult(
                tool="refusal",
                provenance="rule-based",
                answer=routing.refusal_reason,
            )],
            evidence=[],
            confidence=compute_confidence(0.5, 0.5, routing.confidence),
            trace=tracer.build(),
            adapter_present=ADAPTER_EXISTS,
            rs_adaptation_label=(
                "RS-adapted (BigEarthNet LoRA)" if ADAPTER_EXISTS
                else "NOT adapted — generic VLM baseline"
            ),
        )

    # Validate params
    tracer.start_stage("plan")
    tool_name = routing.tool_chain[0] if routing.tool_chain else "vqa"
    cleaned_params, param_warnings = validate_params(tool_name, {**routing.params, **params})
    for w in param_warnings:
        tracer.warn(w)
    tracer.end_stage(tool=tool_name, params=cleaned_params)

    # Preprocess
    tracer.start_stage("preprocess")
    from .preprocessing import make_rgb_preview
    rgb = make_rgb_preview(datas[0], metas[0])
    tracer.end_stage()

    # Execute tool
    tracer.start_stage(f"execute[{tool_name}]")
    tool_results: list[ToolResult] = []

    try:
        if tool_name == "vqa":
            from .tools.vqa import run_vqa
            tr = run_vqa(datas[0], metas[0], query, bundle, cleaned_params, image_rgb=rgb)
            tool_results.append(tr)

        elif tool_name == "caption":
            from .tools.caption import run_caption
            tr = run_caption(datas[0], metas[0], bundle, cleaned_params, image_rgb=rgb)
            tool_results.append(tr)

        elif tool_name == "grounding":
            from .tools.grounding import run_grounding
            tr = run_grounding(datas[0], metas[0], query, bundle, cleaned_params)
            tool_results.append(tr)

        elif tool_name == "change":
            if len(datas) < 2:
                tool_results.append(ToolResult(
                    tool="change", provenance="rule-based",
                    answer="Change analysis requires two images. Please upload a bi-temporal pair.",
                ))
            else:
                from .tools.change import run_change
                tr = run_change(datas[0], metas[0], datas[1], metas[1], query, bundle, cleaned_params)
                tool_results.append(tr)

        elif tool_name == "fusion":
            if len(datas) < 2:
                tool_results.append(ToolResult(
                    tool="fusion", provenance="rule-based",
                    answer="Fusion requires an optical+SAR image pair. Please upload both.",
                ))
            else:
                from .tools.fusion import run_fusion
                # Determine which is optical, which is SAR
                if metas[0]["modality"] == "sar":
                    opt_data, opt_meta = datas[1], metas[1]
                    sar_data, sar_meta = datas[0], metas[0]
                else:
                    opt_data, opt_meta = datas[0], metas[0]
                    sar_data, sar_meta = datas[1], metas[1]
                tr = run_fusion(opt_data, opt_meta, sar_data, sar_meta, query, bundle, cleaned_params)
                tool_results.append(tr)

        elif tool_name == "flood_extent":
            if len(datas) < 2:
                tool_results.append(ToolResult(
                    tool="flood_extent", provenance="rule-based",
                    answer="Flood extent preset requires two images (before/after or optical+SAR).",
                ))
            else:
                from .tools.presets import run_flood_extent
                tr = run_flood_extent(datas[0], metas[0], datas[1], metas[1], bundle, cleaned_params)
                tool_results.append(tr)

        elif tool_name == "land_disturbance":
            if len(datas) < 2:
                tool_results.append(ToolResult(
                    tool="land_disturbance", provenance="rule-based",
                    answer="Land-disturbance preset requires a bi-temporal image pair.",
                ))
            else:
                from .tools.presets import run_land_disturbance
                tr = run_land_disturbance(datas[0], metas[0], datas[1], metas[1], bundle, cleaned_params)
                tool_results.append(tr)

        else:
            tool_results.append(ToolResult(
                tool=tool_name, provenance="not-configured",
                answer=f"Tool '{tool_name}' is not yet implemented.",
            ))

    except Exception as exc:
        import traceback
        tracer.warn(f"Tool execution error: {exc}")
        tool_results.append(ToolResult(
            tool=tool_name, provenance="not-configured",
            answer=f"Tool execution failed: {exc}",
        ))

    tracer.end_stage()

    # Confidence
    tracer.start_stage("confidence")
    nodata_frac = metas[0].get("nodata_fraction", 0.0)
    band_avail = min(1.0, metas[0]["band_count"] / 4.0)
    iq = score_input_quality(band_avail, nodata_frac, True)
    sep = 0.6  # default; grounding/change tools set higher when computed
    for tr in tool_results:
        if "separation_score" in tr.measurements:
            sep = float(tr.measurements["separation_score"])

    coreg_res = None
    agree = None
    if tool_name == "change" and tool_results:
        m = tool_results[0].measurements
        coreg_res = m.get("extra", {}).get("coreg_confidence", 0.7) if hasattr(m, "get") else 0.7
    if tool_name == "fusion" and tool_results:
        agree = tool_results[0].extra.get("cross_modal_agreement", 0.6)

    confidence = compute_confidence(
        input_quality=iq,
        separation_quality=sep,
        router_confidence=routing.confidence,
        coreg_residual=coreg_res,
        cross_modal_agreement=agree,
    )
    tracer.end_stage()

    # Report stage marker
    tracer.start_stage("integrate")
    tracer.end_stage()

    response = AnalyzeResponse(
        session_id=session_id,
        query=query,
        routing=routing,
        tool_results=tool_results,
        evidence=bundle.items(),
        confidence=confidence,
        trace=tracer.build(),
        adapter_present=ADAPTER_EXISTS,
        rs_adaptation_label=(
            "RS-adapted (BigEarthNet LoRA)" if ADAPTER_EXISTS
            else "NOT adapted — generic VLM baseline"
        ),
    )

    sess["responses"][query] = response
    sess["bundles"][query] = bundle
    return response


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor, _run_analysis, req.session_id, req.query, req.params
    )
    return result


# ── SSE streaming analyze ─────────────────────────────────────────────────────

@app.get("/api/analyze/stream")
async def analyze_stream(
    session_id: str = Query(...),
    query: str = Query(...),
):
    async def event_generator():
        stages = ["ingest", "compatibility", "route", "plan", "preprocess",
                  "execute", "integrate", "confidence", "report"]
        for stage in stages:
            yield f"data: {json.dumps({'stage': stage, 'status': 'running'})}\n\n"
            await asyncio.sleep(0.1)

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor, _run_analysis, session_id, query, {}
            )
            yield f"data: {json.dumps({'stage': 'done', 'result': result.model_dump()})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'stage': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    sess = _session(req.session_id)
    # Use the last run bundle
    bundles = sess.get("bundles", {})
    if not bundles:
        return ChatResponse(
            reply="No analysis has been run for this session yet.",
            cited_evidence_ids=[],
        )
    last_bundle: EvidenceBundle = list(bundles.values())[-1]
    return answer_from_bundle(req.message, last_bundle)


# ── Reports ───────────────────────────────────────────────────────────────────

@app.get("/api/report/{session_id}.pdf")
async def report_pdf(session_id: str, query: str = Query("")):
    sess = _session(session_id)
    responses = sess.get("responses", {})
    if not responses:
        raise HTTPException(404, "No analysis found for this session.")
    response = list(responses.values())[-1]

    loop = asyncio.get_event_loop()
    pdf_bytes = await loop.run_in_executor(_executor, build_pdf, response, query or response.query)
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename={session_id}_report.pdf"})


@app.get("/api/report/{session_id}.json")
async def report_json(session_id: str):
    sess = _session(session_id)
    responses = sess.get("responses", {})
    if not responses:
        raise HTTPException(404, "No analysis found for this session.")
    response = list(responses.values())[-1]
    return Response(content=build_json(response), media_type="application/json",
                    headers={"Content-Disposition": f"attachment; filename={session_id}_report.json"})


@app.get("/api/report/{session_id}.geojson.zip")
async def report_geojson(session_id: str):
    sess = _session(session_id)
    responses = sess.get("responses", {})
    if not responses:
        raise HTTPException(404, "No analysis found for this session.")
    response = list(responses.values())[-1]
    return Response(content=build_geojson_zip(response), media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename={session_id}_geojson.zip"})


# ── Benchmarks ────────────────────────────────────────────────────────────────

@app.get("/api/benchmarks")
async def benchmarks():
    results = []
    if RESULTS_DIR.exists():
        for f in RESULTS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                results.append({"file": f.name, **data})
            except Exception:
                pass
    return {"results": results, "note": "Run eval scripts to populate results/."}


# ── App init ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    print(f"SatQuery AI backend starting.")
    print(f"  VLM: {VLM_MODEL_ID}")
    print(f"  Adapter: {'present' if ADAPTER_EXISTS else 'NOT present — generic VLM baseline'}")
    router = get_router()
    print(f"  Router: {router.router_in_use}")
