// API client for SatQuery AI backend
const BASE_URL = '/api';

export interface CompatibilityReport {
  input_config: string;
  checks: { key: string; status: string; message: string }[];
  overall: string;
  images: any[];
}

export interface UploadResponse {
  session_id: string;
  compatibility: CompatibilityReport;
}

export interface AnalyzeResponse {
  session_id: string;
  query: string;
  routing: {
    task: string;
    tool_chain: string[];
    params: Record<string, any>;
    confidence: number;
    router_name: string;
    alternatives: string[];
    refusal_reason?: string;
  };
  tool_results: {
    tool: string;
    provenance: string;
    answer: string;
    measurements: Record<string, any>;
    overlay_url?: string;
    geojson?: any;
    extra?: Record<string, any>;
  }[];
  evidence: {
    evidence_id: string;
    type: string;
    value: any;
    method: string;
    provenance: string;
    confidence: number;
  }[];
  confidence: {
    overall: number;
    input_quality: number;
    separation_quality: number;
    coreg_residual?: number;
    cross_modal_agreement?: number;
    router_confidence: number;
    formula: string;
  };
  trace: {
    session_id: string;
    task: string;
    router: string;
    tool_chain: string[];
    params: Record<string, any>;
    warnings: string[];
    stages: { stage: string; started_at: string; ended_at?: string; duration_ms?: number; details: any }[];
  };
  adapter_present: boolean;
  rs_adaptation_label: string;
}

export async function uploadImages(files: File[], input_config?: string): Promise<UploadResponse> {
  const form = new FormData();
  files.forEach(f => form.append('files', f));
  const url = input_config ? `${BASE_URL}/upload?input_config=${input_config}` : `${BASE_URL}/upload`;
  const res = await fetch(url, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Upload failed');
  }
  return res.json();
}

export async function analyze(session_id: string, query: string, params: Record<string, any> = {}): Promise<AnalyzeResponse> {
  const res = await fetch(`${BASE_URL}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id, query, params }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Analysis failed');
  }
  return res.json();
}

export function streamAnalyze(session_id: string, query: string, onStage: (stage: string, status: string) => void, onDone: (result: AnalyzeResponse) => void, onError: (msg: string) => void) {
  const url = `${BASE_URL}/analyze/stream?session_id=${encodeURIComponent(session_id)}&query=${encodeURIComponent(query)}`;
  const es = new EventSource(url);
  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.stage === 'done') {
        es.close();
        onDone(data.result);
      } else if (data.stage === 'error') {
        es.close();
        onError(data.message);
      } else {
        onStage(data.stage, data.status);
      }
    } catch {}
  };
  es.onerror = () => {
    es.close();
    onError('Stream connection error');
  };
  return () => es.close();
}

export async function chat(session_id: string, message: string) {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id, message }),
  });
  return res.json();
}

export async function getStatus() {
  const res = await fetch(`${BASE_URL}/status`);
  return res.json();
}

export async function getSamples() {
  const res = await fetch(`${BASE_URL}/samples`);
  return res.json();
}

export async function getBenchmarks() {
  const res = await fetch(`${BASE_URL}/benchmarks`);
  return res.json();
}

export function reportUrl(session_id: string, format: 'pdf' | 'json' | 'geojson.zip') {
  return `${BASE_URL}/report/${session_id}.${format}`;
}
