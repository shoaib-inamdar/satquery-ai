import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Navbar from '../components/Navbar'

const STATUS = [
  { cap: 'GeoTIFF/TIFF ingestion, modality detection', status: 'Implemented', color: 'text-green-400', note: 'rasterio' },
  { cap: 'Compatibility check (CRS, overlap, bands)', status: 'Implemented', color: 'text-green-400', note: 'automatic' },
  { cap: 'Spectral indices (NDVI, NDWI, NDBI, BSI…)', status: 'Implemented', color: 'text-green-400', note: 'physics-based' },
  { cap: 'VQA / captioning', status: 'Real VLM + measurements', color: 'text-green-400', note: 'SmolVLM-256M' },
  { cap: 'Grounding (water, vegetation, built-up)', status: 'Classical-CV baseline', color: 'text-cyan-400', note: 'Otsu + spectral idx' },
  { cap: 'Change detection + Change-VQA', status: 'Classical-CV baseline', color: 'text-cyan-400', note: 'CVA + Otsu' },
  { cap: 'Optical–SAR fusion', status: 'Rule-based baseline', color: 'text-amber-400', note: 'mask agreement' },
  { cap: 'Agentic router', status: 'Laya + rule-based fallback', color: 'text-green-400', note: 'auto-selected' },
  { cap: 'RS adaptation (LoRA on BigEarthNet)', status: 'Script provided; adapter NOT trained', color: 'text-amber-400', note: 'UI shows badge' },
  { cap: 'PDF / JSON / GeoJSON reports', status: 'Implemented', color: 'text-green-400', note: 'reportlab' },
  { cap: 'Benchmark scores', status: 'Not run — results/ empty', color: 'text-red-400', note: 'run eval scripts' },
]

const PAPERS = [
  { id: '[6.1]', title: 'RSVQA', venue: 'TGRS 2020', limit: 'Single sensor, no change, no fusion.', use: 'Defines VQA question types we implement.' },
  { id: '[6.2]', title: 'RSVQA + BigEarthNet', venue: 'IGARSS 2021', limit: 'Sentinel-2 only.', use: 'Method our build_vqa_from_bigearthnet.py follows.' },
  { id: '[6.3]', title: 'VRSBench', venue: 'NeurIPS 2024', limit: 'No cross-modal.', use: 'Evidence adaptation helps; our eval set.' },
  { id: '[6.4]', title: 'CDVQA', venue: 'TGRS 2022', limit: 'Optical only.', use: 'Reference design for our change branch.' },
  { id: '[6.5]', title: 'GeoChat', venue: 'CVPR 2024', limit: 'Weights not public.', use: 'Defines RS-adapted VLM; we use SmolVLM + LoRA.' },
]

export default function About() {
  const [status, setStatus] = useState<any>(null)

  useEffect(() => {
    fetch('/api/status').then(r => r.json()).then(setStatus).catch(() => {})
  }, [])

  return (
    <main className="min-h-screen bg-[#080d18] text-slate-100">
      <Navbar />
      <div className="mx-auto max-w-5xl px-6 py-16 lg:px-8 space-y-16">

        {/* Header */}
        <div>
          <p className="mb-3 text-xs font-semibold uppercase tracking-[.2em] text-cyan-300">Architecture & Research</p>
          <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">SatQuery AI</h1>
          <p className="mt-3 text-sm text-slate-500">SIH 2026 · ISRO/SAC Problem Statement 26167 · Theme: Space Technology</p>
        </div>

        {/* Live system status */}
        {status && (
          <div className="rounded-2xl border border-white/10 bg-[#0c1423] p-6">
            <p className="mb-4 text-xs font-semibold uppercase tracking-[.2em] text-slate-500">Live system status</p>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-xl border border-white/10 bg-[#080d18] p-4">
                <p className="text-[10px] text-slate-500 mb-1">Router</p>
                <p className={`font-semibold text-sm ${status.router_in_use === 'laya' ? 'text-green-400' : 'text-amber-400'}`}>{status.router_in_use}</p>
              </div>
              <div className="rounded-xl border border-white/10 bg-[#080d18] p-4">
                <p className="text-[10px] text-slate-500 mb-1">VLM</p>
                <p className="font-mono text-[10px] text-cyan-300 truncate">{status.vlm_model_id}</p>
              </div>
              <div className="rounded-xl border border-white/10 bg-[#080d18] p-4 col-span-2">
                <p className="text-[10px] text-slate-500 mb-1">RS Adaptation</p>
                <p className={`font-semibold text-sm ${status.adapter_present ? 'text-green-400' : 'text-amber-400'}`}>
                  {status.rs_adaptation_label}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Architecture */}
        <div className="rounded-2xl border border-white/10 bg-[#0c1423] p-6">
          <p className="mb-6 text-xs font-semibold uppercase tracking-[.2em] text-slate-500">Pipeline architecture</p>
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
            {['Query + Images', '→', 'Ingest', '→', 'Compatibility', '→', 'Router', '→', 'Tool', '→', 'Evidence', '→', 'Report'].map((n, i) => (
              n === '→' ? <span key={i} className="text-slate-700">→</span> : (
                <span key={i} className="rounded-lg border border-cyan-300/15 bg-cyan-300/5 px-2.5 py-1.5 text-cyan-200">{n}</span>
              )
            ))}
          </div>
          <div className="mt-6 grid grid-cols-3 gap-3 text-xs">
            {[
              { layer: 'Perception', items: ['SmolVLM-256M', 'Spectral indices', 'Scene statistics'] },
              { layer: 'Specialists', items: ['VQA · Caption', 'Grounding', 'Change · Fusion'] },
              { layer: 'Orchestration', items: ['Task router', 'Tool registry', 'Evidence bundle', 'Tracer'] },
            ].map(l => (
              <div key={l.layer} className="rounded-xl border border-white/10 bg-[#080d18] p-4">
                <p className="font-medium text-cyan-300 mb-2">{l.layer}</p>
                {l.items.map(it => <p key={it} className="text-slate-500">{it}</p>)}
              </div>
            ))}
          </div>
        </div>

        {/* Implementation status */}
        <div>
          <p className="mb-4 text-xs font-semibold uppercase tracking-[.2em] text-slate-500">Implementation status (honest)</p>
          <div className="rounded-2xl border border-white/10 bg-[#0c1423] overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-white/[.06]" style={{ background: 'rgba(34,211,238,0.04)' }}>
                  <th className="px-5 py-3 text-left font-semibold uppercase tracking-wider text-slate-500">Capability</th>
                  <th className="px-5 py-3 text-left font-semibold uppercase tracking-wider text-slate-500">Status</th>
                  <th className="px-5 py-3 text-left font-semibold uppercase tracking-wider text-slate-500">Notes</th>
                </tr>
              </thead>
              <tbody>
                {STATUS.map((r, i) => (
                  <tr key={i} className="border-b border-white/[.04]">
                    <td className="px-5 py-3 text-slate-300">{r.cap}</td>
                    <td className={`px-5 py-3 font-medium ${r.color}`}>{r.status}</td>
                    <td className="px-5 py-3 text-slate-600">{r.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Research papers */}
        <div>
          <p className="mb-6 text-xs font-semibold uppercase tracking-[.2em] text-slate-500">Research foundation</p>
          <div className="space-y-3">
            {PAPERS.map(p => (
              <div key={p.id} className="rounded-xl border border-white/10 bg-[#0c1423] p-5">
                <div className="flex flex-wrap items-center gap-2 mb-2">
                  <span className="font-mono text-[10px] text-cyan-400">{p.id}</span>
                  <span className="font-medium text-white">{p.title}</span>
                  <span className="rounded-full border border-cyan-300/15 bg-cyan-300/5 px-2 py-0.5 text-[10px] text-cyan-200">{p.venue}</span>
                </div>
                <div className="grid grid-cols-2 gap-3 text-xs mt-2">
                  <div><span className="text-slate-600">Limitation: </span><span className="text-amber-400">{p.limit}</span></div>
                  <div><span className="text-slate-600">How we use it: </span><span className="text-cyan-400">{p.use}</span></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* CTA */}
        <div className="flex flex-col items-start justify-between gap-4 rounded-2xl border border-white/10 bg-white/[.025] p-7 sm:flex-row sm:items-center">
          <div>
            <p className="font-medium text-white">Ready to see it in action?</p>
            <p className="mt-1 text-sm text-slate-500">Upload satellite imagery and ask anything.</p>
          </div>
          <Link to="/console" className="rounded-full bg-cyan-300 px-6 py-2.5 text-sm font-semibold text-[#07111d] transition hover:bg-cyan-200 whitespace-nowrap">
            Open demo →
          </Link>
        </div>
      </div>
    </main>
  )
}
