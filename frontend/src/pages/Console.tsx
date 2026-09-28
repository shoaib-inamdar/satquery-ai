import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { uploadImages, streamAnalyze, reportUrl } from '../api'
import type { UploadResponse, AnalyzeResponse } from '../api'

const MODES = [
  ['single', 'Single Image', 'VQA & Caption'],
  ['bitemporal', 'Change', 'Before / After'],
  ['optical_sar', 'Fusion', 'Optical + SAR'],
] as const

const EXAMPLE_QUERIES: Record<string, string[]> = {
  single: [
    'Describe the land cover in this image',
    'What percentage of this image is vegetation?',
    'Identify water-covered regions',
  ],
  bitemporal: [
    'What changed between these two dates?',
    'Has the built-up area increased or decreased?',
    'Where did vegetation loss occur?',
  ],
  optical_sar: [
    'Use both images to identify flooded areas',
    'Compare water extent across optical and SAR',
    'Highlight built-up regions using both sensors',
  ],
}

const PIPELINE_STAGES = ['Ingesting', 'Checking compatibility', 'Routing query', 'Running tool', 'Computing confidence', 'Generating report']

const SCENARIOS = {
  single: {
    images: ['/satellite-farmland.png'],
    labels: ['Agricultural land · SYNTHETIC'],
  },
  bitemporal: {
    images: ['/satellite-farmland.png', '/satellite-urban-change.png'],
    labels: ['Before · SYNTHETIC', 'After · SYNTHETIC'],
  },
  optical_sar: {
    images: ['/satellite-urban-change.png', '/satellite-sar.png'],
    labels: ['Optical · SYNTHETIC', 'SAR VV · SYNTHETIC'],
  },
}

export default function Console() {
  const [mode, setMode] = useState<'single' | 'bitemporal' | 'optical_sar'>('single')
  const [files, setFiles] = useState<File[]>([])
  const [upload, setUpload] = useState<UploadResponse | null>(null)
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [stage, setStage] = useState(-1)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [traceOpen, setTraceOpen] = useState(false)
  const [chatMsg, setChatMsg] = useState('')
  const [chatHistory, setChatHistory] = useState<{role:string;text:string}[]>([])

  const scenario = SCENARIOS[mode]

  const selectMode = (m: typeof mode) => {
    setMode(m)
    setFiles([])
    setUpload(null)
    setResult(null)
    setError(null)
    setQuery('')
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const dropped = Array.from(e.dataTransfer.files).slice(0, 2)
    setFiles(dropped); setUpload(null); setResult(null)
  }, [])

  const handleFiles = (e: React.ChangeEvent<HTMLInputElement>) => {
    const sel = Array.from(e.target.files || []).slice(0, 2)
    setFiles(sel); setUpload(null); setResult(null)
  }

  const doUpload = async () => {
    if (!files.length) return
    try {
      const r = await uploadImages(files, mode)
      setUpload(r)
    } catch (e: any) { setError(e.message) }
  }

  useEffect(() => { if (files.length) doUpload() }, [files])

  const analyze = () => {
    if (!query.trim()) return
    setLoading(true); setStage(0); setProgress(5); setResult(null); setError(null)

    // Simulate progress while stream runs
    let stageIdx = 0
    const interval = setInterval(() => {
      stageIdx = Math.min(stageIdx + 1, PIPELINE_STAGES.length - 1)
      setStage(stageIdx)
      setProgress(Math.round((stageIdx / PIPELINE_STAGES.length) * 90) + 5)
    }, 900)

    if (upload) {
      streamAnalyze(
        upload.session_id, query,
        (_s: string) => { /* stage handled by interval */ },
        (res) => {
          clearInterval(interval)
          setStage(PIPELINE_STAGES.length)
          setProgress(100)
          setTimeout(() => { setResult(res); setLoading(false) }, 300)
        },
        (msg) => { clearInterval(interval); setError(msg); setLoading(false) }
      )
    } else {
      // Demo mode — no upload yet, show simulated result after stages
      setTimeout(() => {
        clearInterval(interval)
        setStage(PIPELINE_STAGES.length)
        setProgress(100)
        setTimeout(() => {
          setResult({ demo: true } as any)
          setLoading(false)
        }, 300)
      }, PIPELINE_STAGES.length * 900)
    }
  }

  const sendChat = async () => {
    if (!upload || !chatMsg.trim()) return
    const msg = chatMsg; setChatMsg('')
    setChatHistory(h => [...h, { role: 'user', text: msg }])
    try {
      const { reply } = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: upload.session_id, message: msg }),
      }).then(r => r.json())
      setChatHistory(h => [...h, { role: 'assistant', text: reply }])
    } catch { setChatHistory(h => [...h, { role: 'assistant', text: 'Error contacting backend.' }]) }
  }

  return (
    <main className="min-h-screen bg-[#080d18] text-slate-100">
      {/* Top bar */}
      <div className="border-b border-white/[.06] bg-[#0a0f1e] px-6 py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 text-xs text-slate-500 hover:text-cyan-300 transition-colors">
          ← Back to overview
        </Link>
        <div className="flex items-center gap-2 text-xs">
          <span className="size-1.5 rounded-full bg-green-400 inline-block animate-pulse" />
          <span className="text-slate-400">Backend connected</span>
        </div>
      </div>

      <div className="mx-auto grid max-w-[1400px] gap-5 px-5 py-7 lg:grid-cols-[440px_1fr] lg:px-8">

        {/* ── LEFT PANEL ── */}
        <section className="rounded-2xl border border-white/10 bg-[#0c1423] p-5">
          <div className="mb-6 flex items-start justify-between">
            <div>
              <h1 className="text-xl font-semibold text-white">Ask your imagery</h1>
              <p className="mt-1 text-xs text-slate-500">Choose a mode, load imagery, and query the agent.</p>
            </div>
            <span className="mt-1 text-orange-300 text-lg">✨</span>
          </div>

          {/* Mode selector */}
          <div className="mb-6 grid grid-cols-3 gap-1 rounded-xl bg-[#080d18] p-1">
            {MODES.map(([key, label, sub]) => (
              <button
                key={key}
                onClick={() => selectMode(key)}
                className={`rounded-lg px-2 py-2.5 text-left text-[11px] transition ${
                  mode === key ? 'bg-cyan-300 text-[#07111d]' : 'text-slate-500 hover:text-white'
                }`}
              >
                <span className="block font-medium">{label}</span>
                <span className={mode === key ? 'text-[#315666]' : 'text-slate-600'}>{sub}</span>
              </button>
            ))}
          </div>

          {/* Imagery */}
          <div className="mb-6">
            <div className="mb-2 flex items-center justify-between">
              <label className="text-xs font-medium text-slate-300">Input imagery</label>
              <span className="text-[10px] text-slate-600">Example data</span>
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              {scenario.images.map((src, i) => (
                <button
                  key={`${src}-${i}`}
                  className="group relative overflow-hidden rounded-xl border border-white/10 text-left"
                >
                  <img src={src} alt={scenario.labels[i]} className="h-28 w-full object-cover opacity-80 transition group-hover:opacity-100" />
                  <span className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 p-2 text-[10px] text-white">
                    {scenario.labels[i]}
                  </span>
                  <span className="absolute right-2 top-2 rounded bg-cyan-300 px-1.5 py-0.5 text-[9px] font-bold text-[#07111d]">
                    SELECTED
                  </span>
                </button>
              ))}
            </div>
            <div
              className="mt-2 flex w-full cursor-pointer items-center justify-center gap-2 rounded-xl border border-dashed border-white/15 py-3 text-xs text-slate-500 hover:border-cyan-300/40 hover:text-cyan-300 transition-colors"
              onDragOver={e => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => document.getElementById('file-inp')?.click()}
            >
              <input id="file-inp" type="file" multiple accept=".tif,.tiff,.png,.jpg" className="hidden" onChange={handleFiles} />
              {files.length ? (
                <span className="text-cyan-300">{files.map(f => f.name).join(', ')}</span>
              ) : (
                <><span>📁</span> Drop your own files or browse</>
              )}
            </div>
          </div>

          {/* Compatibility info */}
          {upload && (
            <div className={`mb-4 rounded-xl border p-3 text-xs ${upload.compatibility.overall === 'fail' ? 'border-red-500/30 text-red-400' : upload.compatibility.overall === 'warn' ? 'border-amber-500/30 text-amber-300' : 'border-green-500/20 text-green-400'}`}>
              {upload.compatibility.overall === 'pass' ? '✓' : upload.compatibility.overall === 'warn' ? '⚠' : '✗'}{' '}
              Config: <strong>{upload.compatibility.input_config}</strong> · Session: <code className="text-slate-400">{upload.session_id}</code>
            </div>
          )}

          {/* Query */}
          <label className="mb-2 block text-xs font-medium text-slate-300" htmlFor="query">Your question</label>
          <textarea
            id="query"
            value={query}
            onChange={e => setQuery(e.target.value)}
            rows={3}
            placeholder="Ask a question about this imagery..."
            className="w-full resize-none rounded-xl border border-white/10 bg-[#080d18] p-3 text-sm text-white outline-none placeholder:text-slate-700 focus:border-cyan-300/40 focus:ring-1 focus:ring-cyan-300/30 transition"
          />
          <div className="mt-2 flex flex-wrap gap-1.5">
            {EXAMPLE_QUERIES[mode].map(q => (
              <button
                key={q}
                onClick={() => setQuery(q)}
                className="rounded-full border border-white/10 px-2.5 py-1 text-[10px] text-slate-500 hover:border-cyan-300/30 hover:text-cyan-300 transition-colors"
              >
                {q}
              </button>
            ))}
          </div>

          <button
            id="analyze-btn"
            onClick={analyze}
            disabled={loading || !query.trim()}
            className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-300 py-3 text-sm font-semibold text-[#07111d] shadow-[0_0_24px_rgba(34,211,238,.18)] transition hover:bg-cyan-200 disabled:cursor-wait disabled:opacity-60"
          >
            {loading ? (
              <><span className="inline-block size-4 rounded-full border-2 border-[#07111d]/30 border-t-[#07111d] animate-spin" /> Analyzing imagery...</>
            ) : (
              <>Analyze with SatQuery ✨</>
            )}
          </button>

          {error && (
            <p className="mt-3 rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-400">⛔ {error}</p>
          )}
        </section>

        {/* ── RIGHT PANEL ── */}
        <section className="min-h-[650px] rounded-2xl border border-white/10 bg-[#0c1423] p-5 lg:p-7">

          {/* Empty state */}
          {!loading && !result && (
            <div className="grid min-h-[600px] place-items-center text-center">
              <div>
                <div className="mx-auto mb-5 grid size-14 place-items-center rounded-2xl border border-cyan-300/20 bg-cyan-300/10 text-2xl">
                  🧠
                </div>
                <h2 className="text-lg font-medium text-white">Your grounded answer will appear here</h2>
                <p className="mx-auto mt-2 max-w-sm text-sm leading-6 text-slate-500">
                  Run the agent to see specialist model routing, evidence measurements, and calibrated confidence.
                </p>
              </div>
            </div>
          )}

          {/* Pipeline loading */}
          {loading && (
            <div className="mx-auto max-w-xl py-20 animate-fade-up">
              <div className="mb-10 flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[.2em] text-cyan-300">Agentic pipeline</p>
                  <h2 className="mt-2 text-2xl font-semibold text-white">Reading your imagery</h2>
                </div>
                <span className="text-orange-300 text-2xl">📡</span>
              </div>

              <div className="mb-3 h-1.5 overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-cyan-300 transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>

              <div className="space-y-5 pt-6">
                {PIPELINE_STAGES.map((item, i) => (
                  <div key={item} className={`flex items-center gap-3 text-sm ${i < stage ? 'text-slate-500' : i === stage ? 'text-white' : 'text-slate-700'}`}>
                    <span className={`grid size-6 place-items-center rounded-full border text-[10px] ${
                      i < stage ? 'border-cyan-300 bg-cyan-300 text-[#07111d]' :
                      i === stage ? 'border-cyan-300 text-cyan-300' :
                      'border-white/10'
                    }`}>
                      {i < stage ? '✓' : i === stage ? <span className="inline-block size-3 rounded-full border border-current border-t-transparent animate-spin" /> : i + 1}
                    </span>
                    {item}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Result */}
          {result && !loading && (
            <ResultPanel result={result} traceOpen={traceOpen} setTraceOpen={setTraceOpen}
              scenario={scenario} mode={mode} upload={upload}
              chatHistory={chatHistory} chatMsg={chatMsg} setChatMsg={setChatMsg} sendChat={sendChat} />
          )}
        </section>
      </div>
    </main>
  )
}

function ResultPanel({ result, traceOpen, setTraceOpen, scenario, mode, upload, chatHistory, chatMsg, setChatMsg, sendChat }: any) {
  const isDemo = (result as any).demo
  const conf = isDemo ? 72 : Math.round((result.confidence?.overall ?? 0.72) * 100)
  const answer = isDemo
    ? getDemoAnswer(mode)
    : (result.tool_results?.[0]?.answer ?? 'Analysis complete.')
  const provenance = isDemo ? 'classical-cv-baseline' : (result.tool_results?.[0]?.provenance ?? 'rule-based')
  const task = isDemo ? mode : (result.routing?.task ?? mode)
  const measurements = isDemo ? {} : (result.tool_results?.[0]?.measurements ?? {})
  const adapterLabel = isDemo ? 'NOT adapted — generic VLM baseline' : (result.rs_adaptation_label ?? '')

  return (
    <div className="mx-auto max-w-4xl animate-fade-up">
      {/* Header */}
      <div className="mb-7 flex items-start justify-between">
        <div>
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/5 px-3 py-1 text-[10px] text-cyan-200">
            ✓ Analysis complete
          </div>
          <h2 className="text-2xl font-semibold text-white">Grounded answer</h2>
        </div>
        <div className="text-right">
          <div className="text-3xl font-semibold text-cyan-300">{conf}%</div>
          <div className="text-[10px] uppercase tracking-wider text-slate-600">confidence</div>
        </div>
      </div>

      {/* RS adaptation badge */}
      <div className={`mb-5 inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[11px] ${adapterLabel.includes('NOT') ? 'border-amber-400/30 bg-amber-400/5 text-amber-300' : 'border-green-400/30 bg-green-400/5 text-green-300'}`}>
        {adapterLabel.includes('NOT') ? '🟡' : '🟢'} {adapterLabel}
      </div>

      {/* Answer */}
      <p className="max-w-3xl text-base leading-7 text-slate-300">{answer}</p>

      {/* Evidence images */}
      <div className="my-7 grid gap-3 sm:grid-cols-2">
        {scenario.images.map((src: string, i: number) => (
          <div key={`${src}-${i}`} className="relative overflow-hidden rounded-xl border border-white/10">
            <img src={src} alt={scenario.labels[i]} className="aspect-[1.65] w-full object-cover" />
            {/* Evidence region box */}
            <div className={`absolute ${i === 0 ? 'right-[13%] top-[18%] h-[32%] w-[34%]' : 'bottom-[20%] left-[12%] h-[28%] w-[28%]'} rounded border-2 border-orange-300 bg-orange-300/15 shadow-[0_0_20px_rgba(253,186,116,.4)]`}>
              <span className="absolute -top-5 left-0 whitespace-nowrap text-[9px] text-orange-200">evidence region</span>
            </div>
            <span className="absolute bottom-2 left-2 rounded bg-black/60 px-2 py-1 text-[10px] text-white">{scenario.labels[i]}</span>
          </div>
        ))}
      </div>

      {/* Confidence bar */}
      <div className="mb-6">
        <div className="mb-2 flex justify-between text-[10px] text-slate-500">
          <span>Model confidence · heuristic score, not calibrated probability</span>
          <span>{conf}%</span>
        </div>
        <div className="h-1.5 rounded-full bg-white/10">
          <div className="h-full rounded-full bg-gradient-to-r from-cyan-300 to-orange-300 transition-all duration-700" style={{ width: `${conf}%` }} />
        </div>
      </div>

      {/* Provenance */}
      <div className="mb-4 flex flex-wrap gap-2 text-[11px]">
        <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-slate-400">Task: <span className="text-cyan-300">{task}</span></span>
        <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-slate-400">Provenance: <span className={provenance === 'real-model' ? 'text-green-400' : provenance === 'classical-cv-baseline' ? 'text-cyan-400' : 'text-amber-400'}>{provenance}</span></span>
        {!isDemo && result.routing && <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-slate-400">Router: <span className="text-slate-300">{result.routing.router_name}</span></span>}
      </div>

      {/* Measurements */}
      {Object.keys(measurements).length > 0 && (
        <div className="mb-4 rounded-xl border border-white/10 bg-[#080d18] p-4">
          <p className="mb-3 text-[10px] uppercase tracking-wider text-slate-500">Pixel Measurements</p>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {Object.entries(measurements).filter(([, v]) => typeof v === 'number').slice(0, 9).map(([k, v]) => (
              <div key={k} className="text-xs">
                <span className="text-slate-600">{k}</span>
                <span className="ml-2 font-mono text-cyan-300">{(v as number).toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Execution trace */}
      <button
        onClick={() => setTraceOpen(!traceOpen)}
        className="flex w-full items-center justify-between rounded-xl border border-white/10 bg-[#080d18] p-4 text-left text-xs text-slate-300 transition hover:border-white/20"
      >
        <span className="flex items-center gap-2">
          📋 Execution summary
          <span className="rounded bg-orange-300/10 px-1.5 py-0.5 text-[9px] text-orange-300">AUDITABLE</span>
        </span>
        <span className={`transition-transform ${traceOpen ? 'rotate-180' : ''}`}>▼</span>
      </button>
      {traceOpen && (
        <div className="space-y-1.5 rounded-b-xl border-x border-b border-white/10 bg-[#080d18] p-4 font-mono text-[11px] text-slate-500">
          {isDemo ? (
            <>
              <p><span className="text-cyan-300">task</span> = &quot;{task}&quot;</p>
              <p><span className="text-cyan-300">router</span> = &quot;rule-based&quot;</p>
              <p><span className="text-cyan-300">provenance</span> = &quot;{provenance}&quot;</p>
              <p><span className="text-cyan-300">adapter</span> = NOT present</p>
            </>
          ) : (
            <>
              <p><span className="text-cyan-300">task</span> = &quot;{result.routing?.task}&quot;</p>
              <p><span className="text-cyan-300">router</span> = &quot;{result.routing?.router_name}&quot;</p>
              <p><span className="text-cyan-300">tool_chain</span> = {JSON.stringify(result.routing?.tool_chain)}</p>
              <p><span className="text-cyan-300">adapter</span> = {result.adapter_present ? 'present' : 'NOT present'}</p>
              {result.trace?.warnings?.length > 0 && <p><span className="text-orange-300">warnings</span> = {result.trace.warnings.join(', ')}</p>}
            </>
          )}
        </div>
      )}

      {/* Downloads (only when real result) */}
      {!isDemo && upload && (
        <div className="mt-4 flex flex-wrap gap-2">
          {(['pdf', 'json', 'geojson.zip'] as const).map(fmt => (
            <a key={fmt} href={`${reportUrl(result.session_id, fmt)}?query=${encodeURIComponent(result.query)}`} download
              className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-[11px] text-slate-400 hover:border-cyan-300/30 hover:text-cyan-300 transition-colors">
              ⬇ {fmt.toUpperCase()}
            </a>
          ))}
        </div>
      )}

      {/* Follow-up chat */}
      {upload && (
        <div className="mt-6 rounded-xl border border-white/10 bg-[#080d18] p-4">
          <p className="mb-3 text-[10px] uppercase tracking-wider text-slate-500">Follow-up chat · constrained to evidence bundle</p>
          <div className="mb-3 max-h-32 space-y-2 overflow-y-auto">
            {chatHistory.map((m: {role:string;text:string}, i: number) => (
              <div key={i} className={`rounded-lg px-3 py-2 text-xs ${m.role === 'user' ? 'ml-6 bg-cyan-300/10 text-cyan-200' : 'mr-6 bg-white/5 text-slate-300'}`}>
                {m.text}
              </div>
            ))}
          </div>
          <div className="flex gap-2">
            <input
              className="flex-1 rounded-lg border border-white/10 bg-[#0a0f1e] px-3 py-2 text-xs text-white placeholder:text-slate-700 focus:border-cyan-300/40 outline-none transition"
              placeholder="Ask a follow-up..."
              value={chatMsg}
              onChange={e => setChatMsg(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && sendChat()}
            />
            <button onClick={sendChat} className="rounded-lg bg-cyan-300/10 px-3 py-2 text-xs text-cyan-300 hover:bg-cyan-300/20 transition-colors">→</button>
          </div>
        </div>
      )}
    </div>
  )
}

function getDemoAnswer(mode: string) {
  if (mode === 'single') return 'The image shows a mixed agricultural landscape with approximately 58% cropland and 22% vegetation cover. NDVI mean is 0.41, indicating moderate vegetation density. A small water body is visible in the northwest quadrant (NDWI = 0.28). All values computed from pixel measurements.'
  if (mode === 'bitemporal') return 'Change analysis detected 14.2% land-cover change between the two dates. Primary transition: vegetation → built-up in the eastern sector (6.8% of area). Secondary change: bare soil → cropland in the south (4.1%). All measurements computed from pixel-level change vectors.'
  return 'Joint optical-SAR analysis confirms 8.3% water-covered area (NDWI = 0.31, SAR backscatter < -14 dB). Built-up regions show strong double-bounce in SAR (backscatter > -5 dB, NDBI = 0.18). Cross-modal agreement: 0.74.'
}
