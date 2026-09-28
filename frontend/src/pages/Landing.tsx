import { Link } from 'react-router-dom'
import Navbar from '../components/Navbar'

const features = [
  { icon: '🔍', title: 'Single-Image VQA', text: 'Ask any plain-English question about one image. Numbers come from measured pixel statistics — never from the model\'s memory.' },
  { icon: '📝', title: 'Scene Captioning', text: 'Auto-describe land cover with NDVI, NDWI, and NDBI fractions computed directly from the image bands.' },
  { icon: '🎯', title: 'Text-Guided Grounding', text: 'Highlight water, vegetation, built-up, or bare soil. Returns GeoJSON polygons and km² area.' },
  { icon: '📅', title: 'Bi-temporal Change', text: 'Upload a before/after pair. Get a change map, class transition matrix, and change-VQA answer.' },
  { icon: '📡', title: 'Optical + SAR Fusion', text: 'Joint analysis of co-registered optical and SAR pairs with a 6-class agreement map.' },
  { icon: '🤖', title: 'Agentic Routing', text: 'Query routed automatically to the right specialist. Every decision logged and shown to you.' },
]

const steps = [
  ['01', 'Upload Image(s)', 'Single image, optical+SAR pair, or before/after pair.'],
  ['02', 'Ask a Question', 'Plain English — no GIS expertise needed.'],
  ['03', 'Route the Query', 'Agentic controller classifies the task automatically.'],
  ['04', 'Run Specialist', 'VQA, Grounding, Change, or Fusion tool executes.'],
  ['05', 'Grounded Answer', 'Text + measurements + calibrated confidence.'],
]

const papers = [
  { id: 'RSVQA · 2020', desc: 'Introduced VQA for remote-sensing images.' },
  { id: 'RSVQA + BigEarthNet · 2021', desc: 'Large VQA dataset from Sentinel-2 labels.' },
  { id: 'CDVQA · 2022', desc: 'Change detection + VQA on bi-temporal pairs.' },
  { id: 'VRSBench · 2024', desc: 'RS-tuned models beat generic baselines.' },
  { id: 'GeoChat · 2024', desc: 'RS-adapted VLM with region-level grounding.' },
]

export default function Landing() {
  return (
    <main className="min-h-screen overflow-hidden bg-[#080d18] text-slate-100 selection:bg-cyan-400/30">
      <Navbar />

      {/* ── Hero ── */}
      <section className="relative mx-auto max-w-7xl px-6 pb-24 pt-20 text-center lg:px-8 lg:pt-32">
        {/* Background satellite image */}
        <div className="pointer-events-none absolute inset-x-0 top-0 -z-0 mx-auto h-[600px] max-w-5xl opacity-40 [mask-image:linear-gradient(to_bottom,black,transparent_80%)]">
          <img src="/satellite-urban-change.png" alt="Satellite view" className="h-full w-full object-cover mix-blend-screen" />
          <div className="absolute inset-0 bg-gradient-to-b from-[#080d18]/30 via-[#080d18]/70 to-[#080d18]" />
        </div>

        <div className="relative z-10 mx-auto max-w-4xl">
          {/* Badge */}
          <div className="mb-7 inline-flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/5 px-3 py-1.5 text-xs text-cyan-200">
            <span className="size-1.5 animate-pulse rounded-full bg-cyan-300 inline-block" />
            Agentic remote sensing intelligence · SIH 2026 · ISRO/SAC Problem 26167
          </div>

          <h1 className="text-balance text-5xl font-semibold tracking-[-0.05em] text-white sm:text-7xl">
            Ask your satellite<br />
            <span className="bg-gradient-to-r from-cyan-300 via-sky-200 to-orange-300 bg-clip-text text-transparent">
              images anything.
            </span>
          </h1>

          <p className="mx-auto mt-7 max-w-2xl text-pretty text-lg leading-8 text-slate-400">
            An agentic vision-language assistant that reads optical, SAR, and multitemporal satellite imagery — and answers your questions in plain English.
          </p>

          <Link
            to="/console"
            id="hero-demo-btn"
            className="mt-10 inline-flex items-center gap-3 rounded-full bg-cyan-300 px-7 py-3.5 text-sm font-semibold text-[#07111d] shadow-[0_0_40px_rgba(34,211,238,.35)] transition hover:scale-[1.02] hover:bg-cyan-200"
          >
            Try the live demo →
          </Link>

          {/* Status bar */}
          <div className="relative z-10 mx-auto mt-20 flex max-w-2xl items-center justify-between rounded-2xl border border-white/10 bg-[#0c1423]/80 px-5 py-4 text-left shadow-2xl backdrop-blur">
            <div className="flex items-center gap-3">
              <span className="grid size-9 place-items-center rounded-lg bg-orange-300/10 text-orange-300 text-base">⚡</span>
              <div>
                <p className="text-xs font-medium text-white">Agent online</p>
                <p className="text-[11px] text-slate-500">6 specialist tools ready</p>
              </div>
            </div>
            <div className="hidden items-center gap-5 text-[11px] text-slate-500 sm:flex">
              <span>OPTICAL <b className="text-cyan-300">READY</b></span>
              <span>SAR <b className="text-cyan-300">READY</b></span>
              <span>VQA <b className="text-cyan-300">READY</b></span>
            </div>
          </div>
        </div>
      </section>

      {/* ── Problem strip ── */}
      <section className="border-y border-white/[.06] bg-[#0b1220] px-6 py-20 lg:px-8">
        <div className="mx-auto max-w-5xl">
          <p className="mx-auto max-w-3xl text-center text-lg leading-8 text-slate-300">
            Today, extracting an answer from satellite imagery means knowing GIS tools, picking the right model, and manually cross-referencing optical and radar data.
          </p>
          <div className="mt-12 grid gap-4 md:grid-cols-3">
            {[
              ['🗺️', 'Isolated Tools', 'Analysis is scattered across specialist GIS software.'],
              ['🧩', 'Needs GIS Expertise', 'The insight is locked behind complex workflows.'],
              ['📡', 'No Cross-Modal Reasoning', 'Optical and radar signals rarely meet in one answer.'],
            ].map(([icon, title, text]) => (
              <div key={title} className="rounded-2xl border border-white/10 bg-white/[.025] p-6">
                <div className="mb-5 text-2xl text-orange-300">{icon}</div>
                <h3 className="font-medium text-white">{title}</h3>
                <p className="mt-2 text-sm leading-6 text-slate-500">{text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section id="features" className="mx-auto max-w-7xl px-6 py-24 lg:px-8">
        <div className="mb-12 max-w-xl">
          <p className="mb-3 text-xs font-semibold uppercase tracking-[.2em] text-cyan-300">One intelligent interface</p>
          <h2 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">From pixels to proof.</h2>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {features.map(({ icon, title, text }) => (
            <div key={title} className="group rounded-2xl border border-white/10 bg-gradient-to-br from-white/[.06] to-white/[.015] p-7 transition hover:-translate-y-1 hover:border-cyan-300/30 hover:shadow-[0_0_35px_rgba(34,211,238,.08)]">
              <div className="mb-12 grid size-10 place-items-center rounded-xl border border-cyan-300/20 bg-cyan-300/10 text-lg">{icon}</div>
              <h3 className="text-lg font-medium text-white">{title}</h3>
              <p className="mt-2 max-w-sm text-sm leading-6 text-slate-500">{text}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── How it works ── */}
      <section id="how" className="border-y border-white/[.06] bg-[#0b1220] px-6 py-24 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="mb-12 max-w-xl">
            <p className="mb-3 text-xs font-semibold uppercase tracking-[.2em] text-cyan-300">A transparent pipeline</p>
            <h2 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">One question. Five intelligent steps.</h2>
          </div>
          <div className="grid gap-8 md:grid-cols-5">
            {steps.map(([num, title, text], i) => (
              <div key={num} className="relative">
                <div className="mb-5 flex items-center gap-3">
                  <span className="font-mono text-xs text-orange-300">{num}</span>
                  {i < 4 && <span className="absolute -right-4 hidden text-slate-700 md:block">›</span>}
                </div>
                <h3 className="text-sm font-medium text-white">{title}</h3>
                <p className="mt-2 text-xs leading-5 text-slate-500">{text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Sample imagery ── */}
      <section className="mx-auto max-w-7xl px-6 py-16 lg:px-8">
        <div className="mb-8 max-w-xl">
          <p className="mb-3 text-xs font-semibold uppercase tracking-[.2em] text-cyan-300">Example imagery</p>
          <h2 className="text-2xl font-semibold tracking-tight text-white">See what the agent can read.</h2>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {[
            { src: '/satellite-farmland.png', label: 'Agricultural land', tag: 'VQA · Caption' },
            { src: '/satellite-urban-change.png', label: 'Urban change', tag: 'Change Detection' },
            { src: '/satellite-sar.png', label: 'SAR radar image', tag: 'Fusion · Grounding' },
          ].map(({ src, label, tag }) => (
            <div key={label} className="group relative overflow-hidden rounded-2xl border border-white/10">
              <img src={src} alt={label} className="aspect-video w-full object-cover opacity-80 transition group-hover:opacity-100" />
              <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 p-4">
                <p className="text-sm font-medium text-white">{label}</p>
                <p className="text-[10px] text-cyan-300 mt-0.5">{tag}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Research ── */}
      <section id="research" className="mx-auto max-w-7xl px-6 py-16 lg:px-8">
        <div className="flex flex-col items-start justify-between gap-6 rounded-2xl border border-white/10 bg-white/[.025] p-7 sm:flex-row sm:items-center">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[.2em] text-slate-500">Research foundation</p>
            <p className="mt-2 text-sm text-slate-300">Built on peer-reviewed remote-sensing literature</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {papers.map(p => (
              <span key={p.id} className="rounded-full border border-cyan-300/15 bg-cyan-300/5 px-3 py-1.5 text-[11px] text-cyan-200">{p.id}</span>
            ))}
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="mx-auto flex max-w-7xl items-center justify-between border-t border-white/[.06] px-6 py-8 text-xs text-slate-600 lg:px-8">
        <span>SatQuery AI · SIH 2026 · ISRO/SAC Problem 26167</span>
        <Link to="/about" className="hover:text-white transition-colors">Research ↗</Link>
      </footer>
    </main>
  )
}
