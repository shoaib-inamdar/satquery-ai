import { useEffect, useState } from 'react'
import Navbar from '../components/Navbar'

export default function Benchmarks() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/benchmarks').then(r => r.json()).then(d => { setData(d); setLoading(false) }).catch(() => setLoading(false))
  }, [])

  const results = data?.results || []

  return (
    <main className="min-h-screen bg-[#080d18] text-slate-100">
      <Navbar />
      <div className="mx-auto max-w-5xl px-6 py-16 lg:px-8">
        <div className="mb-12">
          <p className="mb-3 text-xs font-semibold uppercase tracking-[.2em] text-cyan-300">Evaluation</p>
          <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">Benchmark Results</h1>
          <p className="mt-4 text-sm leading-6 text-slate-500">
            Only results actually computed by the eval scripts appear here. We do not show placeholder scores.
          </p>
        </div>

        {loading ? (
          <div className="grid place-items-center py-24">
            <span className="inline-block size-8 rounded-full border-2 border-white/10 border-t-cyan-300 animate-spin" />
          </div>
        ) : results.length === 0 ? (
          <div className="space-y-5">
            <div className="rounded-2xl border border-white/10 bg-[#0c1423] p-10 text-center">
              <div className="mx-auto mb-5 grid size-14 place-items-center rounded-2xl border border-cyan-300/20 bg-cyan-300/10 text-2xl">📊</div>
              <h2 className="text-lg font-medium text-white">Not run yet</h2>
              <p className="mx-auto mt-2 max-w-sm text-sm leading-6 text-slate-500">
                No evaluation results found in <code className="text-cyan-300">results/</code>.
                Run the eval scripts with your own dataset copy to populate this page.
              </p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-[#0c1423] p-6">
              <p className="mb-4 text-xs font-semibold uppercase tracking-[.2em] text-slate-500">How to run</p>
              <pre className="overflow-x-auto rounded-xl bg-[#080d18] p-4 font-mono text-[11px] text-slate-300 leading-6">{
`python scripts/eval_vrsbench.py  --data /path/to/VRSBench_test
python scripts/eval_rsvqa.py     --data /path/to/RSVQA_test
python scripts/eval_cdvqa.py     --data /path/to/CDVQA_test`
              }</pre>
              <p className="mt-3 text-[11px] text-slate-600">
                ⚠ You must download the datasets separately. Dataset links are in the README.
              </p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/[.025] p-6">
              <p className="mb-3 text-xs font-semibold uppercase tracking-[.2em] text-slate-500">Reference numbers from literature</p>
              <div className="space-y-2">
                {[
                  ['RSVQA-LR', 'Presence', 'RSVQA baseline', '~90%'],
                  ['RSVQA-LR', 'Count', 'RSVQA baseline', '~64%'],
                  ['VRSBench', 'VQA', 'RS-tuned VLM (paper)', '~71%'],
                  ['CDVQA', 'Change-VQA', 'CDVQA baseline', '~78%'],
                ].map(([ds, task, model, score]) => (
                  <div key={`${ds}-${task}`} className="flex items-center justify-between rounded-lg border border-white/5 bg-[#080d18] px-4 py-2.5 text-xs">
                    <div className="flex items-center gap-4">
                      <span className="font-medium text-white">{ds}</span>
                      <span className="text-slate-500">{task}</span>
                      <span className="text-slate-600">{model}</span>
                    </div>
                    <span className="font-mono font-semibold text-slate-400">{score} <span className="text-[9px] text-slate-600">literature</span></span>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-[10px] text-slate-700">These are published baselines — not our scores. Our scores will appear above once eval scripts are run.</p>
            </div>
          </div>
        ) : (
          <div className="rounded-2xl border border-white/10 bg-[#0c1423] overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/[.06]" style={{ background: 'rgba(34,211,238,0.04)' }}>
                  {['Dataset', 'Task', 'Samples', 'Metric', 'Score', 'Date'].map(h => (
                    <th key={h} className="px-5 py-3.5 text-left text-[10px] font-semibold uppercase tracking-wider text-slate-500">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.map((r: any, i: number) => (
                  <tr key={i} className="border-b border-white/[.04] hover:bg-white/[.02] transition-colors">
                    <td className="px-5 py-3.5 font-medium text-white">{r.dataset}</td>
                    <td className="px-5 py-3.5 text-slate-400">{r.task}</td>
                    <td className="px-5 py-3.5 font-mono text-slate-400">{r.sample_count}</td>
                    <td className="px-5 py-3.5 text-slate-400">{r.metric}</td>
                    <td className="px-5 py-3.5 font-mono font-semibold text-cyan-300">{(r.score * 100).toFixed(1)}%</td>
                    <td className="px-5 py-3.5 text-slate-600 text-xs">{r.date?.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </main>
  )
}
