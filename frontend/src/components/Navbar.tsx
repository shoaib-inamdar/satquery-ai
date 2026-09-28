import { Link, useLocation } from 'react-router-dom'

export default function Navbar() {
  useLocation()

  return (
    <header className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6 lg:px-8">
      <Link to="/" className="flex items-center gap-2.5 text-sm font-semibold tracking-tight text-white">
        <span className="grid size-8 place-items-center rounded-lg bg-cyan-400 text-[#07111d] shadow-[0_0_24px_rgba(34,211,238,.3)]">
          🛰️
        </span>
        <span>SatQuery <span className="text-cyan-300">AI</span></span>
      </Link>

      <nav className="hidden items-center gap-8 text-sm text-slate-400 md:flex">
        <a href="/#features" className="hover:text-white transition-colors">Features</a>
        <a href="/#how" className="hover:text-white transition-colors">How it works</a>
        <a href="/#research" className="hover:text-white transition-colors">Research</a>
        <Link to="/benchmarks" className="hover:text-white transition-colors">Benchmarks</Link>
      </nav>

      <Link
        to="/console"
        className="rounded-full border border-cyan-400/40 bg-cyan-400/10 px-4 py-2 text-xs font-medium text-cyan-200 transition hover:bg-cyan-400/20"
      >
        Open demo →
      </Link>
    </header>
  )
}
