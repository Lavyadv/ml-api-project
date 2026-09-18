import { ArrowRight, Code2, ShieldCheck } from 'lucide-react'
import { docsUrl, backendUrl } from './api'

const endpoints = [
  ['GET', '/api/v1/health', 'Service health', 'Returns whether the API and model are ready to serve predictions.'],
  ['GET', '/api/v1/model-info', 'Model metadata', 'Returns the loaded model type, version, classes, and training details.'],
  ['POST', '/api/v1/predict', 'Single prediction', 'Classifies one Iris flower from four measurements in centimeters.'],
  ['POST', '/api/v2/predict', 'Single prediction v2', 'Returns ranked probabilities and training metadata.'],
  ['POST', '/api/v1/predict-batch', 'Batch prediction', 'Scores multiple flowers in one request, preserving input order.'],
  ['GET', '/metrics', 'Prometheus metrics', 'Exposes request, latency, and prediction counters.'],
] as const

export default function DocsPage() {
  return <div className="space-y-7">
    <div className="flex flex-col justify-between gap-5 md:flex-row md:items-end">
      <div><p className="mono text-[10px] uppercase tracking-[.16em] text-[#39c9a6]">Developer resources</p><h1 className="display mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">API reference.</h1><p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500">Connect a client to the ML prediction service with these available endpoints.</p></div>
      <a href={docsUrl} target="_blank" rel="noreferrer" className="flex items-center gap-2 self-start rounded-lg border border-white/10 px-4 py-2.5 text-sm text-slate-300 transition hover:border-[#39c9a6]/40 hover:text-white md:self-auto"><Code2 size={15} />Open Swagger UI <ArrowRight size={15} /></a>
    </div>
    <div className="card overflow-hidden">
      <div className="flex flex-col gap-3 border-b border-white/10 bg-white/[.02] p-5 sm:flex-row sm:items-center sm:justify-between sm:p-6"><div><p className="mono text-[10px] uppercase tracking-[.16em] text-[#39c9a6]">Base URL</p><code className="mt-2 block text-sm text-slate-200">{backendUrl}</code></div><span className="flex items-center gap-2 text-xs text-slate-500"><ShieldCheck size={15} className="text-[#39c9a6]" />X-API-Key required for versioned routes</span></div>
      <div className="divide-y divide-white/5">{endpoints.map(([method, path, title, description]) => <div key={`${method}-${path}`} className="grid gap-5 p-5 sm:grid-cols-[170px_1fr] sm:p-6"><div><span className={`mono inline-block rounded px-2 py-1 text-[10px] font-medium ${method === 'GET' ? 'bg-sky-400/10 text-sky-300' : 'bg-amber-400/10 text-amber-300'}`}>{method}</span><code className="mt-3 block break-all text-xs text-slate-300">{path}</code></div><div><h3 className="font-medium text-slate-100">{title}</h3><p className="mt-2 text-sm leading-6 text-slate-500">{description}</p></div></div>)}</div>
    </div>
  </div>
}
