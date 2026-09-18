import axios from 'axios'
import type { Health, ModelInfo, Prediction, PredictionInput } from './types'

const baseURL = import.meta.env.VITE_API_URL || ''
export const api = axios.create({ baseURL, timeout: 10000 })
api.interceptors.request.use((config) => { config.headers['X-API-Key'] = import.meta.env.VITE_API_KEY || ''; return config })

export async function getHealth(): Promise<Health> {
  const { data } = await api.get<Health>('/api/v1/health'); return data
}
export async function getModelInfo(): Promise<ModelInfo> {
  const { data } = await api.get<ModelInfo>('/api/v1/model-info'); return data
}
export async function predict(input: PredictionInput, version: 'v1' | 'v2'): Promise<Prediction> {
  const { data } = await api.post<Prediction>(`/api/${version}/predict`, input); return data
}
export async function getMetrics(): Promise<string> {
  const { data } = await api.get<string>('/metrics', { transformResponse: [(value) => value] }); return data
}
export function metricsSummary(metrics: string) {
  const total = [...metrics.matchAll(/iris_predictions_total\{[^}]*\}\s+([\d.e+-]+)/g)].reduce((sum, match) => sum + Number(match[1]), 0)
  const failures = [...metrics.matchAll(/http_requests_total\{[^}]*status_code="5\d\d"[^}]*\}\s+([\d.e+-]+)/g)].reduce((sum, match) => sum + Number(match[1]), 0)
  return { total, failures }
}
export const backendUrl = baseURL || 'local proxy -> FastAPI'
export const docsUrl = baseURL ? `${baseURL}/docs` : '/docs'
