/**
 * Thin wrapper around the HealthGuard AI FastAPI backend.
 * In dev, Vite proxies /api/* to http://127.0.0.1:8000 (see vite.config.js).
 * In production, serve the built frontend behind the same origin as the
 * API, or set VITE_API_BASE at build time.
 */

const BASE = import.meta.env.VITE_API_BASE || '';

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request(path, options = {}) {
  // TabPFN's SHAP explanation is inherently slow on CPU — this bounds how
  // long the UI will wait before giving up with a clear message, instead
  // of spinning forever if the backend is still slower than expected.
  const TIMEOUT_MS = 90_000;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  let res;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      signal: controller.signal,
      ...options,
    });
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new ApiError(
        `The request took longer than ${TIMEOUT_MS / 1000} seconds and was cancelled. ` +
          'For heart predictions this can happen because TabPFN\u2019s explanation step is ' +
          'computationally expensive on CPU — check the backend terminal to see if it\u2019s ' +
          'still processing.',
        408
      );
    }
    // Network-level failure — almost always "backend isn't running".
    throw new ApiError(
      'Cannot reach the HealthGuard API. Start the backend with `uvicorn main:app --reload` and try again.',
      0
    );
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || body.message || detail;
    } catch {
      /* response wasn't JSON — keep statusText */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json();
}

/**
 * Backend returns:
 *   { condition, risk_score, risk_level, model_used, shap_values: {feature: value}, advisory }
 * Frontend components expect:
 *   { probability, model_used, risk_level, contributions: [{feature, value}], advisory_note }
 * This is the one place that mapping happens, so it never has to be
 * guessed again at each call site.
 */
function mapPredictionResponse(raw) {
  let probability = Number(raw.risk_score);
  // Defensive: if the backend ever returns a 0-100 scale instead of 0-1,
  // this still renders correctly instead of showing "8400%".
  if (Number.isFinite(probability) && probability > 1) probability = probability / 100;

  const contributions = Object.entries(raw.shap_values || {}).map(([feature, value]) => ({
    feature,
    value: Number(value),
  }));

  return {
    probability,
    model_used: raw.model_used,
    risk_level: raw.risk_level,
    contributions,
    advisory_note: raw.advisory,
  };
}

export const api = {
  predictHeart: (payload) =>
    request('/api/predict/heart', { method: 'POST', body: JSON.stringify(payload) }).then(
      mapPredictionResponse
    ),
  predictDiabetes: (payload) =>
    request('/api/predict/diabetes', { method: 'POST', body: JSON.stringify(payload) }).then(
      mapPredictionResponse
    ),
  shapGlobal: (condition) => request(`/api/shap/global/${condition}`),
  history: (limit = 20) =>
    request('/api/history').then((data) => (data.records || []).slice(0, limit)),
  stats: () => request('/api/stats'),
  models: (condition) => request(`/api/models/${condition}`),
};

export { ApiError };
