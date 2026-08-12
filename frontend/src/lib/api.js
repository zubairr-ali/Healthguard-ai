
const BASE = import.meta.env.VITE_API_BASE || '';

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request(path, options = {}) {
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
    }
    throw new ApiError(detail, res.status);
  }
  return res.json();
}

function mapPredictionResponse(raw) {
  let probability = Number(raw.risk_score);
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
