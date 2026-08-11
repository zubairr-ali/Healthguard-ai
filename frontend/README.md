# HealthGuard AI — Frontend

React + Tailwind CSS v4 frontend for the HealthGuard AI clinical decision
support system (module 6WCM0029).

## Setup

```bash
cd frontend/app
npm install
npm run dev
```

Opens at http://localhost:5173. In dev, requests to `/api/*` are proxied to
your local FastAPI backend at `http://127.0.0.1:8000` (see `vite.config.js`).
Start the backend separately:

```bash
cd backend
.\venv\Scripts\activate
uvicorn main:app --reload
```

Both need to be running for predictions to work. The frontend itself will
load and render fine without the backend — you'll just see a "Cannot reach
the HealthGuard API" message when submitting a prediction form.

## Pages

- `/` — landing page
- `/predict/heart` — heart disease risk prediction (calls `/api/predict/heart`)
- `/predict/diabetes` — diabetes risk prediction (calls `/api/predict/diabetes`)
- `/findings` — static research results showcase (ECG, NLP, cross-dataset
  validation) — these are report findings, not live model endpoints
- `/history` — prediction history (calls `/api/history`)

## Design system

Defined in `src/index.css` under `@theme`. Dark is the default; `.light` on
`<html>` (toggled by the header button, persisted to localStorage) flips it
via a custom `light:` variant — used exactly like Tailwind's built-in `dark:`.

Signature visual element: the ECG waveform (`src/components/Waveform.jsx`),
reused in the hero, section dividers, and (compact) as a loading indicator —
grounded in the project's own Phase 2 ECG research rather than borrowed
generic medical iconography.

## Build for production

```bash
npm run build
```

Outputs to `dist/`. Serve `dist/` from the same origin as the FastAPI app,
or set `VITE_API_BASE` at build time if serving from a different origin.
