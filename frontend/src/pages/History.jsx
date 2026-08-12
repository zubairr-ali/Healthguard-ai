import { useEffect, useState } from 'react';
import { Clock, Loader2, AlertTriangle, Inbox } from 'lucide-react';
import { Card, Badge } from '../components/ui';
import { api, ApiError } from '../lib/api';

function riskBadge(pct) {
  if (pct < 30) return <Badge tone="vital">Low · {pct.toFixed(0)}%</Badge>;
  if (pct < 60) return <Badge tone="amber">Moderate · {pct.toFixed(0)}%</Badge>;
  return <Badge tone="signal">High · {pct.toFixed(0)}%</Badge>;
}

export default function History() {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api
      .history(30)
      .then((rows) => setRows(rows))
      .catch((err) => setError(err instanceof ApiError ? err.message : 'Could not load history.'));
  }, []);

  return (
    <div className="max-w-4xl mx-auto px-5 sm:px-8 py-12">
      <div className="flex items-center gap-2.5 mb-2">
        <Clock size={16} className="text-vital-400" />
        <span className="text-xs font-semibold tracking-wide uppercase text-vital-400">Stored in SQLite</span>
      </div>
      <h1 className="font-display font-semibold text-3xl text-ink-50 light:text-ink-900">
        Prediction history
      </h1>
      <p className="text-ink-400 light:text-ink-500 mt-2 max-w-2xl">
        Every prediction made through this interface is logged to the local database via the
        FastAPI backend's <code className="data-readout text-xs">/api/history</code> endpoint.
      </p>

      <div className="mt-8">
        {error && (
          <Card className="p-4 border-signal-500/30 bg-signal-500/[0.04] flex gap-3 items-start">
            <AlertTriangle size={16} className="text-signal-400 shrink-0 mt-0.5" />
            <p className="text-sm text-signal-300">{error}</p>
          </Card>
        )}

        {!error && rows === null && (
          <div className="flex items-center gap-2.5 text-ink-400 py-16 justify-center">
            <Loader2 size={16} className="animate-spin" /> Loading history…
          </div>
        )}

        {!error && rows?.length === 0 && (
          <Card className="p-12 flex flex-col items-center text-center">
            <Inbox size={26} className="text-ink-600 mb-3" />
            <p className="text-sm text-ink-500 light:text-ink-400 max-w-xs">
              No predictions yet. Run one from the Heart or Diabetes risk pages and it will
              appear here.
            </p>
          </Card>
        )}

        {rows?.length > 0 && (
          <Card className="overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-ink-800 light:border-ink-100 text-left">
                  <th className="px-5 py-3 font-medium text-ink-400 light:text-ink-500 text-xs uppercase tracking-wide">Timestamp</th>
                  <th className="px-5 py-3 font-medium text-ink-400 light:text-ink-500 text-xs uppercase tracking-wide">Type</th>
                  <th className="px-5 py-3 font-medium text-ink-400 light:text-ink-500 text-xs uppercase tracking-wide">Model</th>
                  <th className="px-5 py-3 font-medium text-ink-400 light:text-ink-500 text-xs uppercase tracking-wide text-right">Risk</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={r.id ?? i} className="border-b border-ink-800/60 light:border-ink-100 last:border-0">
                    <td className="px-5 py-3 data-readout text-xs text-ink-400 light:text-ink-500">
                      {r.timestamp ?? r.created_at ?? '—'}
                    </td>
                    <td className="px-5 py-3 text-ink-200 light:text-ink-700 capitalize">{r.condition ?? r.disease ?? '—'}</td>
                    <td className="px-5 py-3 text-ink-400 light:text-ink-500">{r.model_used ?? '—'}</td>
                    <td className="px-5 py-3 text-right">{riskBadge(Number(r.risk_score ?? 0))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        )}
      </div>
    </div>
  );
}
