import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList,
} from 'recharts';
import { GitCompare, Loader2, AlertTriangle, Info } from 'lucide-react';
import { Card, Badge } from '../components/ui';
import { api, ApiError } from '../lib/api';

const TICK = { fill: 'var(--color-ink-400)', fontSize: 11, fontFamily: 'var(--font-data)' };
const LABEL = { fontSize: 11, fontFamily: 'var(--font-data)' };

const CONDITIONS = [
  { key: 'heart', label: 'Heart disease' },
  { key: 'diabetes', label: 'Diabetes' },
];

function niceAxisMax(max) {
  const padded = max + 9;
  const rounded = Math.ceil(padded / 5) * 5;
  return Math.min(100, rounded);
}

const INSIDE_GAP_THRESHOLD = 10;

const FAMILY = {
  'Logistic Regression': 'baseline',
  'Random Forest': 'tree',
  XGBoost: 'tree',
  LightGBM: 'tree',
  CatBoost: 'tree',
  Stacking: 'tree',
  MLP: 'deep',
  'FT-Transformer': 'transformer',
  TabTransformer: 'transformer',
  TabNet: 'transformer',
  TabPFN: 'foundation',
};

const FAMILY_META = {
  baseline: { label: 'Baseline', fill: 'var(--color-chart-gray)' },
  tree: { label: 'Tree ensemble', fill: 'var(--color-chart-blue)' },
  deep: { label: 'Deep learning', fill: 'var(--color-chart-violet)' },
  transformer: { label: 'Transformer', fill: 'var(--color-chart-magenta)' },
  foundation: { label: 'Pretrained foundation model', fill: 'var(--color-chart-green)' },
};

const BEST_FILL = 'var(--color-vital-400)';

const LEGEND_ITEMS = [{ label: 'Best model', fill: BEST_FILL }, ...Object.values(FAMILY_META)];

const INSIDE_LABEL_TEXT = {
  [BEST_FILL]: 'var(--color-ink-950)',
  'var(--color-chart-gray)': 'var(--color-ink-50)',
  'var(--color-chart-blue)': 'var(--color-ink-950)',
  'var(--color-chart-violet)': 'var(--color-ink-50)',
  'var(--color-chart-magenta)': 'var(--color-ink-950)',
  'var(--color-chart-green)': 'var(--color-ink-50)',
};

function barFill(name, bestModel) {
  if (name === bestModel) return BEST_FILL;
  return FAMILY_META[FAMILY[name]].fill;
}

function labelFill(barColor) {
  return INSIDE_LABEL_TEXT[barColor] ?? 'var(--color-ink-50)';
}

function ChartTooltip({ active, payload, bestModel }) {
  if (!active || !payload?.length) return null;
  const { name, roc_auc } = payload[0].payload;
  const fill = barFill(name, bestModel);
  return (
    <div
      style={{
        background: 'var(--color-ink-900)',
        border: '1px solid var(--color-ink-700)',
        borderRadius: 8,
        padding: '8px 12px',
        fontSize: 12,
      }}
    >
      <p style={{ margin: 0, color: 'var(--color-ink-50)', fontWeight: 600 }}>{name}</p>
      <p style={{ margin: '4px 0 0', color: fill }}>{roc_auc.toFixed(2)}% ROC-AUC</p>
    </div>
  );
}

export default function ModelComparison() {
  const [data, setData] = useState({ heart: null, diabetes: null });
  const [active, setActive] = useState('heart');
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([api.models('heart'), api.models('diabetes')])
      .then(([heart, diabetes]) => setData({ heart, diabetes }))
      .catch((err) =>
        setError(err instanceof ApiError ? err.message : 'Could not load the model comparison.')
      );
  }, []);

  const current = data[active];
  const chartData =
    current &&
    Object.entries(current.results)
      .map(([name, m]) => ({ name, roc_auc: m.roc_auc }))
      .sort((a, b) => b.roc_auc - a.roc_auc);
  const axisMax = chartData && niceAxisMax(chartData[0].roc_auc);

  return (
    <div className="max-w-4xl mx-auto px-5 sm:px-8 py-12">
      <div className="flex items-center gap-2.5 mb-2">
        <GitCompare size={16} className="text-vital-400" />
        <span className="text-xs font-semibold tracking-wide uppercase text-vital-400">
          11-model benchmark
        </span>
      </div>
      <h1 className="font-display font-semibold text-3xl text-ink-50 light:text-ink-900">
        Model comparison
      </h1>
      <p className="text-ink-400 light:text-ink-500 mt-2 max-w-2xl">
        Every model trained during the pipeline, ranked by ROC-AUC on the held-out test split.
        Toggle between the heart and diabetes benchmarks below.
      </p>

      <div
        role="tablist"
        aria-label="Condition"
        className="inline-flex items-center gap-1 mt-8 p-1 rounded-lg bg-ink-900/60 light:bg-ink-100 border border-ink-800 light:border-ink-200"
      >
        {CONDITIONS.map((c) => (
          <button
            key={c.key}
            role="tab"
            aria-selected={active === c.key}
            onClick={() => setActive(c.key)}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${
              active === c.key
                ? 'bg-vital-500 text-ink-950'
                : 'text-ink-300 light:text-ink-500 hover:text-ink-100 light:hover:text-ink-900'
            }`}
          >
            {c.label}
          </button>
        ))}
      </div>

      <div className="mt-6">
        {error && (
          <Card className="p-4 border-signal-500/30 bg-signal-500/[0.04] flex gap-3 items-start">
            <AlertTriangle size={16} className="text-signal-400 shrink-0 mt-0.5" />
            <p className="text-sm text-signal-300">{error}</p>
          </Card>
        )}

        {!error && !chartData && (
          <div className="flex items-center gap-2.5 text-ink-400 py-16 justify-center">
            <Loader2 size={16} className="animate-spin" /> Loading benchmark…
          </div>
        )}

        {!error && chartData && (
          <Card className="p-6">
            <div className="flex items-center justify-between gap-3 mb-4">
              <p className="text-xs font-semibold text-ink-400 light:text-ink-500">
                ROC-AUC by model — {CONDITIONS.find((c) => c.key === active).label}
              </p>
              <Badge tone="vital">Best: {current.best_model}</Badge>
            </div>

            <ResponsiveContainer width="100%" height={420}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 48 }}>
                <XAxis
                  type="number"
                  domain={[0, axisMax]}
                  tick={TICK}
                  axisLine={false}
                  tickLine={false}
                  unit="%"
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ ...TICK, fontFamily: 'var(--font-body)' }}
                  width={120}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  content={<ChartTooltip bestModel={current.best_model} />}
                  cursor={{ fill: 'var(--color-ink-800)', opacity: 0.4 }}
                />
                <Bar dataKey="roc_auc" radius={[0, 6, 6, 0]}>
                  {chartData.map((d) => (
                    <Cell key={d.name} fill={barFill(d.name, current.best_model)} />
                  ))}
                  <LabelList
                    dataKey="roc_auc"
                    content={({ x, y, width, height, value, index }) => {
                      const fill = barFill(chartData[index].name, current.best_model);
                      const inside = axisMax - value < INSIDE_GAP_THRESHOLD;
                      const textY = y + height / 2;
                      return (
                        <text
                          x={inside ? x + width - 8 : x + width + 6}
                          y={textY}
                          textAnchor={inside ? 'end' : 'start'}
                          dominantBaseline="central"
                          style={{ ...LABEL, fill: inside ? labelFill(fill) : 'var(--color-ink-300)' }}
                        >
                          {value.toFixed(2)}%
                        </text>
                      );
                    }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>

            <div className="mt-5 pt-4 border-t border-ink-800 light:border-ink-100 flex flex-wrap items-center gap-x-5 gap-y-2">
              {LEGEND_ITEMS.map((item) => (
                <div key={item.label} className="flex items-center gap-1.5 text-xs text-ink-400 light:text-ink-500">
                  <span
                    className="w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ background: item.fill }}
                    aria-hidden="true"
                  />
                  {item.label}
                </div>
              ))}
            </div>

            {active === 'diabetes' && (
              <div className="mt-5 flex gap-3 items-start p-4 rounded-lg bg-signal-500/[0.04] border border-signal-500/20">
                <Info size={15} className="text-signal-400 shrink-0 mt-0.5" />
                <p className="text-sm text-ink-300 light:text-ink-600 leading-relaxed">
                  <span className="font-medium text-signal-300">Why TabTransformer lags here: </span>
                  its attention mechanism only operates over categorical tokens, and after
                  encoding, the diabetes schema (Pima Indians dataset) has just one categorical
                  column (Pregnancies) — everything else is continuous. With almost nothing for
                  attention to attend over, the architecture is starved of the input it's built
                  around. Heart disease encodes several categorical fields (sex, chest pain type,
                  resting ECG, and more) into one-hot columns, giving the same architecture far
                  more to work with — which is why it's competitive there instead.
                </p>
              </div>
            )}
          </Card>
        )}
      </div>
    </div>
  );
}
