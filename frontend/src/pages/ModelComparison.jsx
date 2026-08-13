import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList,
} from 'recharts';
import { GitCompare, Loader2, AlertTriangle, ChevronDown } from 'lucide-react';
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

const FAMILY_GLOSSARY = [
  {
    ...FAMILY_META.baseline,
    what: 'A single logistic regression model — no ensembling, no nonlinear feature interactions, just a straight-line decision boundary.',
    why: "Included as a floor: if the far more expensive models here can't clearly beat it, that's evidence their added complexity isn't earning its keep.",
  },
  {
    ...FAMILY_META.tree,
    what: 'Random Forest, XGBoost, LightGBM, CatBoost, and a stacked ensemble of them — models built from many decision trees voting together.',
    why: 'These are the established workhorses for tabular data, and the ones the newer architectures below actually have to beat rather than being assumed better by default.',
  },
  {
    ...FAMILY_META.deep,
    what: 'A standard multi-layer perceptron — a plain feedforward neural network, with none of the tabular-specific architecture built into the transformers below.',
    why: 'Acts as a control: it isolates whether any gain from the transformer models comes from their attention mechanism specifically, or just from being a neural network at all.',
  },
  {
    ...FAMILY_META.transformer,
    what: 'Three separate architectures — FT-Transformer, TabTransformer, and TabNet — that adapt attention mechanisms built for language into tabular feature columns.',
    why: "Testing three designs instead of one matters here: a single architecture's result could easily be a quirk of that implementation rather than real evidence about tabular transformers in general.",
  },
  {
    ...FAMILY_META.foundation,
    what: 'TabPFN — a single model pretrained in advance on synthetic tabular data, applied here through in-context learning rather than training from scratch.',
    why: 'Small clinical datasets like these (768–918 rows) are exactly the regime foundation models are designed for; this checks whether that theoretical edge actually holds up.',
  },
];

const HEART_NOTES = {
  TabPFN:
    'TabPFN arrives pretrained on a huge library of synthetic tabular tasks, so it starts with strong general priors about tabular structure instead of learning everything from these 918 rows alone. That head start matters most on a dataset this small, which is why it edges out the field here.',
  LightGBM:
    "Like the other tree-based models clustered just behind TabPFN, LightGBM benefits from heart disease's features splitting cleanly along thresholds — chest pain type, ST slope, and similar fields suit the if-then splits trees are built around, consistent with research showing tree ensembles remain highly competitive on small tabular datasets. Its leaf-wise growth can find sharper individual splits, though at this dataset size the gap to its closest peers comes down to regularization choices more than a real edge.",
  CatBoost:
    "CatBoost sits in the same tightly-packed tree-ensemble group for the same underlying reason — threshold-friendly features suit trees generally at this data scale. Its ordered-boosting approach, built to reduce the prediction leakage boosted trees can pick up on smaller datasets, is a genuine distinguishing trait, but here it lands within noise of its closest peers rather than clearly ahead.",
  Stacking:
    "The stacked ensemble combines Random Forest, XGBoost, and LightGBM through a logistic-regression meta-learner, so its result is bounded by how much extra signal that combination can extract beyond what those base models already capture individually. At this dataset size, that keeps it inside the same closely-clustered tree-based group rather than clearly ahead of any one contributor.",
  'Random Forest':
    "Random Forest's bagged, independently-grown trees share the same threshold-splitting advantage as the boosted models here. Averaging many independent trees trades some of the fit boosting can achieve for lower variance, consistent with it sitting toward the lower end of the tree-ensemble cluster rather than the top.",
  XGBoost:
    "XGBoost's gradient-boosted trees, built with explicit regularization and second-order gradient information, suit heart disease's threshold-based feature relationships well, placing it comfortably inside the tightly-clustered tree-ensemble group. The remaining gap to its closest peers reflects differing regularization defaults more than any structural advantage.",
  MLP:
    'The plain multi-layer perceptron performs respectably but trails the top tier — deep networks typically need substantially more than 918 training rows before they reliably out-predict tree ensembles on tabular data, and this dataset falls well short of that.',
  'FT-Transformer':
    'FT-Transformer comes closest of the three new tabular-transformer architectures to the leading models, because it tokenizes every feature — continuous values included, not just categorical ones — giving its attention mechanism the full feature set to reason over rather than a partial view.',
  'Logistic Regression':
    'A plain logistic regression lands in a solid mid-table position because several of the strongest heart-disease risk relationships — age and cholesterol among them — are close to linear, so a model with no capacity for interactions still captures a meaningful share of the real signal.',
  TabTransformer:
    "TabTransformer sits mid-table here — heart disease encodes several genuine categorical fields (sex, chest pain type, resting ECG, and more), giving its attention mechanism, which only operates over categorical tokens, enough to work with. It still can't reason over continuous relationships as directly as FT-Transformer, which is why it trails it.",
  TabNet:
    'TabNet finishes last of the eleven. Its sequential attention-based feature-selection mechanism is designed to learn which features to focus on at each decision step, and that selection policy typically needs more training examples than 918 rows provide to converge on a genuinely useful strategy.',
};

const DIABETES_NOTES = {
  XGBoost:
    "XGBoost's gradient-boosted trees are well matched to diabetes's mostly-continuous features, which contain meaningful nonlinear thresholds — glucose cutoffs around the prediabetes and diabetes ranges being the clearest example — that boosting is specifically built to capture. That fit is enough to put it ahead of the rest of the field here.",
  'Random Forest':
    "Random Forest sits in the closely-clustered group just behind XGBoost for a similar reason — diabetes's continuous, threshold-driven features suit trees generally. Bagging independent trees trades some of the fit boosting can achieve for lower variance, consistent with it landing just behind the boosted models rather than ahead of them.",
  Stacking:
    "The stacked ensemble again combines Random Forest, XGBoost, and LightGBM through a logistic-regression meta-learner, inheriting the same tree-based advantage on diabetes's continuous features. Its position within the closely-packed group just behind XGBoost reflects how much extra signal that combination extracts beyond its own base models, plus ordinary noise at 768 rows.",
  TabPFN:
    "TabPFN's pretrained priors, learned across a large library of synthetic tabular tasks, again give it a real advantage on a dataset this small — but here that lands it among the closely-clustered group just behind XGBoost rather than at the top, reflecting a boosting-versus-meta-learned-priors tradeoff more than any clear gap between them.",
  CatBoost:
    "CatBoost's ordered-boosting approach, built to reduce the prediction leakage boosted trees can pick up on smaller datasets, is a genuine distinguishing trait, and it clusters closely with the other tree-based models here — the differences within this group reflect boosting-versus-bagging tradeoffs and ordinary noise at this dataset size rather than any one model being categorically better.",
  'FT-Transformer':
    "FT-Transformer again benefits from tokenizing continuous features directly rather than only categorical ones, which matters even more on diabetes than on heart disease — nearly every field here is continuous, so full-feature attention coverage counts for more.",
  TabNet:
    "TabNet lands mid-table for the same reason it trails on heart disease: its sequential attention-based feature-selection mechanism needs more training examples than it's given here to learn a reliable selection policy, and that limitation is slightly more pronounced on diabetes's 768 rows than on heart disease's 918.",
  LightGBM:
    "LightGBM's leaf-wise tree growth can find sharper individual splits than XGBoost's level-wise default, but that same flexibility makes it somewhat more prone to overfitting on a smaller dataset — a tradeoff that shows up here as a small step behind XGBoost rather than matching it.",
  'Logistic Regression':
    "Logistic regression performs reasonably because several of diabetes's strongest predictors — glucose, BMI, and age — have fairly monotonic relationships with risk, so a linear model captures a meaningful share of the real signal even without modelling interactions. What it misses is exactly the kind of interaction effects the tree-based models above it are built to exploit.",
  MLP:
    "The plain multi-layer perceptron trails the top tier noticeably here, for the same small-data reason it lags on heart disease — deep networks generally need more than a few hundred rows to reliably out-predict trees on tabular data — and that gap is more pronounced on diabetes's smaller, 768-row dataset.",
  TabTransformer:
    "TabTransformer finishes last here because its attention mechanism only operates over categorical tokens, and after encoding, diabetes's schema (the Pima Indians dataset) leaves it just one genuine categorical column — Pregnancies — with everything else continuous. With almost nothing for attention to attend over, the architecture is starved of the input it's built around; heart disease's several categorical fields give the same architecture far more to work with, which is why it's competitive there instead.",
};

const MODEL_NOTES = { heart: HEART_NOTES, diabetes: DIABETES_NOTES };

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
  const [notesOpen, setNotesOpen] = useState(false);

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

            <div className="mt-5 pt-4 border-t border-ink-800 light:border-ink-100">
              <button
                type="button"
                onClick={() => setNotesOpen((o) => !o)}
                aria-expanded={notesOpen}
                className="flex items-center gap-1.5 text-sm font-medium text-ink-200 light:text-ink-700 hover:text-vital-400 transition-colors"
              >
                <ChevronDown
                  size={15}
                  className={`transition-transform ${notesOpen ? 'rotate-180' : ''}`}
                />
                Why did each model rank where it did?
              </button>

              {notesOpen && (
                <div className="mt-4 space-y-3.5">
                  {chartData.map((d) => (
                    <div key={d.name} className="flex gap-3 items-start">
                      <span
                        className="w-2 h-2 mt-1.5 rounded-full shrink-0"
                        style={{ background: barFill(d.name, current.best_model) }}
                        aria-hidden="true"
                      />
                      <p className="text-sm text-ink-300 light:text-ink-600 leading-relaxed">
                        <span className="font-medium text-ink-100 light:text-ink-800">{d.name}: </span>
                        {MODEL_NOTES[active][d.name]}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Card>
        )}
      </div>

      <div className="mt-12">
        <span className="text-xs font-semibold tracking-wide uppercase text-vital-400">
          Reading the chart
        </span>
        <h2 className="font-display font-semibold text-2xl text-ink-50 light:text-ink-900 mt-2">
          What each color family means
        </h2>
        <p className="text-ink-400 light:text-ink-500 mt-2 mb-6 max-w-2xl">
          Every bar above is colored by one of five model families — what each represents, and why
          it's part of this comparison specifically.
        </p>
        <div className="grid sm:grid-cols-2 gap-4">
          {FAMILY_GLOSSARY.map((f) => (
            <Card key={f.label} className="p-5">
              <div className="flex items-center gap-2 mb-2.5">
                <span
                  className="w-2.5 h-2.5 rounded-full shrink-0"
                  style={{ background: f.fill }}
                  aria-hidden="true"
                />
                <h3 className="font-display font-semibold text-sm text-ink-50 light:text-ink-900">
                  {f.label}
                </h3>
              </div>
              <p className="text-sm text-ink-400 light:text-ink-500 leading-relaxed">{f.what}</p>
              <p className="text-sm text-ink-400 light:text-ink-500 leading-relaxed mt-2">{f.why}</p>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
