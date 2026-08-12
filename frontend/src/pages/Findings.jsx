import { useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion, useInView, useReducedMotion } from 'framer-motion';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
} from 'recharts';
import {
  Activity, FileText, Network, AlertOctagon, Layers, Lightbulb, HeartPulse, ArrowRight, Scale,
  AlertTriangle,
} from 'lucide-react';
import { Card, Badge } from '../components/ui';
import Waveform from '../components/Waveform';
import { useScrollSpy } from '../hooks/useScrollSpy';
import { useCountUp } from '../hooks/useCountUp';

const NAV_SECTIONS = [
  { id: 'overview', label: 'Overview' },
  { id: 'structured-records', label: 'Structured records' },
  { id: 'ecg', label: 'ECG classification' },
  { id: 'nlp', label: 'Symptom text' },
  { id: 'cross-dataset', label: 'Cross-dataset validation' },
  { id: 'contamination', label: 'Dataset provenance finding' },
  { id: 'synthesis', label: 'Synthesis' },
];
const NAV_IDS = NAV_SECTIONS.map((s) => s.id);

const TICK = { fill: 'var(--color-ink-400)', fontSize: 11, fontFamily: 'var(--font-data)' };
const GRID = 'var(--color-ink-800)';
const CURSOR = { fill: 'var(--color-ink-800)', opacity: 0.4 };

const ECG_DATA = [
  { name: 'CNN', macroF1: 0.8776, params: 343397, color: 'var(--color-vital-400)' },
  { name: 'CNN-BiLSTM', macroF1: 0.866, params: 184325, color: 'var(--color-ink-500)' },
  { name: 'BiLSTM (converged)', macroF1: 0.7464, params: 141189, color: 'var(--color-ink-500)' },
];

const NLP_FILL = 'var(--color-vital-400)';
const NLP_DATA = [
  { name: 'TF-IDF + SVM', macroF1: 0.9675, color: NLP_FILL },
  { name: 'DistilBERT', macroF1: 0.9635, color: NLP_FILL },
  { name: 'TF-IDF + LogReg', macroF1: 0.9626, color: NLP_FILL },
  { name: 'Frozen DistilBERT', macroF1: 0.945, color: NLP_FILL },
];

const LEARNING_CURVE = [
  { n: 80, tfidf: 0.7075, bert: 0.6897 },
  { n: 201, tfidf: 0.8573, bert: 0.8563 },
  { n: 403, tfidf: 0.9324, bert: 0.9331 },
  { n: 807, tfidf: 0.9516, bert: 0.9635 },
];
const LEARNING_SERIES = {
  tfidf: { label: 'TF-IDF', color: 'var(--color-ink-400)' },
  bert: { label: 'DistilBERT', color: 'var(--color-vital-400)' },
};

const XVAL_DATA = [
  { model: 'TabPFN', internal: 0.7627, external: 0.6342 },
  { model: 'LogReg', internal: 0.7606, external: 0.6276 },
  { model: 'CatBoost', internal: 0.7481, external: 0.6222 },
  { model: 'XGBoost', internal: 0.7411, external: 0.6133 },
  { model: 'LightGBM', internal: 0.7208, external: 0.6019 },
  { model: 'Random Forest', internal: 0.6994, external: 0.6103 },
];
const XVAL_SERIES = {
  internal: { label: 'Internal (CV)', color: 'var(--color-ink-500)' },
  external: { label: 'External (independent cohort)', color: 'var(--color-signal-500)' },
};

// Closing synthesis: the project's simplest winning model against its most
// complex, on four axes that determine whether a model is actually worth
// using, not just how it scores. Every value is derived from numbers
// already established above, not new measurements:
//   - performance: macro F1 on the full symptom-text benchmark (NLP_DATA).
//   - training speed: inverse wall-clock time, normalized to the faster
//     model. Derived from the "~1,600x faster" claim in the NLP section —
//     DistilBERT's relative speed is 1/1600.
//   - interpretability: a qualitative 0-1 judgement, not a measurement. A
//     linear model over transparent bag-of-words features scores high; a
//     67M-parameter transformer, legible only through post-hoc
//     approximation, scores low.
//   - data efficiency: inverse of the training-set size each model needs
//     to reach 90% of its own ceiling macro F1, read off LEARNING_CURVE.
//     TF-IDF crosses that bar at n=201 (0.8573 vs a 0.85644 threshold);
//     DistilBERT needs n=403 (0.9331 vs a 0.86715 threshold) — roughly
//     twice the data for the same relative milestone.
const RADAR_DATA = [
  { axis: 'Performance', simple: 0.9675, complex: 0.9635 },
  { axis: 'Training speed', simple: 1, complex: 1 / 1600 },
  { axis: 'Interpretability', simple: 0.9, complex: 0.2 },
  { axis: 'Data efficiency', simple: 1, complex: 201 / 403 },
];
const RADAR_SERIES = {
  simple: { label: 'TF-IDF + Linear SVM', color: 'var(--color-vital-400)' },
  complex: { label: 'DistilBERT (fine-tuned)', color: 'var(--color-ink-500)' },
};

// ink-500 reads fine as a bar fill against the page background, but at
// 12px tooltip-text size its contrast against the tooltip's own fixed
// dark background (ink-900) drops to ~2.3:1 — below a legible floor.
// Swap to the lighter ink-300 for tooltip text only; charts keep ink-500.
const TOOLTIP_TEXT_OVERRIDE = { 'var(--color-ink-500)': 'var(--color-ink-300)' };
const tooltipTextColor = (fill) => TOOLTIP_TEXT_OVERRIDE[fill] ?? fill;

// Recharts derives each tooltip entry's text color from the originating
// <Bar>/<Line> element's own `fill`/`stroke` prop (see ModelComparison.jsx
// for the full diagnosis). None of the charts on this page set that prop —
// colors are painted per-Cell or per-series instead — so Recharts falls
// back to itemStyle's hardcoded '#000', invisible against the tooltip's
// dark background. This shared content component sidesteps the inference
// entirely: the label is always pinned to ink-50, and each value's color
// comes from an explicit `seriesInfo` resolver reading the same color the
// chart itself was painted with, never from Recharts' own state.
function ChartTooltip({ active, payload, labelKey, labelFormatter, seriesInfo, valueFormatter = (v) => v.toFixed(4) }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
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
      {labelKey && (
        <p style={{ margin: 0, color: 'var(--color-ink-50)', fontWeight: 600 }}>
          {labelFormatter ? labelFormatter(row[labelKey]) : row[labelKey]}
        </p>
      )}
      {payload.map((entry) => {
        const { label, color } = seriesInfo(entry, row);
        return (
          <p key={entry.dataKey} style={{ margin: labelKey ? '4px 0 0' : 0, color: tooltipTextColor(color) }}>
            {label ? `${label}: ` : ''}
            {valueFormatter(entry.value)}
          </p>
        );
      })}
    </div>
  );
}

// Plain-English glossary for the jargon this page can't avoid using. Keyed
// by the exact display text so <Term> can look itself up without a
// separate `term` prop in the common case; pass `term` explicitly when the
// text on the page (e.g. a plural or different casing) doesn't match the
// canonical key.
const GLOSSARY = {
  'macro F1': 'A single accuracy score averaged equally across every class, so a model can’t hide poor performance on a rare class behind strong performance on a common one.',
  'ROC-AUC': 'A 0–1 score for how well a model ranks true cases above false ones across every possible decision threshold. 0.5 is a coin flip; 1.0 is perfect separation.',
  epoch: 'One full pass through the entire training dataset. Neural networks are trained over many epochs, adjusting their internal weights a little each time.',
  SHAP: 'A method that breaks a single prediction down feature by feature, showing exactly how much each input pushed the result up or down — the "why" behind one score.',
  convergence: 'The point in training where further passes over the data stop meaningfully improving the model — the point of diminishing returns.',
  'Spearman correlation': 'A 0–1 score for how consistently two rankings agree with each other, regardless of the exact numbers behind them.',
  'external validation': 'Testing a model on data from a source it never trained on, to check whether its performance generalises beyond its own dataset.',
};

function Term({ children, term }) {
  const [open, setOpen] = useState(false);
  const definition = GLOSSARY[term ?? children];
  if (!definition) return children;
  return (
    <span className="relative inline-block group">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        onBlur={() => setOpen(false)}
        title={definition}
        aria-expanded={open}
        className="border-b border-dotted border-ink-400 hover:border-vital-400 focus-visible:border-vital-400 transition-colors cursor-help"
      >
        {children}
      </button>
      <span
        role="tooltip"
        className={`pointer-events-none absolute z-20 left-1/2 -translate-x-1/2 bottom-full mb-2 w-56 rounded-lg border border-ink-700 bg-ink-900 p-2.5 text-xs leading-relaxed text-ink-200 normal-case font-normal shadow-lg transition-opacity duration-150 group-hover:opacity-100 ${
          open ? 'opacity-100' : 'opacity-0'
        }`}
      >
        {definition}
      </span>
    </span>
  );
}

// One-sentence, jargon-free restatement of a section's finding, shown
// before the technical paragraph rather than instead of it — for a reader
// who wants the headline without the metrics.
function PlainEnglish({ children }) {
  return (
    <div className="mb-5 flex gap-3 items-start p-3.5 rounded-lg bg-vital-500/[0.05] border border-vital-500/15">
      <Lightbulb size={15} className="text-vital-400 shrink-0 mt-0.5" />
      <p className="text-sm text-ink-200 light:text-ink-700 leading-relaxed">
        <span className="font-semibold text-vital-400">Plain-language summary: </span>
        {children}
      </p>
    </div>
  );
}

function SectionNav({ activeId }) {
  return (
    <>
      <nav aria-label="Findings sections" className="hidden lg:block sticky top-24 self-start">
        <ul className="space-y-1 border-l border-ink-800 light:border-ink-100">
          {NAV_SECTIONS.map((s) => (
            <li key={s.id}>
              <a
                href={`#${s.id}`}
                aria-current={activeId === s.id ? 'true' : undefined}
                className={`block -ml-px pl-4 py-1.5 text-sm border-l-2 transition-colors ${
                  activeId === s.id
                    ? 'border-vital-400 text-vital-400 font-medium'
                    : 'border-transparent text-ink-400 light:text-ink-500 hover:text-ink-200 light:hover:text-ink-800'
                }`}
              >
                {s.label}
              </a>
            </li>
          ))}
        </ul>
      </nav>

      <nav aria-label="Findings sections" className="lg:hidden -mx-5 sm:-mx-8 px-5 sm:px-8 mb-8 overflow-x-auto">
        <ul className="flex gap-2 w-max">
          {NAV_SECTIONS.map((s) => (
            <li key={s.id}>
              <a
                href={`#${s.id}`}
                aria-current={activeId === s.id ? 'true' : undefined}
                className={`inline-block whitespace-nowrap px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                  activeId === s.id
                    ? 'border-vital-500/40 bg-vital-500/10 text-vital-400'
                    : 'border-ink-800 light:border-ink-200 text-ink-400 light:text-ink-500'
                }`}
              >
                {s.label}
              </a>
            </li>
          ))}
        </ul>
      </nav>
    </>
  );
}

function Section({ id, icon: Icon, eyebrow, phase, title, children }) {
  return (
    <motion.section
      id={id}
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.15 }}
      transition={{ duration: 0.45, ease: 'easeOut' }}
      className="scroll-mt-24 py-16 border-t border-ink-800 light:border-ink-100 first:border-t-0 first:pt-0"
    >
      <div className="flex items-center gap-2.5 mb-3 flex-wrap">
        <Icon size={16} className="text-vital-400" />
        <span className="text-xs font-semibold tracking-wide uppercase text-vital-400">{eyebrow}</span>
        {phase && <Badge tone="neutral">{phase}</Badge>}
      </div>
      <h2 className="font-display font-semibold text-2xl sm:text-3xl text-ink-50 light:text-ink-900 mb-6">
        {title}
      </h2>
      {children}
    </motion.section>
  );
}

// A stat tile that counts up from 0 once it scrolls into view, reusing the
// same useCountUp hook RiskGauge uses for its percentage readout. The
// target itself stays 0 until `inView` flips, which — since useCountUp
// re-runs its animation whenever `target` changes — is what defers the
// count-up until the reveal, rather than animating immediately on mount.
function StatCard({ value, prefix = '', suffix = '', decimals = 0, label, tone = 'text-ink-50 light:text-ink-900' }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, amount: 0.4 });
  const prefersReducedMotion = useReducedMotion();
  const display = useCountUp(inView ? value : 0, !prefersReducedMotion);
  return (
    <div ref={ref}>
      <Card className="p-4 text-center h-full">
        <div className={`data-readout text-2xl font-semibold ${tone}`}>
          {prefix}
          {display.toFixed(decimals)}
          {suffix}
        </div>
        <div className="text-xs text-ink-400 light:text-ink-500 mt-1.5 leading-snug">{label}</div>
      </Card>
    </div>
  );
}

// The hero's right side has genuine empty width on desktop (its text column
// is capped at max-w-2xl inside a wider max-w-6xl container) — fill it with
// a small on-load animation of the same four phases and icons used by the
// sections below, rather than generic decoration. Purely illustrative: the
// four-phase explainer box underneath already carries this same information
// as accessible text, so the diagram is hidden from assistive tech.
const PHASE_FLOW = [
  { icon: HeartPulse, label: 'Tabular' },
  { icon: Activity, label: 'ECG' },
  { icon: FileText, label: 'Text' },
  { icon: Network, label: 'Cross-dataset' },
];

function PhaseFlowDiagram({ className = '' }) {
  return (
    <div className={`flex items-start ${className}`} aria-hidden="true">
      {PHASE_FLOW.map((phase, i) => (
        <div key={phase.label} className="flex items-start">
          <div className="flex flex-col items-center gap-2 w-16">
            <motion.div
              initial={{ opacity: 0, scale: 0.6 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: 0.15 + i * 0.18, ease: 'easeOut' }}
              className="flex items-center justify-center size-11 rounded-full border border-vital-500/30 bg-vital-500/[0.06] shrink-0"
            >
              <phase.icon size={18} className="text-vital-400" />
            </motion.div>
            <span className="text-[10px] font-medium text-ink-400 light:text-ink-500 whitespace-nowrap">
              {phase.label}
            </span>
          </div>
          {i < PHASE_FLOW.length - 1 && (
            <motion.div
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              transition={{ duration: 0.3, delay: 0.3 + i * 0.18, ease: 'easeOut' }}
              style={{ originX: 0 }}
              className="w-6 sm:w-8 h-px bg-ink-700 light:bg-ink-200 mt-[22px]"
            />
          )}
        </div>
      ))}
    </div>
  );
}

export default function Findings() {
  const activeId = useScrollSpy(NAV_IDS);

  return (
    <div className="max-w-6xl mx-auto px-5 sm:px-8 py-12">
      <div id="overview" className="scroll-mt-24">
        <div className="lg:flex lg:items-center lg:justify-between lg:gap-10">
          <div className="max-w-2xl">
            <Badge tone="vital" className="mb-4">Research findings</Badge>
            <h1 className="font-display font-semibold text-3xl sm:text-4xl text-ink-50 light:text-ink-900">
              Three data modalities, one question:
              <span className="text-vital-400"> does model complexity earn its cost?</span>
            </h1>
            <p className="mt-4 text-ink-400 light:text-ink-500 leading-relaxed">
              Structured records, ECG waveforms, and free-text symptom descriptions were each modelled
              with the most sophisticated architecture available and compared against a simpler
              baseline under identical conditions. Across all three, the more complex model won once.
            </p>
          </div>
          <PhaseFlowDiagram className="hidden lg:flex shrink-0 mt-2" />
        </div>

        <div className="mt-6 max-w-2xl flex gap-3 items-start p-4 rounded-lg bg-vital-500/[0.04] border border-vital-500/20">
          <Layers size={15} className="text-vital-400 shrink-0 mt-0.5" />
          <p className="text-sm text-ink-300 light:text-ink-600 leading-relaxed">
            This investigation runs in four phases.{' '}
            <span className="font-medium text-ink-100 light:text-ink-800">Phase 1</span> — structured
            tabular records for heart disease and diabetes — lives on the{' '}
            <Link to="/predict/heart" className="text-vital-400 hover:underline">
              Predict
            </Link>{' '}
            pages and the full{' '}
            <Link to="/models" className="text-vital-400 hover:underline">
              model comparison
            </Link>
            . This page picks up from Phase 2 onward: ECG waveforms, free-text symptoms, and
            cross-dataset validation.
          </p>
        </div>

        <div className="mt-6 max-w-2xl">
          <h2 className="text-sm font-semibold text-ink-100 light:text-ink-800 mb-2">How to read this page</h2>
          <p className="text-sm text-ink-400 light:text-ink-500 leading-relaxed">
            A &ldquo;model,&rdquo; in every section below, is simply a program trained on past patient
            records to make a prediction &mdash; a risk score, a heartbeat category, a symptom label
            &mdash; on a case it has never seen before. Several models are always compared rather than
            just building one, because there is no way to know in advance which approach will actually
            perform best on a given kind of data; a model&rsquo;s size or how advanced it sounds is not
            a reliable predictor of how well it works. Throughout this page, &ldquo;winning&rdquo; means
            exactly one thing: scoring better on data the model was not trained on &mdash; not running
            faster, not looking more sophisticated, not having more parameters. As you scroll, each
            section follows the same pattern: a plain-language summary first, a plot of the actual
            results, then a short technical paragraph for anyone who wants the detail. Any underlined
            term along the way can be tapped or hovered for a definition, so every technical detail is
            optional, not required, to follow the argument.
          </p>
        </div>
      </div>

      <div className="mt-10 lg:grid lg:grid-cols-[200px_1fr] lg:gap-12">
        <SectionNav activeId={activeId} />

        <div className="max-w-3xl min-w-0">
          <Section id="structured-records" icon={HeartPulse} eyebrow="Structured risk prediction" phase="Phase 1" title="Tabular heart &amp; diabetes risk models">
            <PlainEnglish>
              before touching ECGs or symptom text, the project first benchmarked eleven AI models on
              standard patient records — the same models behind the live prediction tool — and
              explained every one of their predictions.
            </PlainEnglish>
            <p className="text-ink-400 light:text-ink-500 mb-6 leading-relaxed">
              Eleven models — from a logistic regression baseline through gradient-boosted trees, a
              deep neural network, transformer architectures, and TabPFN, a pretrained tabular
              foundation model — were tuned and benchmarked on heart disease and diabetes records
              under identical cross-validation splits. Every prediction is explained with{' '}
              <Term>SHAP</Term>, breaking a single risk score down into the features that drove it.
              This is Phase 1 of the investigation; the full model-by-model comparison lives on its
              own page.
            </p>
            <div className="grid sm:grid-cols-2 gap-4 mb-4">
              <StatCard value={11} decimals={0} label="models benchmarked per condition" />
              <StatCard value={0.9584} decimals={4} label="best ROC-AUC · heart disease (TabPFN)" />
            </div>
            <Link to="/models" className="inline-flex items-center gap-1.5 text-sm font-semibold text-vital-400 hover:gap-2.5 transition-all">
              See the full comparison <ArrowRight size={14} />
            </Link>
          </Section>

          <Section id="ecg" icon={Activity} eyebrow="ECG rhythm classification" phase="Phase 2" title="ECG arrhythmia classification">
            <PlainEnglish>
              we tested three different AI designs for reading heartbeat patterns, and the simplest,
              most direct design won.
            </PlainEnglish>
            <p className="text-ink-400 light:text-ink-500 mb-6 leading-relaxed">
              109,446 heartbeats from MIT-BIH, three architectures under an identical training budget.
              A <Term>convergence</Term> check on the BiLSTM &mdash; 40 <Term term="epoch">epochs</Term> was
              insufficient &mdash; raised its <Term>macro F1</Term> from 0.7187 to 0.7464 and confirmed the
              gap to the CNN is architectural, not a training-budget artefact.
            </p>
            <Card className="p-6">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={ECG_DATA} margin={{ left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
                  <XAxis dataKey="name" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
                  <YAxis domain={[0, 1]} tick={TICK} axisLine={false} tickLine={false} />
                  <Tooltip
                    cursor={CURSOR}
                    content={<ChartTooltip labelKey="name" seriesInfo={(entry, row) => ({ label: null, color: row.color })} />}
                  />
                  <Bar dataKey="macroF1" radius={[6, 6, 0, 0]}>
                    {ECG_DATA.map((d) => (
                      <Cell key={d.name} fill={d.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <Waveform repeats={14} height={36} strokeWidth={1.5} animate={false} className="opacity-30 mt-4" />
            </Card>
            <div className="mt-5 flex gap-3 items-start p-4 rounded-lg bg-amber-500/[0.04] border border-amber-500/20">
              <AlertTriangle size={15} className="text-amber-400 shrink-0 mt-0.5" />
              <p className="text-sm text-ink-300 light:text-ink-600 leading-relaxed">
                <span className="font-medium text-amber-300">A methodology caveat: </span>
                this benchmark splits heartbeats randomly across train and test, so beats from the
                same patient can appear on both sides of the split. Under a stricter
                patient-disjoint protocol (de Chazal et al., 2004), reported accuracy on this
                dataset typically falls from around 98% into the low 80s &mdash; a gap this
                project has not yet closed with its own patient-level re-run.
              </p>
            </div>
          </Section>

          <Section id="nlp" icon={FileText} eyebrow="Symptom text classification" phase="Phase 3" title="Free-text symptom classification">
            <PlainEnglish>
              a fast, simple text-matching model read symptom descriptions just as accurately as a
              much larger AI language model — while training roughly 1,600&times; faster.
            </PlainEnglish>
            <p className="text-ink-400 light:text-ink-500 mb-6 leading-relaxed">
              A linear SVM on TF-IDF features matched a fine-tuned 67M-parameter DistilBERT at roughly
              1,600&times; faster training. The learning curve below shows the two staying within one
              standard deviation of each other at every training size tested.
            </p>
            <div className="grid sm:grid-cols-2 gap-5">
              <Card className="p-6">
                <p className="text-xs font-semibold text-ink-400 light:text-ink-500 mb-4">
                  <Term>macro F1</Term> by model
                </p>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={NLP_DATA} layout="vertical" margin={{ left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
                    <XAxis type="number" domain={[0.9, 1]} tick={TICK} axisLine={false} tickLine={false} />
                    <YAxis type="category" dataKey="name" tick={{ ...TICK, fontFamily: 'var(--font-body)' }} width={110} axisLine={false} tickLine={false} />
                    <Tooltip
                      cursor={CURSOR}
                      content={<ChartTooltip labelKey="name" seriesInfo={() => ({ label: null, color: NLP_FILL })} />}
                    />
                    <Bar dataKey="macroF1" radius={[0, 6, 6, 0]} fill={NLP_FILL} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
              <Card className="p-6">
                <p className="text-xs font-semibold text-ink-400 light:text-ink-500 mb-4">
                  Learning curve (test <Term>macro F1</Term>)
                </p>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={LEARNING_CURVE}>
                    <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
                    <XAxis dataKey="n" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
                    <YAxis domain={[0.6, 1]} tick={TICK} axisLine={false} tickLine={false} />
                    <Tooltip
                      content={
                        <ChartTooltip
                          labelKey="n"
                          labelFormatter={(n) => `n = ${n} examples`}
                          seriesInfo={(entry) => LEARNING_SERIES[entry.dataKey]}
                        />
                      }
                    />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line type="monotone" dataKey="tfidf" name={LEARNING_SERIES.tfidf.label} stroke={LEARNING_SERIES.tfidf.color} strokeWidth={2} dot={{ r: 3 }} />
                    <Line type="monotone" dataKey="bert" name={LEARNING_SERIES.bert.label} stroke={LEARNING_SERIES.bert.color} strokeWidth={2} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </div>
          </Section>

          <Section id="cross-dataset" icon={Network} eyebrow="Cross-dataset validation" phase="Phase 4" title="Cross-dataset external validation">
            <PlainEnglish>
              every model was tested twice — once on data like what it trained on, and once on a
              completely different hospital&rsquo;s data (<Term term="external validation">external
              validation</Term>) — and how well a model ranked on the first test barely predicted how
              it ranked on the second.
            </PlainEnglish>
            <p className="text-ink-400 light:text-ink-500 mb-2 leading-relaxed">
              Every model above is validated internally. This phase asks the question that matters
              clinically: trained at one site, does it hold up at another? Heart disease (918 rows)
              and an independent cardiovascular cohort (70,000 rows) were harmonised onto a shared
              five-feature subset and tested in both directions.
            </p>
            <p className="text-sm font-semibold text-signal-400 mb-6">
              <Term term="Spearman correlation">Spearman rank correlation</Term> between internal and
              external performance: 0.943 one direction, 0.257 the other. In-sample ranking is not a
              reliable guide to real-world performance.
            </p>
            <Card className="p-6">
              <p className="text-xs font-semibold text-ink-400 light:text-ink-500 mb-4">
                Internal (5-fold CV) vs external (independent cohort) <Term>ROC-AUC</Term> — Heart &rarr; Cardio
              </p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={XVAL_DATA} margin={{ left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
                  <XAxis dataKey="model" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
                  <YAxis domain={[0.5, 0.85]} tick={TICK} axisLine={false} tickLine={false} />
                  <Tooltip
                    cursor={CURSOR}
                    content={<ChartTooltip labelKey="model" seriesInfo={(entry) => XVAL_SERIES[entry.dataKey]} />}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="internal" name={XVAL_SERIES.internal.label} fill={XVAL_SERIES.internal.color} radius={[6, 6, 0, 0]} />
                  <Bar dataKey="external" name={XVAL_SERIES.external.label} fill={XVAL_SERIES.external.color} radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </Section>

          <Section id="contamination" icon={AlertOctagon} eyebrow="Methodology" title="A dataset provenance failure">
            <PlainEnglish>
              a planned test turned out to be checking a dataset against a disguised copy of itself,
              not a genuinely separate one — so we built an automatic check that catches this kind of
              hidden duplication before it can produce a misleading result.
            </PlainEnglish>
            <p className="text-ink-400 light:text-ink-500 mb-6 leading-relaxed">
              A second external-validation experiment, between the Pima diabetes dataset and a
              distributed &ldquo;Frankfurt Hospital&rdquo; cohort, was aborted after row-level
              fingerprinting found that 95.6% of Pima's records also appear in the &ldquo;independent&rdquo;
              file &mdash; it is Pima resampled with replacement, not a separate cohort. The validation
              pipeline was then extended with an automatic independence gate that refuses to score
              any dataset pair sharing more than 5% of rows.
            </p>
            <div className="grid grid-cols-3 gap-4">
              <StatCard value={62.8} decimals={1} suffix="%" tone="text-signal-400" label='exact duplicate rows within the "Frankfurt" file' />
              <StatCard value={95.6} decimals={1} suffix="%" tone="text-signal-400" label='of Pima rows also found in "Frankfurt"' />
              <StatCard value={10} decimals={0} prefix="~" tone="text-signal-400" label="unique rows once duplicates are removed" />
            </div>
          </Section>

          <Section id="synthesis" icon={Scale} eyebrow="Synthesis" title="What the four phases add up to">
            <PlainEnglish>
              across every test in this project, a bigger, more complex AI model won only once —
              and even then, not by much. Simplicity was usually cheaper, faster, and just as accurate.
            </PlainEnglish>
            <p className="text-ink-400 light:text-ink-500 mb-6 leading-relaxed">
              The clearest single contrast in this project is the free-text symptom task: a linear
              SVM over TF-IDF features against a fine-tuned 67M-parameter DistilBERT. Plotting both
              across four independent dimensions, not just accuracy, makes the shape of the tradeoff
              visible at a glance &mdash; DistilBERT&rsquo;s edge, where it exists at all, is confined
              to raw performance, while the simpler model dominates every axis that determines
              whether a model is practical to train, explain, and deploy.
            </p>
            <Card className="p-6">
              <ResponsiveContainer width="100%" height={340}>
                <RadarChart data={RADAR_DATA} outerRadius="70%">
                  <PolarGrid stroke={GRID} />
                  <PolarAngleAxis dataKey="axis" tick={{ fill: 'var(--color-ink-300)', fontSize: 12, fontFamily: 'var(--font-body)' }} />
                  <PolarRadiusAxis domain={[0, 1]} tickCount={5} tick={{ ...TICK, fontSize: 9 }} axisLine={false} />
                  <Tooltip
                    content={
                      <ChartTooltip
                        labelKey="axis"
                        seriesInfo={(entry) => RADAR_SERIES[entry.dataKey]}
                        valueFormatter={(v) => v.toFixed(2)}
                      />
                    }
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Radar
                    name={RADAR_SERIES.simple.label}
                    dataKey="simple"
                    stroke={RADAR_SERIES.simple.color}
                    fill={RADAR_SERIES.simple.color}
                    fillOpacity={0.28}
                    strokeWidth={2}
                  />
                  <Radar
                    name={RADAR_SERIES.complex.label}
                    dataKey="complex"
                    stroke={RADAR_SERIES.complex.color}
                    fill={RADAR_SERIES.complex.color}
                    fillOpacity={0.18}
                    strokeWidth={2}
                  />
                </RadarChart>
              </ResponsiveContainer>
              <p className="text-xs text-ink-500 light:text-ink-400 mt-2 leading-relaxed">
                Performance is macro F1 on the full symptom-text benchmark. Training speed is inverse
                wall-clock time, normalized to the faster model &mdash; DistilBERT trains roughly
                1,600&times; slower, which is why its line is barely visible on that axis.
                Interpretability is a qualitative judgement: a linear model over transparent features
                scores high; a transformer legible only through post-hoc approximation scores low.
                Data efficiency comes from the learning curve above &mdash; TF-IDF+SVM reaches 90% of
                its ceiling <Term>macro F1</Term> at 201 training examples, DistilBERT needs 403.
              </p>
            </Card>
            <p className="text-ink-300 light:text-ink-600 mt-6 leading-relaxed">
              Zoom out across all four phases, and the answer to the question this page opened with
              is neither yes nor no &mdash; it depends on what, specifically, a model is being asked
              to do. On structured patient records, the most sophisticated option on offer, a
              pretrained foundation model, produced the best result. On ECG rhythm classification,
              the opposite happened: the simplest of three designs beat a more elaborate one
              outright. On free-text symptom classification, the simple model and the sophisticated
              one tied within noise, and the simple one got there roughly 1,600&times; faster. And
              when models trained at one hospital were tested on another&rsquo;s data, added
              sophistication bought no immunity &mdash; the model that ranked best on its own data
              was not reliably the one that ranked best on someone else&rsquo;s.{' '}
              <span className="font-semibold text-ink-50 light:text-ink-900">
                Complexity is not a fixed advantage; it is a cost that occasionally, not routinely,
                buys something back &mdash; and the only way to know if it did is to check, every
                time, against data the model has never seen.
              </span>
            </p>
          </Section>
        </div>
      </div>
    </div>
  );
}
