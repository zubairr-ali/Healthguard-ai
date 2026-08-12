import { useRef } from 'react';
import { Link } from 'react-router-dom';
import { motion, useInView, useReducedMotion } from 'framer-motion';
import {
  HeartPulse, Activity, FlaskConical, ArrowRight, ShieldCheck,
  Layers, ScanSearch, Network, Lock, ClipboardList, Gauge, CheckCircle2,
} from 'lucide-react';
import Waveform from '../components/Waveform';
import { Card, Badge } from '../components/ui';
import RiskGauge from '../components/RiskGauge';
import { useCountUp } from '../hooks/useCountUp';

const WHY = [
  {
    icon: Layers,
    title: 'Eleven models, not one',
    desc: 'Gradient boosting, ensembles, a deep neural network, three tabular transformer architectures, and a pretrained foundation model — tuned and compared rather than a single model taken on faith.',
  },
  {
    icon: ScanSearch,
    title: 'Every prediction explained',
    desc: 'SHAP breaks each result down feature by feature, so a risk score is never just a number — you see exactly what drove it, in either direction.',
  },
  {
    icon: Network,
    title: 'Tested beyond its own data',
    desc: 'Models are also validated on independent cohorts, reporting where accuracy holds up at a different site and where it quietly falls apart.',
  },
  {
    icon: Lock,
    title: 'Runs entirely on your machine',
    desc: 'No patient data leaves this app. Predictions and history are served locally by the FastAPI backend and stored in a local SQLite database.',
  },
];

const PHASES = [
  {
    icon: HeartPulse,
    title: 'Structured risk prediction',
    desc: 'Eleven tuned models — gradient boosting, ensembles, three tabular transformers, and a pretrained foundation model — compared on heart disease and diabetes records, each explained with SHAP.',
    to: '/predict/heart',
    cta: 'Run a prediction',
  },
  {
    icon: Activity,
    title: 'ECG arrhythmia classification',
    desc: 'A 1D CNN, a BiLSTM, and a hybrid architecture compared on 109,446 MIT-BIH heartbeats — with a convergence check that changed the conclusion.',
    to: '/findings#ecg',
    cta: 'See the comparison',
  },
  {
    icon: FlaskConical,
    title: 'Cross-dataset validation',
    desc: 'Every internal result re-tested on an independent cohort. The most striking finding: internal ranking failed to predict external ranking.',
    to: '/findings#cross-dataset',
    cta: 'See the finding',
  },
];

// Same connected-node visual grammar as Findings' hero PhaseFlowDiagram —
// icon circle, thin connector, staggered entrance — adapted for scroll
// triggering (this section sits well below the fold, so whileInView
// replaces the mount-time animation) and for a title/description per step
// rather than a single word label.
const HOW_IT_WORKS = [
  { icon: ClipboardList, title: 'Enter clinical values', desc: 'The same measurements a clinician would record, entered once.' },
  { icon: Layers, title: 'Compared across eleven models', desc: 'Every model scores the same input under identical conditions.' },
  { icon: ScanSearch, title: 'SHAP explains the result', desc: 'Feature by feature, showing exactly what pushed the score up or down.' },
  { icon: Gauge, title: 'Get a risk band, plainly explained', desc: 'A score paired with a plain-language advisory note, not a bare number.' },
];

function HowItWorks() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-x-4 gap-y-10">
      {HOW_IT_WORKS.map((step, i) => (
        <div key={step.title} className="relative">
          {i < HOW_IT_WORKS.length - 1 && (
            <motion.div
              initial={{ scaleX: 0 }}
              whileInView={{ scaleX: 1 }}
              viewport={{ once: true, amount: 0.6 }}
              transition={{ duration: 0.4, delay: 0.15 + i * 0.15, ease: 'easeOut' }}
              style={{ originX: 0, width: 'calc(100% + 1rem)' }}
              className="hidden lg:block absolute top-6 left-1/2 h-px bg-ink-700 light:bg-ink-200 z-0"
            />
          )}
          <motion.div
            initial={{ opacity: 0, scale: 0.7 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true, amount: 0.6 }}
            transition={{ duration: 0.35, delay: i * 0.15, ease: 'easeOut' }}
            className="relative z-10 mx-auto w-12 h-12 rounded-full border border-vital-500/30 bg-ink-950 light:bg-paper-50 flex items-center justify-center"
          >
            <step.icon size={19} className="text-vital-400" />
          </motion.div>
          <p className="mt-4 text-center text-[11px] font-semibold tracking-wide uppercase text-vital-400">
            Step {i + 1}
          </p>
          <h3 className="mt-1 text-center font-semibold text-sm text-ink-100 light:text-ink-800">
            {step.title}
          </h3>
          <p className="mt-1.5 text-center text-xs text-ink-400 light:text-ink-500 leading-relaxed max-w-[15rem] mx-auto">
            {step.desc}
          </p>
        </div>
      ))}
    </div>
  );
}

// Every figure here is checked against the codebase, not asserted: 26 is
// pytest's own passing count under backend/tests (`pytest -q` → "26
// passed"), 5 is the row count of the Datasets table in the project
// README, and the contamination figure is the Frankfurt-cohort finding
// documented on the Findings page. Reuses the same useCountUp hook as
// RiskGauge and Findings' StatCard, gated on scroll into view the same way.
const MANIFESTO = [
  'No single model taken on faith.',
  'Every prediction explained, feature by feature.',
  'Every claim tested on data it has never seen.',
  'Failures reported as honestly as successes.',
  'Built to be checked, not just believed.',
];

const TRUST_STATS = [
  { value: 26, label: 'automated backend tests, all passing' },
  { value: 5, label: 'independent datasets used across the project' },
  { value: 1, label: 'dataset-contamination bug caught and closed automatically' },
];

function CountStat({ value, label }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, amount: 0.5 });
  const prefersReducedMotion = useReducedMotion();
  const display = useCountUp(inView ? value : 0, !prefersReducedMotion);
  return (
    <div ref={ref} className="text-center">
      <div className="data-readout text-3xl sm:text-4xl font-semibold text-vital-400">
        {display.toFixed(0)}
      </div>
      <div className="text-xs text-ink-400 light:text-ink-500 mt-2 leading-snug max-w-[12rem] mx-auto">
        {label}
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <div>
      {/* ── Hero ───────────────────────────────────────────────── */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_50%_at_50%_0%,rgba(20,184,166,0.12),transparent)]" />
        <div className="max-w-6xl mx-auto px-5 sm:px-8 pt-16 sm:pt-24 pb-8 relative">
          <div className="lg:flex lg:items-center lg:justify-between lg:gap-12">
            <div className="text-center lg:text-left max-w-2xl mx-auto lg:mx-0">
              <Badge tone="vital" className="mb-6">
                <ShieldCheck size={13} /> Computer Science Project · 6WCM0029
              </Badge>
              <h1 className="font-display font-semibold text-4xl sm:text-6xl tracking-tight text-ink-50 light:text-ink-900 leading-[1.05]">
                Cutting through the noise in
                <br />
                <span className="text-vital-400">patient risk data.</span>
              </h1>
              <p className="mt-6 text-lg text-ink-300 light:text-ink-600 leading-relaxed">
                HealthGuard AI predicts heart disease and diabetes risk from clinical records,
                explains every prediction with SHAP, and stress-tests each model against
                independent cohorts — reporting where they hold up and where they don't.
              </p>
              <div className="mt-9 flex flex-wrap items-center justify-center lg:justify-start gap-3">
                <Link
                  to="/predict/heart"
                  className="inline-flex items-center gap-2 rounded-lg bg-vital-500 text-ink-950 px-6 py-3 text-sm font-semibold hover:bg-vital-400 transition-colors shadow-[0_0_0_1px_rgba(20,184,166,0.3),0_10px_24px_-8px_rgba(20,184,166,0.55)]"
                >
                  Try a live prediction <ArrowRight size={16} />
                </Link>
                <Link
                  to="/findings"
                  className="inline-flex items-center gap-2 rounded-lg border border-ink-700 light:border-ink-200 px-6 py-3 text-sm font-semibold text-ink-200 light:text-ink-700 hover:bg-ink-800 light:hover:bg-ink-50 transition-colors"
                >
                  Explore the research
                </Link>
              </div>
            </div>

            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2, ease: 'easeOut' }}
              className="relative mt-14 lg:mt-0 flex justify-center lg:justify-end shrink-0"
            >
              <div
                aria-hidden="true"
                className="absolute inset-0 m-auto w-52 h-52 rounded-full bg-vital-500/25 light:bg-vital-500/20 blur-[64px]"
              />
              <div className="relative w-60 rounded-3xl border border-white/10 light:border-ink-900/[0.06] bg-ink-900/30 light:bg-white/50 backdrop-blur-xl shadow-[0_1px_0_0_rgba(255,255,255,0.06)_inset,0_20px_50px_-18px_rgba(0,0,0,0.45)] p-6">
                <p className="text-xs font-semibold tracking-wide uppercase text-ink-400 light:text-ink-500 text-center mb-4">
                  Sample output
                </p>
                <RiskGauge probability={0.18} size={132} />
                <p className="text-[11px] text-ink-400 light:text-ink-500 text-center mt-4 leading-snug">
                  Illustrative prediction &mdash; not a real patient.
                </p>
              </div>
            </motion.div>
          </div>
        </div>

        <div className="max-w-6xl mx-auto px-5 sm:px-8 mt-14 sm:mt-20">
          <Waveform repeats={22} height={110} strokeWidth={2.5} />
        </div>
      </section>

      {/* ── Headline stats, read like an instrument panel ─────── */}
      <section className="max-w-6xl mx-auto px-5 sm:px-8 mt-4 mb-20">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-px rounded-2xl overflow-hidden border border-ink-800 light:border-ink-100 bg-ink-800 light:bg-ink-100">
          {[
            ['0.9584', 'ROC-AUC · heart (TabPFN)'],
            ['0.8776', 'Macro F1 · ECG (CNN)'],
            ['0.9675', 'Macro F1 · symptom text'],
            ['0.634', 'ROC-AUC · external cohort'],
          ].map(([n, label]) => (
            <div key={label} className="bg-ink-950 light:bg-white p-5 sm:p-6">
              <div className="data-readout text-2xl sm:text-3xl font-semibold text-ink-50 light:text-ink-900">
                {n}
              </div>
              <div className="text-xs text-ink-400 light:text-ink-500 mt-1.5 leading-snug">{label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Welcome / why this exists ──────────────────────────── */}
      <section className="max-w-6xl mx-auto px-5 sm:px-8 pb-20">
        <div className="max-w-2xl mb-10">
          <span className="text-xs font-semibold tracking-wide uppercase text-vital-400">Welcome</span>
          <h2 className="font-display font-semibold text-2xl sm:text-3xl text-ink-50 light:text-ink-900 mt-2">
            A second opinion built from evidence, not a black box.
          </h2>
          <p className="mt-3 text-ink-400 light:text-ink-500 leading-relaxed">
            Most risk calculators return a single number from a single model and stop there.
            HealthGuard AI takes a different approach: eleven separate models — from established
            statistical baselines to a modern pretrained AI system — are trained and compared
            honestly on the same clinical data, rather than picking one model on faith and hoping
            it holds up.
          </p>
          <p className="mt-3 text-ink-400 light:text-ink-500 leading-relaxed">
            Every prediction is explained feature by feature using SHAP, so a risk score is never
            just a number on a screen — you can see exactly which values pushed it up or down, and
            by how much. Beyond that, each model is also tested on a second, independent patient
            population it has never seen during training, to check whether strong results in the
            lab actually survive contact with real-world data. Where that testing succeeded, it's
            reported. Where it didn't, that's reported too.
          </p>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {WHY.map((w) => (
            <div key={w.title}>
              <div className="w-9 h-9 rounded-lg bg-vital-500/10 flex items-center justify-center mb-3">
                <w.icon size={17} className="text-vital-400" />
              </div>
              <h3 className="font-semibold text-sm text-ink-100 light:text-ink-800 mb-1.5">{w.title}</h3>
              <p className="text-sm text-ink-400 light:text-ink-500 leading-relaxed">{w.desc}</p>
            </div>
          ))}
        </div>
        <Link
          to="/features"
          className="mt-8 inline-flex items-center gap-1.5 text-sm font-semibold text-vital-400 hover:gap-2.5 transition-all"
        >
          See all features <ArrowRight size={14} />
        </Link>
      </section>

      {/* ── Three phases ───────────────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-5 sm:px-8 pb-24">
        <div className="grid sm:grid-cols-3 gap-5">
          {PHASES.map((p) => (
            <Card key={p.title} className="p-6 flex flex-col group hover:border-vital-500/40 transition-colors">
              <div className="w-10 h-10 rounded-lg bg-vital-500/10 flex items-center justify-center mb-4">
                <p.icon size={19} className="text-vital-400" />
              </div>
              <h3 className="font-display font-semibold text-ink-50 light:text-ink-900 mb-2">{p.title}</h3>
              <p className="text-sm text-ink-400 light:text-ink-500 leading-relaxed flex-1">{p.desc}</p>
              <Link
                to={p.to}
                className="mt-5 inline-flex items-center gap-1.5 text-sm font-semibold text-vital-400 group-hover:gap-2.5 transition-all"
              >
                {p.cta} <ArrowRight size={14} />
              </Link>
            </Card>
          ))}
        </div>
      </section>

      {/* ── How it works ───────────────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-5 sm:px-8 pb-24">
        <div className="max-w-2xl mb-14 mx-auto text-center">
          <span className="text-xs font-semibold tracking-wide uppercase text-vital-400">How it works</span>
          <h2 className="font-display font-semibold text-2xl sm:text-3xl text-ink-50 light:text-ink-900 mt-2">
            From clinical values to an explained risk score.
          </h2>
        </div>
        <HowItWorks />
      </section>

      {/* ── Built on rigour ────────────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-5 sm:px-8 pb-24">
        <Card className="p-8 sm:p-10">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-8">
            {TRUST_STATS.map((s) => (
              <CountStat key={s.label} {...s} />
            ))}
            <div className="text-center">
              <div className="flex items-center justify-center text-vital-400">
                <CheckCircle2 size={30} strokeWidth={1.75} />
              </div>
              <div className="text-xs text-ink-400 light:text-ink-500 mt-2 leading-snug max-w-[12rem] mx-auto">
                Every result reproducible from the committed codebase
              </div>
            </div>
          </div>
        </Card>
      </section>

      {/* ── Closing statement ──────────────────────────────────── */}
      <section className="max-w-3xl mx-auto px-5 sm:px-8 py-20 sm:py-24 text-center">
        <div className="space-y-4 sm:space-y-5">
          {MANIFESTO.map((line, i) => (
            <motion.p
              key={line}
              initial={{ opacity: 0, y: 14 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.7 }}
              transition={{ duration: 0.5, delay: i * 0.12, ease: 'easeOut' }}
              className={
                i === MANIFESTO.length - 1
                  ? 'font-display font-semibold text-2xl sm:text-3xl text-vital-400'
                  : 'font-display font-medium text-xl sm:text-2xl text-ink-200 light:text-ink-700'
              }
            >
              {line}
            </motion.p>
          ))}
        </div>
      </section>
    </div>
  );
}
