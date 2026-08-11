import { Link } from 'react-router-dom';
import {
  HeartPulse, Activity, FlaskConical, ArrowRight, ShieldCheck,
  Layers, ScanSearch, Network, Lock,
} from 'lucide-react';
import Waveform from '../components/Waveform';
import { Card, Badge } from '../components/ui';

const WHY = [
  {
    icon: Layers,
    title: 'Eight models, not one',
    desc: 'Gradient boosting, ensembles, a deep neural network, and a pretrained transformer, tuned and compared rather than a single model taken on faith.',
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
    desc: 'Eight tuned models — gradient boosting, ensembles, and a pretrained transformer — compared on heart disease and diabetes records, each explained with SHAP.',
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

export default function Home() {
  return (
    <div>
      {/* ── Hero ───────────────────────────────────────────────── */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_50%_at_50%_0%,rgba(20,184,166,0.12),transparent)]" />
        <div className="max-w-5xl mx-auto px-5 sm:px-8 pt-16 sm:pt-24 pb-8 text-center relative">
          <Badge tone="vital" className="mb-6">
            <ShieldCheck size={13} /> Computer Science Project · 6WCM0029
          </Badge>
          <h1 className="font-display font-semibold text-4xl sm:text-6xl tracking-tight text-ink-50 light:text-ink-900 leading-[1.05]">
            Reading the signal in
            <br />
            <span className="text-vital-400">patient risk data.</span>
          </h1>
          <p className="mt-6 text-lg text-ink-300 light:text-ink-600 max-w-2xl mx-auto leading-relaxed">
            HealthGuard AI predicts heart disease and diabetes risk from clinical records,
            explains every prediction with SHAP, and stress-tests each model against
            independent cohorts — reporting where they hold up and where they don't.
          </p>
          <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
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
            HealthGuard AI compares several models honestly, shows its working on every
            prediction, and reports the results of testing itself against data it has never
            seen — including the cases where that testing didn't go well.
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
    </div>
  );
}
