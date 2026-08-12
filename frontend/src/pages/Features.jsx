import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ListChecks, Layers, ScanSearch, Network, ClipboardCheck, Lock, Clock, ArrowRight, AlertTriangle,
} from 'lucide-react';
import { Card, Badge } from '../components/ui';
import Waveform from '../components/Waveform';

// Every fact below is checked against the codebase, not asserted — model
// names against ModelComparison.jsx's FAMILY map and the README's model
// lineup, the advisory note's data path against PredictionResult.jsx (which
// calls buildAdvisory() from clinicalNarrative.js directly; the backend's
// own advisory_note field is fetched but never rendered), and the
// local-only claim against main.py, where both predict endpoints call
// generate_fallback_advisory — never the OpenAI-backed generate_advisory —
// so no prediction ever leaves the machine it runs on.
const FEATURES = [
  {
    icon: Layers,
    title: 'Eleven models, compared honestly',
    what: "Every prediction — for both heart disease and diabetes — comes from the best of eleven separately trained models: a logistic-regression baseline, four gradient-boosted and tree-ensemble models (XGBoost, LightGBM, CatBoost, Random Forest), a stacked ensemble, a deep neural network (MLP), three tabular transformer architectures (FT-Transformer, TabTransformer, TabNet), and TabPFN, a pretrained tabular foundation model — all benchmarked under identical conditions.",
    why: "A single model can look strong in isolation and still be the wrong fit for a particular dataset's shape and size. Comparing eleven honestly — and reporting where the weaker ones fall short, not just the winner — is what makes the eventual choice (TabPFN for heart disease, XGBoost for diabetes) a documented result rather than an assumption.",
    cta: { label: 'See the full comparison', to: '/models' },
  },
  {
    icon: ScanSearch,
    title: 'Every prediction explained with SHAP',
    what: 'Each prediction is broken down feature by feature using SHAP, showing exactly which of the submitted values pushed the risk score up or down, and by roughly how much — not just the final number.',
    why: 'A risk score with no reason behind it isn’t something a clinician, or the patient themselves, can meaningfully question or act on. Knowing that, say, cholesterol and ST slope were the two strongest factors turns an opaque output into a specific, checkable claim.',
  },
  {
    icon: Network,
    title: 'Tested on data it has never seen',
    what: 'Beyond standard cross-validation, six of these models were also re-tested on a second, genuinely independent population — trained on one patient cohort, evaluated on another the model never touched during training.',
    why: 'Strong performance on a model’s own training data answers a different question than "will this hold up on a new patient at a different clinic." This project measured that gap directly instead of assuming it away.',
    cta: { label: 'See how large that gap turned out to be', to: '#real-world-performance', hash: true },
  },
  {
    icon: ClipboardCheck,
    title: 'An advisory note grounded in the real numbers',
    what: 'Alongside the risk score, the app writes a plain-language advisory note built directly from that patient’s own submitted values and that specific prediction’s SHAP contributions — citing the actual clinical reference range each value falls into (WHO glucose and BMI categories, ACC/AHA blood pressure categories, NCEP ATP III cholesterol categories), not a generic disclaimer.',
    why: '"Your cholesterol is elevated" says very little. "Cholesterol was recorded at 254 mg/dL — in the high range by NCEP ATP III reference — and was among the strongest factors increasing this prediction" is something a person can actually bring to a doctor.',
  },
  {
    icon: Lock,
    title: 'Runs entirely on your machine',
    what: 'No patient data leaves the app. Predictions run against locally-trained model files, SHAP explanations are computed locally, and the FastAPI backend never calls an external service to produce a result — the advisory note above is written by a fully local, deterministic process, not a hosted AI service.',
    why: 'This is health data. Keeping the entire prediction path on the machine running it — not proxied through a third party — is a privacy property built into the architecture, not an afterthought.',
  },
  {
    icon: Clock,
    title: 'A local prediction history',
    what: 'Every prediction made through the interface — timestamp, condition, model used, and risk score — is logged to a local SQLite database and viewable on the History page.',
    why: 'A single reading in isolation is a snapshot. A logged history is something a person — or, eventually, a clinician — can look back across over time, rather than treating each check as disconnected from the last.',
    cta: { label: 'View prediction history', to: '/history' },
  },
];

export default function Features() {
  return (
    <div className="max-w-6xl mx-auto px-5 sm:px-8 py-12">
      <Badge tone="vital" className="mb-4">
        <ListChecks size={13} /> What it does &amp; what that means
      </Badge>
      <h1 className="font-display font-semibold text-3xl sm:text-4xl text-ink-50 light:text-ink-900 max-w-2xl">
        What the system can do &mdash;
        <br />
        <span className="text-vital-400">and what that means for a real patient.</span>
      </h1>
      <p className="mt-5 text-lg text-ink-300 light:text-ink-600 max-w-2xl leading-relaxed">
        This page covers two things together, on purpose: the concrete capabilities behind every
        prediction &mdash; eleven models compared, every result explained, each one tested against
        data it has never seen &mdash; and, just as importantly, what those capabilities actually
        mean once a real person's clinical values go in. Capability without that context is a
        marketing page. This is meant to be neither more nor less than what's actually true.
      </p>

      <div className="mt-10">
        <Waveform repeats={18} height={70} strokeWidth={2} />
      </div>

      {/* ── The real-world problem ─────────────────────────────── */}
      <section className="max-w-3xl mt-16 mb-20">
        <span className="text-xs font-semibold tracking-wide uppercase text-vital-400">
          Why this exists
        </span>
        <h2 className="font-display font-semibold text-2xl sm:text-3xl text-ink-50 light:text-ink-900 mt-2 mb-5">
          A real gap this project set out to investigate.
        </h2>
        <p className="text-ink-300 light:text-ink-600 leading-relaxed">
          Across many health systems, early risk signals for chronic conditions like heart disease
          and diabetes often go unassessed until symptoms are already present &mdash; not because
          the risk factors are hidden, but because turning routine clinical measurements into an
          actionable risk estimate usually requires a clinician's time, judgement, and access to
          specialist tools. HealthGuard AI investigates whether that step can be partially
          automated: given the same measurements a clinician would already record, can a
          well-tested model produce a risk estimate immediately, with its reasoning shown rather
          than hidden &mdash; so it becomes useful evidence a person can bring into a conversation
          with a doctor, not a replacement for that conversation.
        </p>
      </section>

      {/* ── Feature-by-feature breakdown ───────────────────────── */}
      <section className="mb-24">
        <div className="max-w-2xl mb-10">
          <span className="text-xs font-semibold tracking-wide uppercase text-vital-400">
            What the system does
          </span>
          <h2 className="font-display font-semibold text-2xl sm:text-3xl text-ink-50 light:text-ink-900 mt-2">
            Six capabilities, each with a reason it exists.
          </h2>
        </div>

        <div className="grid lg:grid-cols-2 gap-5">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.2 }}
              transition={{ duration: 0.4, delay: (i % 2) * 0.08, ease: 'easeOut' }}
            >
              <Card className="p-6 h-full flex flex-col">
                <div className="w-10 h-10 rounded-lg bg-vital-500/10 flex items-center justify-center mb-4">
                  <f.icon size={19} className="text-vital-400" />
                </div>
                <h3 className="font-display font-semibold text-ink-50 light:text-ink-900 mb-3">
                  {f.title}
                </h3>
                <p className="text-sm text-ink-400 light:text-ink-500 leading-relaxed mb-3">
                  <span className="font-semibold text-ink-200 light:text-ink-700">What it is: </span>
                  {f.what}
                </p>
                <p className="text-sm text-ink-400 light:text-ink-500 leading-relaxed flex-1">
                  <span className="font-semibold text-ink-200 light:text-ink-700">Why it matters: </span>
                  {f.why}
                </p>
                {f.cta && (f.cta.hash ? (
                  <a
                    href={f.cta.to}
                    className="mt-5 inline-flex items-center gap-1.5 text-sm font-semibold text-vital-400 hover:gap-2.5 transition-all"
                  >
                    {f.cta.label} <ArrowRight size={14} />
                  </a>
                ) : (
                  <Link
                    to={f.cta.to}
                    className="mt-5 inline-flex items-center gap-1.5 text-sm font-semibold text-vital-400 hover:gap-2.5 transition-all"
                  >
                    {f.cta.label} <ArrowRight size={14} />
                  </Link>
                ))}
              </Card>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── How would this perform for a real patient? ─────────── */}
      <motion.section
        id="real-world-performance"
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.15 }}
        transition={{ duration: 0.45, ease: 'easeOut' }}
        className="scroll-mt-24 mb-20"
      >
        <div className="max-w-3xl mb-8">
          <span className="text-xs font-semibold tracking-wide uppercase text-signal-400">
            The honest answer
          </span>
          <h2 className="font-display font-semibold text-2xl sm:text-3xl text-ink-50 light:text-ink-900 mt-2">
            How would this perform for a real patient?
          </h2>
        </div>

        <div className="max-w-3xl space-y-4 text-ink-300 light:text-ink-600 leading-relaxed">
          <p>
            It's tempting to read a headline number like TabPFN's 95.84% ROC-AUC on heart disease
            and assume that accuracy would carry over to any new patient, anywhere. This project
            tested that assumption directly, rather than assuming it &mdash; and the honest answer
            is: not reliably.
          </p>
          <p>
            When six of these models were trained on one patient population and tested on a
            second, genuinely independent cohort of 70,000 patients, performance did not travel
            cleanly. Internal cross-validated accuracy of 0.70&ndash;0.80 fell to 0.51&ndash;0.63
            once tested externally &mdash; in one direction, close to a coin flip. A model's rank
            on its own training data also did not reliably predict its rank on new data: the model
            that performed best internally in one experiment dropped to fifth out of six once
            tested on the independent cohort.
          </p>
          <p>
            This is exactly why HealthGuard AI is built as decision support, not diagnosis. In a
            real clinical setting, a prediction from this system should be treated the way a
            single lab test result would be &mdash; one data point a clinician weighs alongside
            history, other tests, and their own judgement, not a verdict to act on alone. The
            system's job is to make an honest risk estimate visible and explain its reasoning; the
            job of interpreting that estimate for a specific patient still belongs to a trained
            clinician.
          </p>
        </div>

        <Card className="max-w-3xl mt-8 p-6 sm:p-8">
          <div className="grid sm:grid-cols-3 gap-6 sm:divide-x divide-ink-800 light:divide-ink-100">
            <div className="text-center">
              <div className="data-readout text-2xl sm:text-3xl font-semibold text-ink-50 light:text-ink-900">
                0.70&ndash;0.80
              </div>
              <div className="text-xs text-ink-400 light:text-ink-500 mt-2 leading-snug">
                internal, cross-validated ROC-AUC
              </div>
            </div>
            <div className="text-center sm:px-2">
              <div className="data-readout text-2xl sm:text-3xl font-semibold text-signal-400">
                0.51&ndash;0.63
              </div>
              <div className="text-xs text-ink-400 light:text-ink-500 mt-2 leading-snug">
                external, independent-cohort ROC-AUC
              </div>
            </div>
            <div className="text-center sm:pl-2">
              <div className="data-readout text-2xl sm:text-3xl font-semibold text-signal-400">
                1st &rarr; 5th of 6
              </div>
              <div className="text-xs text-ink-400 light:text-ink-500 mt-2 leading-snug">
                best-ranked internal model's external rank, one direction
              </div>
            </div>
          </div>
        </Card>

        <div className="max-w-3xl mt-6 flex gap-3 items-start p-4 rounded-lg bg-amber-500/[0.04] border border-amber-500/20">
          <AlertTriangle size={15} className="text-amber-400 shrink-0 mt-0.5" />
          <p className="text-sm text-ink-300 light:text-ink-600 leading-relaxed">
            <span className="font-medium text-amber-300">One further limitation, transparently: </span>
            the ECG classification results were evaluated with heartbeats split randomly rather
            than by patient, which the project has documented can inflate reported accuracy
            relative to real deployment conditions.{' '}
            <Link to="/findings#ecg" className="font-medium text-amber-300 hover:underline">
              See the full caveat on the Research Findings page
            </Link>
            .
          </p>
        </div>
      </motion.section>
    </div>
  );
}
