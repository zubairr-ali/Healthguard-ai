import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend, Cell,
} from 'recharts';
import { Activity, FileText, Network, AlertOctagon } from 'lucide-react';
import { Card, Badge } from '../components/ui';
import Waveform from '../components/Waveform';

const TICK = { fill: 'var(--color-ink-400)', fontSize: 11, fontFamily: 'var(--font-data)' };
const GRID = 'var(--color-ink-800)';

const ECG_DATA = [
  { name: 'CNN', macroF1: 0.8776, params: 343397 },
  { name: 'CNN-BiLSTM', macroF1: 0.866, params: 184325 },
  { name: 'BiLSTM (converged)', macroF1: 0.7464, params: 141189 },
];

const NLP_DATA = [
  { name: 'TF-IDF + SVM', macroF1: 0.9675 },
  { name: 'DistilBERT', macroF1: 0.9635 },
  { name: 'TF-IDF + LogReg', macroF1: 0.9626 },
  { name: 'Frozen DistilBERT', macroF1: 0.945 },
];

const LEARNING_CURVE = [
  { n: 80, tfidf: 0.7075, bert: 0.6897 },
  { n: 201, tfidf: 0.8573, bert: 0.8563 },
  { n: 403, tfidf: 0.9324, bert: 0.9331 },
  { n: 807, tfidf: 0.9516, bert: 0.9635 },
];

const XVAL_DATA = [
  { model: 'TabPFN', internal: 0.7627, external: 0.6342 },
  { model: 'LogReg', internal: 0.7606, external: 0.6276 },
  { model: 'CatBoost', internal: 0.7481, external: 0.6222 },
  { model: 'XGBoost', internal: 0.7411, external: 0.6133 },
  { model: 'LightGBM', internal: 0.7208, external: 0.6019 },
  { model: 'Random Forest', internal: 0.6994, external: 0.6103 },
];

function Section({ id, icon: Icon, eyebrow, title, children }) {
  return (
    <section id={id} className="scroll-mt-24 py-16 border-t border-ink-800 light:border-ink-100 first:border-t-0 first:pt-0">
      <div className="flex items-center gap-2.5 mb-3">
        <Icon size={16} className="text-vital-400" />
        <span className="text-xs font-semibold tracking-wide uppercase text-vital-400">{eyebrow}</span>
      </div>
      <h2 className="font-display font-semibold text-2xl sm:text-3xl text-ink-50 light:text-ink-900 mb-6">
        {title}
      </h2>
      {children}
    </section>
  );
}

export default function Findings() {
  return (
    <div className="max-w-5xl mx-auto px-5 sm:px-8 py-12">
      <Badge tone="vital" className="mb-4">Research findings</Badge>
      <h1 className="font-display font-semibold text-3xl sm:text-4xl text-ink-50 light:text-ink-900 max-w-2xl">
        Three data modalities, one question:
        <span className="text-vital-400"> does model complexity earn its cost?</span>
      </h1>
      <p className="mt-4 text-ink-400 light:text-ink-500 max-w-2xl leading-relaxed">
        Structured records, ECG waveforms, and free-text symptom descriptions were each modelled
        with the most sophisticated architecture available and compared against a simpler
        baseline under identical conditions. Across all three, the more complex model won once.
      </p>

      <Section id="ecg" icon={Activity} eyebrow="Phase 2" title="ECG arrhythmia classification">
        <p className="text-ink-400 light:text-ink-500 mb-6 leading-relaxed">
          109,446 heartbeats from MIT-BIH, three architectures under an identical training budget.
          A convergence check on the BiLSTM \u2014 40 epochs was insufficient \u2014 raised its macro F1 from
          0.7187 to 0.7464 and confirmed the gap to the CNN is architectural, not a training-budget artefact.
        </p>
        <Card className="p-6">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={ECG_DATA} margin={{ left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
              <XAxis dataKey="name" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
              <YAxis domain={[0, 1]} tick={TICK} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: 'var(--color-ink-900)', border: '1px solid var(--color-ink-700)', borderRadius: 8, fontSize: 12 }}
                formatter={(v) => v.toFixed(4)}
              />
              <Bar dataKey="macroF1" radius={[6, 6, 0, 0]}>
                {ECG_DATA.map((d, i) => (
                  <Cell key={d.name} fill={i === 0 ? 'var(--color-vital-400)' : 'var(--color-ink-500)'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <Waveform repeats={14} height={36} strokeWidth={1.5} animate={false} className="opacity-30 mt-4" />
        </Card>
      </Section>

      <Section id="nlp" icon={FileText} eyebrow="Phase 3" title="Free-text symptom classification">
        <p className="text-ink-400 light:text-ink-500 mb-6 leading-relaxed">
          A linear SVM on TF-IDF features matched a fine-tuned 67M-parameter DistilBERT at roughly
          1,600&times; faster training. The learning curve below shows the two staying within one
          standard deviation of each other at every training size tested.
        </p>
        <div className="grid sm:grid-cols-2 gap-5">
          <Card className="p-6">
            <p className="text-xs font-semibold text-ink-400 light:text-ink-500 mb-4">Macro F1 by model</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={NLP_DATA} layout="vertical" margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
                <XAxis type="number" domain={[0.9, 1]} tick={TICK} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" tick={{ ...TICK, fontFamily: 'var(--font-body)' }} width={110} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: 'var(--color-ink-900)', border: '1px solid var(--color-ink-700)', borderRadius: 8, fontSize: 12 }}
                  formatter={(v) => v.toFixed(4)}
                />
                <Bar dataKey="macroF1" radius={[0, 6, 6, 0]} fill="var(--color-vital-400)" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
          <Card className="p-6">
            <p className="text-xs font-semibold text-ink-400 light:text-ink-500 mb-4">Learning curve (test macro F1)</p>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={LEARNING_CURVE}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
                <XAxis dataKey="n" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
                <YAxis domain={[0.6, 1]} tick={TICK} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: 'var(--color-ink-900)', border: '1px solid var(--color-ink-700)', borderRadius: 8, fontSize: 12 }}
                  formatter={(v) => v.toFixed(4)}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="tfidf" name="TF-IDF" stroke="var(--color-ink-400)" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="bert" name="DistilBERT" stroke="var(--color-vital-400)" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      </Section>

      <Section id="cross-dataset" icon={Network} eyebrow="Phase 4" title="Cross-dataset external validation">
        <p className="text-ink-400 light:text-ink-500 mb-2 leading-relaxed">
          Every model above is validated internally. This phase asks the question that matters
          clinically: trained at one site, does it hold up at another? Heart disease (918 rows)
          and an independent cardiovascular cohort (70,000 rows) were harmonised onto a shared
          five-feature subset and tested in both directions.
        </p>
        <p className="text-sm font-semibold text-signal-400 mb-6">
          Spearman rank correlation between internal and external performance: 0.943 one direction, 0.257 the other.
          In-sample ranking is not a reliable guide to real-world performance.
        </p>
        <Card className="p-6">
          <p className="text-xs font-semibold text-ink-400 light:text-ink-500 mb-4">
            Internal (5-fold CV) vs external (independent cohort) ROC-AUC — Heart &rarr; Cardio
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={XVAL_DATA} margin={{ left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
              <XAxis dataKey="model" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
              <YAxis domain={[0.5, 0.85]} tick={TICK} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: 'var(--color-ink-900)', border: '1px solid var(--color-ink-700)', borderRadius: 8, fontSize: 12 }}
                formatter={(v) => v.toFixed(4)}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="internal" name="Internal (CV)" fill="var(--color-ink-500)" radius={[6, 6, 0, 0]} />
              <Bar dataKey="external" name="External (independent cohort)" fill="var(--color-signal-500)" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </Section>

      <Section id="contamination" icon={AlertOctagon} eyebrow="Methodology" title="A dataset provenance failure">
        <p className="text-ink-400 light:text-ink-500 mb-6 leading-relaxed">
          A second external-validation experiment, between the Pima diabetes dataset and a
          distributed &ldquo;Frankfurt Hospital&rdquo; cohort, was aborted after row-level
          fingerprinting found that 95.6% of Pima's records also appear in the &ldquo;independent&rdquo;
          file &mdash; it is Pima resampled with replacement, not a separate cohort. The validation
          pipeline was then extended with an automatic independence gate that refuses to score
          any dataset pair sharing more than 5% of rows.
        </p>
        <div className="grid grid-cols-3 gap-4">
          {[
            ['62.8%', 'exact duplicate rows within the "Frankfurt" file'],
            ['95.6%', 'of Pima rows also found in "Frankfurt"'],
            ['~10', 'unique rows once duplicates are removed'],
          ].map(([n, label]) => (
            <Card key={label} className="p-4 text-center">
              <div className="data-readout text-2xl font-semibold text-signal-400">{n}</div>
              <div className="text-xs text-ink-400 light:text-ink-500 mt-1.5 leading-snug">{label}</div>
            </Card>
          ))}
        </div>
      </Section>
    </div>
  );
}
