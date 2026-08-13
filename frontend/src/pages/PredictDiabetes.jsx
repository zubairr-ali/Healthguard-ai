import { useState } from 'react';
import { Droplets, AlertTriangle } from 'lucide-react';
import { Card, Field, NumberInput, Button, ModelBadge } from '../components/ui';
import PredictionResult from '../components/PredictionResult';
import WhyThisModel from '../components/WhyThisModel';
import Waveform, { WaveformPulse } from '../components/Waveform';
import { api, ApiError } from '../lib/api';

const DEFAULTS = {
  Pregnancies: 2, Glucose: 120, BloodPressure: 72, SkinThickness: 23,
  Insulin: 85, BMI: 28.5, DiabetesPedigreeFunction: 0.47, Age: 33,
};

const FIELDS = [
  ['Pregnancies', 'Pregnancies', null, 0, 20, 1],
  ['Glucose', 'Glucose', 'mg/dL', 0, 300, 1],
  ['BloodPressure', 'Blood pressure', 'mm Hg', 0, 200, 1],
  ['SkinThickness', 'Skin thickness', 'mm', 0, 100, 1],
  ['Insulin', 'Insulin', 'mu U/mL', 0, 900, 1],
  ['BMI', 'BMI', 'kg/m²', 0, 70, 0.1],
  ['DiabetesPedigreeFunction', 'Pedigree function', 'genetic risk score', 0, 3, 0.01],
  ['Age', 'Age', 'years', 1, 120, 1],
];

export default function PredictDiabetes() {
  const [form, setForm] = useState(DEFAULTS);
  const [submitted, setSubmitted] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  async function onSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = Object.fromEntries(Object.entries(form).map(([k, v]) => [k, Number(v)]));
      const res = await api.predictDiabetes(payload);
      setResult(res);
      setSubmitted(payload);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-5 sm:px-8 py-12">
      <div className="flex items-center gap-3 mb-2">
        <div className="w-9 h-9 rounded-lg bg-amber-500/10 flex items-center justify-center">
          <Droplets size={18} className="text-amber-400" />
        </div>
        <ModelBadge label="Validated model" name="XGBoost" metric="0.8233 ROC-AUC" />
      </div>
      <WhyThisModel>
        XGBoost was chosen for diabetes predictions because it outperformed ten other models
        during testing, including a modern pretrained AI system and three specialised
        transformer models. It's particularly good at picking up on meaningful thresholds in the
        data — for example, the glucose levels that mark a shift from normal to at-risk — which
        fits well with how diabetes risk factors typically behave.
      </WhyThisModel>
      <h1 className="font-display font-semibold text-3xl text-ink-50 light:text-ink-900 mt-3">
        Diabetes risk
      </h1>
      <p className="text-ink-400 light:text-ink-500 mt-2 max-w-2xl">
        Enter diagnostic measurements to estimate diabetes risk, based on the Pima Indians
        dataset schema.
      </p>

      <div className="grid lg:grid-cols-[420px_minmax(0,1fr)] gap-8 mt-10 items-start">
        <Card className="p-6">
          <form onSubmit={onSubmit} className="space-y-5">
            <div className="grid grid-cols-2 gap-4">
              {FIELDS.map(([key, label, hint, min, max, step]) => (
                <Field key={key} label={label} hint={hint} htmlFor={key}>
                  <NumberInput
                    id={key} min={min} max={max} step={step}
                    value={form[key]} onChange={set(key)} required
                  />
                </Field>
              ))}
            </div>

            <Button type="submit" disabled={loading} className="w-full mt-2">
              {loading ? <WaveformPulse size={14} /> : null}
              {loading ? 'Running prediction…' : 'Predict risk'}
            </Button>
          </form>
        </Card>

        <div>
          {error && (
            <Card className="p-4 border-signal-500/30 bg-signal-500/[0.04] flex gap-3 items-start mb-6">
              <AlertTriangle size={16} className="text-signal-400 shrink-0 mt-0.5" />
              <p className="text-sm text-signal-300">{error}</p>
            </Card>
          )}
          {loading ? (
            <Card className="p-10 flex flex-col items-center justify-center text-center h-full min-h-[320px]">
              <Waveform
                repeats={6}
                height={56}
                strokeWidth={2.5}
                loop
                className="w-full max-w-xs mb-5"
              />
              <p className="text-sm font-medium text-ink-200 light:text-ink-700">
                Running XGBoost and computing the SHAP explanation…
              </p>
            </Card>
          ) : result ? (
            <Card className="p-6 sm:p-8">
              <PredictionResult result={result} disease="diabetes" diseaseLabel="diabetes" formValues={submitted} />
            </Card>
          ) : !error ? (
            <Card className="p-10 flex flex-col items-center justify-center text-center h-full min-h-[320px]">
              <Droplets size={28} className="text-ink-600 mb-3" />
              <p className="text-sm text-ink-500 light:text-ink-400 max-w-xs">
                Fill in the form and run a prediction to see the risk estimate and its
                explanation here.
              </p>
            </Card>
          ) : null}
        </div>
      </div>
    </div>
  );
}
