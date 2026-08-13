import { useState } from 'react';
import { HeartPulse, AlertTriangle } from 'lucide-react';
import { Card, Field, NumberInput, Select, Button, ModelBadge } from '../components/ui';
import PredictionResult from '../components/PredictionResult';
import WhyThisModel from '../components/WhyThisModel';
import Waveform, { WaveformPulse } from '../components/Waveform';
import { api, ApiError } from '../lib/api';

const DEFAULTS = {
  Age: 54, Sex: 'M', ChestPainType: 'ATA', RestingBP: 130, Cholesterol: 220,
  FastingBS: '0', RestingECG: 'Normal', MaxHR: 145, ExerciseAngina: 'N',
  Oldpeak: 1.0, ST_Slope: 'Up',
};

export default function PredictHeart() {
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
      const payload = {
        ...form,
        Age: Number(form.Age), RestingBP: Number(form.RestingBP),
        Cholesterol: Number(form.Cholesterol), FastingBS: Number(form.FastingBS),
        MaxHR: Number(form.MaxHR), Oldpeak: Number(form.Oldpeak),
      };
      const res = await api.predictHeart(payload);
      setResult(res);
      setSubmitted(form);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-5 sm:px-8 py-12">
      <div className="flex items-center gap-3 mb-2">
        <div className="w-9 h-9 rounded-lg bg-signal-500/10 flex items-center justify-center">
          <HeartPulse size={18} className="text-signal-400" />
        </div>
        <ModelBadge label="High-accuracy model" name="TabPFN" metric="0.9584 ROC-AUC" />
      </div>
      <WhyThisModel>
        TabPFN was chosen for heart disease predictions because it consistently outscored ten
        other AI models during testing — including well-established methods and several newer
        designs. It comes pre-trained on a huge variety of data patterns, which helps it perform
        reliably even with a relatively small set of patient records like this one, and it
        reached its best result without needing any manual fine-tuning.
      </WhyThisModel>
      <h1 className="font-display font-semibold text-3xl text-ink-50 light:text-ink-900 mt-3">
        Heart disease risk
      </h1>
      <p className="text-ink-400 light:text-ink-500 mt-2 max-w-2xl">
        Enter clinical measurements to estimate coronary disease risk, with a feature-level
        explanation of the result.
      </p>

      <div className="grid lg:grid-cols-[420px_minmax(0,1fr)] gap-8 mt-10 items-start">
        <Card className="p-6">
          <form onSubmit={onSubmit} className="space-y-5">
            <div className="grid grid-cols-2 gap-4">
              <Field label="Age" htmlFor="age">
                <NumberInput id="age" min="1" max="120" value={form.Age} onChange={set('Age')} required />
              </Field>
              <Field label="Sex" htmlFor="sex">
                <Select id="sex" value={form.Sex} onChange={set('Sex')}>
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                </Select>
              </Field>
            </div>

            <Field label="Chest pain type" htmlFor="cp">
              <Select id="cp" value={form.ChestPainType} onChange={set('ChestPainType')}>
                <option value="ATA">Atypical angina</option>
                <option value="NAP">Non-anginal pain</option>
                <option value="ASY">Asymptomatic</option>
                <option value="TA">Typical angina</option>
              </Select>
            </Field>

            <div className="grid grid-cols-2 gap-4">
              <Field label="Resting BP" hint="mm Hg" htmlFor="bp">
                <NumberInput id="bp" min="60" max="240" value={form.RestingBP} onChange={set('RestingBP')} required />
              </Field>
              <Field label="Cholesterol" hint="mg/dL" htmlFor="chol">
                <NumberInput id="chol" min="0" max="700" value={form.Cholesterol} onChange={set('Cholesterol')} required />
              </Field>
            </div>

            <Field label="Fasting blood sugar" htmlFor="fbs">
              <Select id="fbs" value={form.FastingBS} onChange={set('FastingBS')}>
                <option value="0">120 mg/dL or below</option>
                <option value="1">Above 120 mg/dL</option>
              </Select>
            </Field>

            <Field label="Resting ECG" htmlFor="recg">
              <Select id="recg" value={form.RestingECG} onChange={set('RestingECG')}>
                <option value="Normal">Normal</option>
                <option value="ST">ST-T wave abnormality</option>
                <option value="LVH">Left ventricular hypertrophy</option>
              </Select>
            </Field>

            <div className="grid grid-cols-2 gap-4">
              <Field label="Max heart rate" htmlFor="mhr">
                <NumberInput id="mhr" min="60" max="220" value={form.MaxHR} onChange={set('MaxHR')} required />
              </Field>
              <Field label="Exercise angina" htmlFor="exa">
                <Select id="exa" value={form.ExerciseAngina} onChange={set('ExerciseAngina')}>
                  <option value="N">No</option>
                  <option value="Y">Yes</option>
                </Select>
              </Field>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <Field label="Oldpeak" hint="ST depression" htmlFor="op">
                <NumberInput id="op" step="0.1" min="-3" max="7" value={form.Oldpeak} onChange={set('Oldpeak')} required />
              </Field>
              <Field label="ST slope" htmlFor="sts">
                <Select id="sts" value={form.ST_Slope} onChange={set('ST_Slope')}>
                  <option value="Up">Upsloping</option>
                  <option value="Flat">Flat</option>
                  <option value="Down">Downsloping</option>
                </Select>
              </Field>
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
                Running TabPFN and computing the SHAP explanation…
              </p>
              <p className="text-xs text-ink-500 light:text-ink-400 mt-1.5 max-w-xs">
                This can take up to 30 seconds — the explanation step is computationally
                heavy on CPU.
              </p>
            </Card>
          ) : result ? (
            <Card className="p-6 sm:p-8">
              <PredictionResult result={result} disease="heart" diseaseLabel="heart disease" formValues={submitted} />
            </Card>
          ) : !error ? (
            <Card className="p-10 flex flex-col items-center justify-center text-center h-full min-h-[320px]">
              <HeartPulse size={28} className="text-ink-600 mb-3" />
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
