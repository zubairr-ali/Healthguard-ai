import { Sparkles, ShieldAlert } from 'lucide-react';
import RiskGauge from './RiskGauge';
import ShapBars from './ShapBars';
import { Badge } from './ui';
import { buildAdvisory } from '../lib/clinicalNarrative';

function summarySentence(diseaseLabel, probability, band) {
  const pct = (probability * 100).toFixed(1);
  return `Based on the values provided, the model estimates a ${pct}% probability consistent with ${diseaseLabel} risk factors — placing this case in the ${band.toLowerCase()} band.`;
}

function bandFor(p) {
  if (p < 0.3) return 'Low risk';
  if (p < 0.6) return 'Moderate risk';
  return 'High risk';
}

export default function PredictionResult({ result, disease, diseaseLabel = 'elevated', formValues }) {
  if (!result) return null;
  const { probability, model_used, contributions } = result;
  const validProbability = typeof probability === 'number' && !Number.isNaN(probability);
  const band = validProbability ? bandFor(probability) : null;
  const advisory = validProbability
    ? buildAdvisory({ disease, formValues, contributions, band })
    : null;

  return (
    <div>
      <div className="flex flex-col items-center text-center pb-8 mb-8 border-b border-ink-800 light:border-ink-100">
        {validProbability ? (
          <>
            <RiskGauge probability={probability} />
            <p className="mt-5 max-w-md text-[15px] text-ink-200 light:text-ink-700 leading-relaxed">
              {summarySentence(diseaseLabel, probability, band)}
            </p>
          </>
        ) : (
          <div className="flex flex-col items-center gap-3 py-6">
            <ShieldAlert size={28} className="text-amber-400" />
            <p className="text-sm text-ink-300 light:text-ink-600 max-w-sm">
              The model returned a result the interface couldn't parse into a risk
              percentage. This usually means the backend response format doesn't match
              what the frontend expects — worth checking the API response shape.
            </p>
          </div>
        )}
        {model_used && (
          <Badge tone="neutral" className="mt-4">
            <Sparkles size={12} /> Predicted using {model_used}
          </Badge>
        )}
      </div>

      {contributions?.length > 0 && (
        <div className="mb-8">
          <h4 className="text-sm font-semibold text-ink-100 light:text-ink-800 mb-1">
            What drove this prediction
          </h4>
          <p className="text-xs text-ink-500 light:text-ink-400 mb-4">
            SHAP feature contributions, ranked by magnitude. Bars extending right increase
            predicted risk; bars extending left decrease it.
          </p>
          <ShapBars contributions={contributions} />
        </div>
      )}

      {advisory && (
        <div className="mb-8">
          <h4 className="text-sm font-semibold text-ink-100 light:text-ink-800 mb-2">
            Advisory note
          </h4>
          <p className="text-sm text-ink-300 light:text-ink-600 leading-relaxed">{advisory}</p>
        </div>
      )}

      <p className="text-xs text-ink-500 light:text-ink-400 leading-relaxed pt-5 border-t border-ink-800 light:border-ink-100">
        This estimate is decision support generated from a statistical model trained on
        historical clinical data. It is not a medical diagnosis, does not account for
        information outside the fields provided above, and should not be used in place of
        professional medical advice. Always consult a qualified clinician about any health
        concern.
      </p>
    </div>
  );
}
