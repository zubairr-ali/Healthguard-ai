/**
 * Builds a grounded, per-field advisory paragraph from the patient's actual
 * submitted values and the SHAP contributions for this specific prediction —
 * rather than the backend's generic top-3-factor summary, which never sees
 * the raw input values at all (llm_advisor.py only receives feature names
 * and SHAP direction, not the numbers behind them).
 *
 * Thresholds cited are standard clinical reference ranges, named inline so
 * they're defensible rather than invented:
 *   - Glucose: WHO 2-hour OGTT categories
 *   - Blood pressure: ACC/AHA 2017 categories
 *   - BMI: WHO categories
 *   - Cholesterol: NCEP ATP III categories (same standard used elsewhere
 *     in this project's cross-dataset harmonisation)
 * Fields without a widely-agreed public threshold (Insulin, SkinThickness,
 * Pregnancies, DiabetesPedigreeFunction, MaxHR, Oldpeak, and the categorical
 * heart fields) are described by direction and magnitude only — the model's
 * reasoning, not an invented medical cutoff.
 */

const DIABETES_LABELS = {
  Pregnancies: 'pregnancies',
  Glucose: 'glucose',
  BloodPressure: 'diastolic blood pressure',
  SkinThickness: 'skin thickness',
  Insulin: 'insulin',
  BMI: 'BMI',
  DiabetesPedigreeFunction: 'diabetes pedigree score',
  Age: 'age',
};

const DIABETES_RANGES = {
  Glucose: (v) =>
    v < 140
      ? ['within the normal 2-hour OGTT range', 'WHO']
      : v < 200
      ? ['in the impaired-glucose-tolerance range', 'WHO']
      : ['in the diabetic range', 'WHO'],
  BloodPressure: (v) =>
    v < 80
      ? ['within the normal range', 'ACC/AHA']
      : v < 90
      ? ['elevated', 'ACC/AHA']
      : ['in the high range', 'ACC/AHA'],
  BMI: (v) =>
    v < 18.5
      ? ['in the underweight range', 'WHO']
      : v < 25
      ? ['within the normal range', 'WHO']
      : v < 30
      ? ['in the overweight range', 'WHO']
      : ['in the obese range', 'WHO'],
};

const DIABETES_UNITS = {
  Glucose: 'mg/dL', BloodPressure: 'mmHg', SkinThickness: 'mm',
  Insulin: 'mu U/mL', BMI: 'kg/m²', Age: 'years', Pregnancies: '',
  DiabetesPedigreeFunction: '',
};

const HEART_LABELS = {
  Age: 'age', RestingBP: 'systolic blood pressure', Cholesterol: 'cholesterol',
  FastingBS: 'fasting blood sugar', MaxHR: 'maximum heart rate', Oldpeak: 'ST depression',
  Sex: 'sex', ChestPainType: 'chest pain type', RestingECG: 'resting ECG',
  ExerciseAngina: 'exercise-induced angina', ST_Slope: 'ST slope',
};

const HEART_RANGES = {
  RestingBP: (v) =>
    v < 120
      ? ['within the normal range', 'ACC/AHA']
      : v < 130
      ? ['elevated', 'ACC/AHA']
      : v < 140
      ? ['in the Stage 1 hypertension range', 'ACC/AHA']
      : ['in the Stage 2 hypertension range', 'ACC/AHA'],
  Cholesterol: (v) =>
    v < 200
      ? ['within the desirable range', 'NCEP ATP III']
      : v < 240
      ? ['borderline high', 'NCEP ATP III']
      : ['high', 'NCEP ATP III'],
};

const HEART_UNITS = { Age: 'years', RestingBP: 'mmHg', Cholesterol: 'mg/dL', MaxHR: 'bpm', Oldpeak: 'mm' };

const HEART_CATEGORICAL_LABELS = {
  Sex: { M: 'male', F: 'female' },
  ChestPainType: { ATA: 'atypical angina', NAP: 'non-anginal pain', ASY: 'asymptomatic', TA: 'typical angina' },
  RestingECG: { Normal: 'normal', ST: 'ST-T wave abnormality', LVH: 'left ventricular hypertrophy' },
  ExerciseAngina: { Y: 'present', N: 'absent' },
  ST_Slope: { Up: 'upsloping', Flat: 'flat', Down: 'downsloping' },
};

const HEART_CATEGORICAL_NOTE = {
  ChestPainType: { ASY: 'an asymptomatic presentation is notable, since it can mask underlying disease' },
  ST_Slope: {
    Flat: 'a flat ST slope is a pattern associated with elevated cardiac risk in the literature',
    Down: 'a downsloping ST slope is a pattern associated with elevated cardiac risk in the literature',
  },
  ExerciseAngina: { Y: 'exercise-induced angina is a recognised marker of possible ischaemia' },
};

function describeFeature(disease, feature, contribValue, formValues) {
  const positive = contribValue >= 0;
  const direction = positive ? 'increasing' : 'lowering';
  const raw = formValues?.[feature];

  if (disease === 'diabetes') {
    const label = DIABETES_LABELS[feature] || feature;
    const unit = DIABETES_UNITS[feature] || '';
    if (raw === undefined) return `${label} was a ${direction} factor in this prediction.`;
    const rangeFn = DIABETES_RANGES[feature];
    if (rangeFn) {
      const [desc, standard] = rangeFn(Number(raw));
      return `${cap(label)} was recorded at ${raw}${unit ? ` ${unit}` : ''}, ${desc} (${standard} reference), and was among the strongest factors ${direction} the predicted risk.`;
    }
    return `${cap(label)} was recorded at ${raw}${unit ? ` ${unit}` : ''} and contributed to ${direction} the predicted risk.`;
  }

  // heart
  const label = HEART_LABELS[feature] || feature;
  const unit = HEART_UNITS[feature] || '';
  if (HEART_CATEGORICAL_LABELS[feature]) {
    const readable = HEART_CATEGORICAL_LABELS[feature][raw] || raw;
    const note = HEART_CATEGORICAL_NOTE[feature]?.[raw];
    return `${cap(label)} was recorded as ${readable}${note ? ` — ${note}` : ''}, contributing to ${direction} the predicted risk.`;
  }
  if (raw === undefined) return `${label} was a ${direction} factor in this prediction.`;
  const rangeFn = HEART_RANGES[feature];
  if (rangeFn) {
    const [desc, standard] = rangeFn(Number(raw));
    return `${cap(label)} was recorded at ${raw}${unit ? ` ${unit}` : ''}, ${desc} (${standard} reference), and was among the strongest factors ${direction} the predicted risk.`;
  }
  return `${cap(label)} was recorded at ${raw}${unit ? ` ${unit}` : ''} and contributed to ${direction} the predicted risk.`;
}

function cap(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function closingParagraph(band) {
  if (band === 'High risk') {
    return (
      'Taken together, this pattern — particularly the values sitting outside typical reference ' +
      'ranges above — is worth discussing with a clinician in the near term. A clinical work-up ' +
      'would help determine whether these signals reflect an underlying condition or a combination ' +
      'of borderline readings that warrant monitoring rather than immediate concern.'
    );
  }
  if (band === 'Moderate risk') {
    return (
      'None of these values are individually alarming, but several sit outside the typical reference ' +
      'range and the combination is worth tracking. A routine check-up would help establish whether ' +
      'this pattern is stable, improving, or worsening over time.'
    );
  }
  return (
    'The values driving this prediction sit largely within expected reference ranges. Maintaining ' +
    'current habits and attending routine screening remains a reasonable course.'
  );
}

export function buildAdvisory({ disease, formValues, contributions, band }) {
  if (!contributions?.length) return null;
  const top = contributions.slice(0, 3);
  const sentences = top.map((c) => describeFeature(disease, c.feature, c.value, formValues));
  return [...sentences, closingParagraph(band)].join(' ');
}
