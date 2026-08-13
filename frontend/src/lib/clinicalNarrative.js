const DIABETES_LABELS = {
  Pregnancies: 'pregnancy count',
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

const PATTERN_RULES = {
  diabetes: { features: ['Glucose', 'BMI', 'Age', 'BloodPressure'], minMatches: 3, name: 'metabolic risk factors' },
  heart: { features: ['Cholesterol', 'RestingBP', 'ST_Slope'], minMatches: 2, name: 'cardiovascular risk factors' },
};

const FOLLOW_UP_TEST = {
  heart: 'a follow-up lipid panel, resting ECG, or cardiology review',
  diabetes: 'a follow-up HbA1c test or fasting glucose panel',
};

const CONNECTORS = ['', 'In addition, ', 'Similarly, ', 'Further, '];

function cap(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function joinList(items) {
  if (items.length === 0) return '';
  if (items.length === 1) return items[0];
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(', ')}, and ${items[items.length - 1]}`;
}

function describeFeatureClause(disease, feature, contribValue, formValues) {
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
      return `${label} was recorded at ${raw}${unit ? ` ${unit}` : ''}, ${desc} (${standard} reference), and was among the factors ${direction} the predicted risk.`;
    }
    return `${label} was recorded at ${raw}${unit ? ` ${unit}` : ''} and contributed to ${direction} the predicted risk.`;
  }

  const label = HEART_LABELS[feature] || feature;
  const unit = HEART_UNITS[feature] || '';
  if (HEART_CATEGORICAL_LABELS[feature]) {
    const readable = HEART_CATEGORICAL_LABELS[feature][raw] || raw;
    const note = HEART_CATEGORICAL_NOTE[feature]?.[raw];
    return `${label} was recorded as ${readable}${note ? ` — ${note}` : ''}, contributing to ${direction} the predicted risk.`;
  }
  if (raw === undefined) return `${label} was a ${direction} factor in this prediction.`;
  const rangeFn = HEART_RANGES[feature];
  if (rangeFn) {
    const [desc, standard] = rangeFn(Number(raw));
    return `${label} was recorded at ${raw}${unit ? ` ${unit}` : ''}, ${desc} (${standard} reference), and was among the factors ${direction} the predicted risk.`;
  }
  return `${label} was recorded at ${raw}${unit ? ` ${unit}` : ''} and contributed to ${direction} the predicted risk.`;
}

function sentenceFromClause(clause, index) {
  const connector = CONNECTORS[Math.min(index, CONNECTORS.length - 1)];
  return connector ? connector + clause : cap(clause);
}

function synthesisSentence(disease, increasing) {
  const rule = PATTERN_RULES[disease];
  if (!rule) return null;
  const matched = increasing.filter((c) => rule.features.includes(c.feature));
  if (matched.length < rule.minMatches) return null;
  const labels = disease === 'diabetes' ? DIABETES_LABELS : HEART_LABELS;
  const names = matched.map((c) => labels[c.feature] || c.feature);
  return `${cap(joinList(names))} co-occurring here forms a recognisable pattern of ${rule.name}, rather than a set of isolated, unrelated readings.`;
}

function closingParagraph(band, disease) {
  const test = FOLLOW_UP_TEST[disease] || 'appropriate follow-up testing';
  if (band === 'High risk') {
    return (
      'Taken together, this pattern — particularly the values sitting outside typical reference ' +
      'ranges above — is worth discussing with a clinician in the near term. Further investigation, ' +
      `such as ${test}, may be appropriate to help clarify whether these signals reflect an ` +
      'underlying condition or a combination of borderline readings that warrant monitoring rather ' +
      'than immediate concern.'
    );
  }
  if (band === 'Moderate risk') {
    return (
      'None of these values are individually alarming, but several sit outside the typical reference ' +
      'range and the combination is worth tracking. A routine check-up — potentially including ' +
      `${test}, at a clinician's discretion — would help establish whether this pattern is ` +
      'stable, improving, or worsening over time.'
    );
  }
  return (
    'The values driving this prediction sit largely within expected reference ranges. Maintaining ' +
    'current habits and attending routine screening remains a reasonable course.'
  );
}

export function buildAdvisory({ disease, formValues, contributions, band }) {
  if (!contributions?.length) return null;

  const increasing = contributions.filter((c) => c.value >= 0).slice(0, 4);
  const offsetting = contributions.filter((c) => c.value < 0).slice(0, 4);

  const toParagraph = (list) =>
    list.length
      ? list
          .map((c, i) => sentenceFromClause(describeFeatureClause(disease, c.feature, c.value, formValues), i))
          .join(' ')
      : null;

  return {
    increasing: toParagraph(increasing),
    offsetting: toParagraph(offsetting),
    synthesis: synthesisSentence(disease, increasing),
    closing: closingParagraph(band, disease),
  };
}
