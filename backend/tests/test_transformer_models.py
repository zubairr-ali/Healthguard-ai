import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from model import load_heart, load_diabetes, build_base_models, NO_TUNING_MODELS
from deep_tabular_models import (
    split_categorical_continuous, FTTransformerClassifier, TabTransformerClassifier,
    make_tabnet, fit_tabnet_with_early_stopping, TabNetClassifier as DeepTabNetClassifier,
)
from shap_explainer import _make_explainer, _positive_class_shap

# Small, fast settings — these tests exist to prove fit/predict/predict_proba
# wire together correctly on real data, not to reproduce production accuracy,
# so epoch counts and model width are cut down from build_base_models' defaults
# (dim=32, depth=3, max_epochs=200) purely to keep the suite fast.
FAST_KW = dict(dim=8, depth=1, heads=2, max_epochs=3, patience=2, batch_size=32)
TRANSFORMER_MODEL_NAMES = ["FT-Transformer", "TabTransformer", "TabNet"]


@pytest.fixture(scope="module")
def heart_data():
    return load_heart()


@pytest.fixture(scope="module")
def diabetes_data():
    return load_diabetes()


def _fit_transformer(name, X, y):
    """Builds and fits one of the three transformer models directly (bypassing
    build_base_models' production-sized defaults) so fit/predict/predict_proba
    can be exercised quickly on real heart/diabetes data."""
    categ_idx, cont_idx, cardinalities = split_categorical_continuous(np.asarray(X))
    if name == "FT-Transformer":
        model = FTTransformerClassifier(categ_idx=categ_idx, cont_idx=cont_idx, cardinalities=cardinalities, **FAST_KW)
        model.fit(X, y)
    elif name == "TabTransformer":
        model = TabTransformerClassifier(categ_idx=categ_idx, cont_idx=cont_idx, cardinalities=cardinalities, **FAST_KW)
        model.fit(X, y)
    elif name == "TabNet":
        model = make_tabnet(categ_idx, cardinalities)
        fit_tabnet_with_early_stopping(model, X, y, max_epochs=5, patience=3)
    else:
        raise ValueError(name)
    return model


# ── build_base_models() / NO_TUNING_MODELS wiring ───────────────────────────

def test_no_tuning_models_contains_all_three_transformers():
    assert {"FT-Transformer", "TabTransformer", "TabNet"} <= NO_TUNING_MODELS


def test_build_base_models_includes_correctly_typed_transformers(heart_data):
    X, y, _ = heart_data
    models = build_base_models(X)
    assert set(TRANSFORMER_MODEL_NAMES) <= set(models.keys())
    assert isinstance(models["FT-Transformer"], FTTransformerClassifier)
    assert isinstance(models["TabTransformer"], TabTransformerClassifier)
    assert isinstance(models["TabNet"], DeepTabNetClassifier)


# ── fit / predict / predict_proba on both datasets ──────────────────────────

@pytest.mark.parametrize("condition", ["heart", "diabetes"])
@pytest.mark.parametrize("model_name", TRANSFORMER_MODEL_NAMES)
def test_transformer_fits_and_predicts(model_name, condition, heart_data, diabetes_data):
    X, y, _ = heart_data if condition == "heart" else diabetes_data

    model = _fit_transformer(model_name, X, y)

    sample = np.asarray(X)[:15]
    preds = model.predict(sample)
    probs = model.predict_proba(sample)

    assert preds.shape == (15,)
    assert set(np.unique(preds)) <= {0, 1}

    assert probs.shape == (15, 2)
    assert probs.min() >= -1e-6
    assert probs.max() <= 1 + 1e-6
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)


# ── SHAP generic/fallback explainer path on a transformer model ────────────

def test_shap_generic_fallback_produces_valid_contributions_for_transformer(heart_data):
    """TreeExplainer rejects FT-Transformer just like it rejects TabPFN, so
    _make_explainer should fall back to the generic permutation explainer and
    still return real, per-feature contributions — not just run without
    crashing."""
    X, y, features = heart_data
    model = _fit_transformer("FT-Transformer", X, y)

    explainer, kind = _make_explainer(model, "heart", features)
    assert kind == "generic"

    sample_row = X.iloc[[0]]
    max_evals = 2 * len(features) + 1
    sv = _positive_class_shap(explainer, kind, sample_row, max_evals=max_evals)

    assert sv.shape == (1, len(features))

    contributions = dict(zip(features, sv[0].tolist()))
    assert set(contributions.keys()) == set(features)
    assert all(np.isfinite(v) for v in contributions.values())
    # Not every contribution should be a no-op zero, and none should blow up
    # to an implausible magnitude — both would indicate the fallback path is
    # silently broken rather than actually explaining the model.
    assert any(abs(v) > 1e-6 for v in contributions.values())
    assert all(abs(v) < 50 for v in contributions.values())
