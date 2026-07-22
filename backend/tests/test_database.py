import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db, save_prediction, get_all_predictions, get_stats


def test_save_and_retrieve_prediction():
    init_db()
    before = get_stats()["total"]

    save_prediction(
        condition="heart",
        patient_data={"Age": 54, "Sex": "M"},
        risk_score=88.5,
        risk_level="High",
        model_used="XGBClassifier",
        top_factors={"ST_Slope": 2.0},
        advisory="Test advisory note."
    )

    after = get_stats()["total"]
    assert after == before + 1


def test_history_returns_records():
    records = get_all_predictions(limit=5)
    assert isinstance(records, list)
    if records:
        assert "risk_score" in records[0]
        assert "condition" in records[0]