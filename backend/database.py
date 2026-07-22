import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "healthguard.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition TEXT NOT NULL,
            patient_data TEXT NOT NULL,
            risk_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            model_used TEXT NOT NULL,
            top_factors TEXT NOT NULL,
            advisory TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database initialised.")


def save_prediction(condition, patient_data, risk_score,
                    risk_level, model_used, top_factors, advisory):
    import json
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions
        (condition, patient_data, risk_score, risk_level,
         model_used, top_factors, advisory, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        condition,
        json.dumps(patient_data),
        risk_score,
        risk_level,
        model_used,
        json.dumps(top_factors),
        advisory,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def get_all_predictions(limit=50):
    import json
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        r = dict(row)
        r["patient_data"] = json.loads(r["patient_data"])
        r["top_factors"] = json.loads(r["top_factors"])
        result.append(r)
    return result


def get_stats():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total FROM predictions")
    total = cursor.fetchone()["total"]
    cursor.execute("""
        SELECT risk_level, COUNT(*) as count
        FROM predictions
        GROUP BY risk_level
    """)
    risk_dist = {row["risk_level"]: row["count"] for row in cursor.fetchall()}
    cursor.execute("""
        SELECT condition, COUNT(*) as count
        FROM predictions
        GROUP BY condition
    """)
    condition_dist = {row["condition"]: row["count"] for row in cursor.fetchall()}
    conn.close()
    return {
        "total": total,
        "risk_distribution": risk_dist,
        "condition_distribution": condition_dist
    }


if __name__ == "__main__":
    init_db()
    print("Testing save...")
    save_prediction(
        condition="heart",
        patient_data={"age": 54, "sex": 1},
        risk_score=78.5,
        risk_level="High",
        model_used="RandomForestClassifier",
        top_factors={"cp": -0.71, "trestbps": -0.54},
        advisory="High risk patient. Prompt clinical review advised."
    )
    print("Testing stats...")
    stats = get_stats()
    print("Stats:", stats)
    print("Testing history...")
    history = get_all_predictions()
    print("Records found:", len(history))
    print("✅ Database module working.")
    