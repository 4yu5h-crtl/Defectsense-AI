"""
Database layer — SQLite schema and CRUD operations for DefectSense-AI.
"""
import sqlite3
import os
import pandas as pd
from contextlib import contextmanager

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_connection(db_path=None):
    """Get a SQLite connection."""
    path = db_path or config.DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db(db_path=None):
    """Context manager for database connections."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path=None):
    """Create all tables if they don't exist."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                temperature REAL NOT NULL,
                vibration REAL NOT NULL,
                pressure REAL NOT NULL,
                cycle_time REAL NOT NULL,
                is_defect INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS engineered_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reading_id INTEGER NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value REAL,
                FOREIGN KEY (reading_id) REFERENCES sensor_readings(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reading_id INTEGER NOT NULL,
                station_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                defect_probability REAL NOT NULL,
                predicted_label INTEGER NOT NULL,
                model_version TEXT DEFAULT '1.0',
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (reading_id) REFERENCES sensor_readings(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS genai_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                station_id TEXT NOT NULL,
                root_cause TEXT,
                explanation TEXT,
                recommendations TEXT,
                confidence_score REAL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_readings_station ON sensor_readings(station_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_readings_timestamp ON sensor_readings(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_station ON predictions(station_id)")


def save_sensor_readings(df, db_path=None):
    """Save a DataFrame of sensor readings to the database."""
    with get_db(db_path) as conn:
        df.to_sql("sensor_readings", conn, if_exists="append", index=False)


def load_sensor_readings(station_id=None, limit=None, db_path=None):
    """Load sensor readings from the database."""
    with get_db(db_path) as conn:
        query = "SELECT * FROM sensor_readings"
        params = []
        if station_id:
            query += " WHERE station_id = ?"
            params.append(station_id)
        query += " ORDER BY timestamp"
        if limit:
            query += f" LIMIT {limit}"
        return pd.read_sql_query(query, conn, params=params)


def save_predictions(df, db_path=None):
    """Save predictions to the database."""
    with get_db(db_path) as conn:
        df.to_sql("predictions", conn, if_exists="append", index=False)


def load_predictions(station_id=None, limit=None, db_path=None):
    """Load predictions from the database."""
    with get_db(db_path) as conn:
        query = "SELECT * FROM predictions"
        params = []
        if station_id:
            query += " WHERE station_id = ?"
            params.append(station_id)
        query += " ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"
        return pd.read_sql_query(query, conn, params=params)


def save_genai_report(prediction_id, station_id, root_cause, explanation, recommendations, confidence, db_path=None):
    """Save a GenAI report to the database."""
    with get_db(db_path) as conn:
        conn.execute(
            """INSERT INTO genai_reports (prediction_id, station_id, root_cause, explanation, recommendations, confidence_score)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (prediction_id, station_id, root_cause, explanation, recommendations, confidence)
        )


def load_genai_reports(station_id=None, limit=None, db_path=None):
    """Load GenAI reports from the database."""
    with get_db(db_path) as conn:
        query = "SELECT * FROM genai_reports"
        params = []
        if station_id:
            query += " WHERE station_id = ?"
            params.append(station_id)
        query += " ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"
        return pd.read_sql_query(query, conn, params=params)
