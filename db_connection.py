"""
# ─────────────────────────────────────────────────────────────────────────────
# db_connection.py  –  MySQL Connection & Prediction History  v4.0
# Author : Sumersing Patil | B.Tech (AI)
# FIXED  : Password encoding (urllib.parse.quote_plus)
#          get_history_stats returns proper dict even when DB empty
#          SQLAlchemy 2.x compatible (text() wrapping)
#          CLI --csv default path updated
# ─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
import warnings
import urllib.parse
warnings.filterwarnings("ignore")

import pandas as pd

# ── Try MySQL imports (graceful degradation) ──────────────────────────────
try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    MYSQL_CONNECTOR_AVAILABLE = True
except ImportError:
    MYSQL_CONNECTOR_AVAILABLE = False

try:
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# ── Try Streamlit (only when running inside Streamlit app) ────────────────
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def _get_config() -> dict:
    """
    Priority order:
    1. Streamlit secrets  (.streamlit/secrets.toml)
    2. Environment variables  (MYSQL_HOST, MYSQL_USER, etc.)
    3. Local defaults (localhost dev)
    """
    # 1. Streamlit secrets
    if STREAMLIT_AVAILABLE:
        try:
            return {
                "host"    : st.secrets["mysql"]["host"],
                "user"    : st.secrets["mysql"]["user"],
                "password": st.secrets["mysql"]["password"],
                "database": st.secrets["mysql"]["database"],
                "port"    : int(st.secrets["mysql"].get("port", 3306)),
            }
        except Exception:
            pass

    # 2. Environment variables
    if os.environ.get("MYSQL_HOST"):
        return {
            "host"    : os.environ.get("MYSQL_HOST", "localhost"),
            "user"    : os.environ.get("MYSQL_USER", "root"),
            "password": os.environ.get("MYSQL_PASSWORD", ""),
            "database": os.environ.get("MYSQL_DATABASE", "fake_job_detection"),
            "port"    : int(os.environ.get("MYSQL_PORT", 3306)),
        }

    # 3. Local defaults
    return {
        "host"    : "localhost",
        "user"    : "jobdetect_user",
        "password": "jobdetect_pass",
        "database": "fake_job_detection",
        "port"    : 3306,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE & CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_engine():
    """
    Returns a SQLAlchemy engine.
    FIXED: password is URL-encoded with urllib.parse.quote_plus
           to handle special characters (@, #, !, etc.)
    """
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("sqlalchemy not installed. Run: pip install sqlalchemy")

    cfg      = _get_config()
    password = urllib.parse.quote_plus(cfg['password'])   # ← FIXED
    url      = (
        f"mysql+mysqlconnector://"
        f"{cfg['user']}:{password}"
        f"@{cfg['host']}:{cfg['port']}"
        f"/{cfg['database']}?charset=utf8mb4"
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)


def get_connection():
    """
    Returns a raw mysql.connector connection for DDL/DML.
    """
    if not MYSQL_CONNECTOR_AVAILABLE:
        raise ImportError("mysql-connector-python not installed. Run: pip install mysql-connector-python")

    cfg = _get_config()
    return mysql.connector.connect(
        host       = cfg["host"],
        user       = cfg["user"],
        password   = cfg["password"],
        database   = cfg["database"],
        port       = cfg["port"],
        charset    = "utf8mb4",
        autocommit = True,
    )


def test_connection() -> bool:
    """Returns True if MySQL connection succeeds."""
    try:
        conn = get_connection()     
        conn.close()
        return True
    except Exception as e:
        print(f"[db_connection] MySQL connection failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# QUERY RUNNER  (FIXED: SQLAlchemy 2.x compatible)
# ─────────────────────────────────────────────────────────────────────────────

def run_query(sql: str, params: dict = None) -> pd.DataFrame:
    """
    Execute a SELECT query and return a DataFrame.
    FIXED: SQLAlchemy 2.x requires text() wrapping and .mappings().fetchall()
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            rows   = result.fetchall()
            cols   = result.keys()
            return pd.DataFrame(rows, columns=list(cols))
    except Exception as e:
        print(f"[run_query] Error: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HISTORY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def create_prediction_table():
    """Creates the prediction_history table if it doesn't exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS prediction_history (
        id             INT           AUTO_INCREMENT PRIMARY KEY,
        timestamp      DATETIME      DEFAULT CURRENT_TIMESTAMP,
        job_title      VARCHAR(500)  NOT NULL,
        company        VARCHAR(300),
        fraud_prob     DECIMAL(6,4)  NOT NULL,
        prediction     ENUM('FRAUD','LEGIT') NOT NULL,
        red_flag_count TINYINT       DEFAULT 0,
        threshold_used DECIMAL(4,2)  DEFAULT 0.35,
        INDEX idx_timestamp  (timestamp),
        INDEX idx_prediction (prediction),
        INDEX idx_prob       (fraud_prob)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    try:
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute(ddl)
        conn.commit()
        cursor.close()
        conn.close()
        print("[create_prediction_table] ✅ Table ready")
    except Exception as e:
        print(f"[create_prediction_table] Error: {e}")


def log_prediction(
    title         : str,
    company       : str,
    prob          : float,
    is_fraud      : bool,
    red_flag_count: int,
    threshold     : float = 0.35,
) -> bool:
    """
    Insert one prediction record.
    Returns True on success, False on failure.
    """
    sql = """
        INSERT INTO prediction_history
            (job_title, company, fraud_prob, prediction, red_flag_count, threshold_used)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    try:
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute(sql, (
            (title[:500]   if title   else "Unknown"),
            (company[:300] if company else ""),
            round(float(prob), 4),
            "FRAUD" if is_fraud else "LEGIT",
            int(red_flag_count),
            float(threshold),
        ))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[log_prediction] Error: {e}")
        return False


def get_prediction_history(limit: int = 50) -> pd.DataFrame:
    """Fetch recent prediction history records."""
    sql = """
        SELECT
            id,
            timestamp,
            job_title,
            company,
            ROUND(fraud_prob * 100, 1) AS fraud_prob_pct,
            prediction,
            red_flag_count
        FROM prediction_history
        ORDER BY timestamp DESC
        LIMIT :limit
    """
    return run_query(sql, {"limit": limit})


def get_history_stats() -> dict:
    """
    Returns summary statistics for prediction history.
    FIXED: Returns proper default dict when DB is empty or unavailable.
    """
    default = {
        "total_checks" : 0,
        "fraud_detected": 0,
        "legit_detected": 0,
        "avg_prob_pct"  : 0.0,
        "fraud_rate_pct": 0.0,
        "last_check"    : "Never",
    }

    sql = """
        SELECT
            COUNT(*)                                          AS total_checks,
            SUM(prediction = 'FRAUD')                        AS fraud_detected,
            SUM(prediction = 'LEGIT')                        AS legit_detected,
            ROUND(AVG(fraud_prob) * 100, 1)                  AS avg_prob_pct,
            ROUND(SUM(prediction='FRAUD') / COUNT(*) * 100, 1) AS fraud_rate_pct,
            MAX(timestamp)                                    AS last_check
        FROM prediction_history
    """
    try:
        result = run_query(sql)
        if result.empty or result.iloc[0]['total_checks'] is None:
            return default
        row = result.iloc[0].to_dict()
        # Fill None values with defaults
        for key, dval in default.items():
            if row.get(key) is None:
                row[key] = dval
        return row
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# CSV → MYSQL UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

def upload_csv_to_mysql(
    csv_path  : str = "outputs/cleaned_job_postings.csv",
    table_name: str = "job_postings",
    if_exists : str = "replace",
    chunksize : int = 500,
) -> int:
    """Upload cleaned CSV to MySQL."""
    print(f"[upload_csv] Reading {csv_path}…")
    df = pd.read_csv(csv_path, low_memory=False)

    # Fix dtypes for MySQL
    bool_cols = [
        "telecommuting", "has_company_logo", "has_questions", "fraudulent",
        "has_salary", "has_company_profile", "has_requirements",
        "has_benefits", "has_department", "has_urgency_words",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    int_cols = ["job_id", "desc_length", "req_length", "profile_completeness", "title_length"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Drop combined_text (large, not needed for BI queries)
    drop_cols = [c for c in ["combined_text"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    print(f"[upload_csv] Uploading {len(df):,} rows → '{table_name}'…")
    engine = get_engine()
    df.to_sql(
        name      = table_name,
        con       = engine,
        if_exists = if_exists,
        index     = False,
        chunksize = chunksize,
        method    = "multi",
    )
    print(f"[upload_csv] ✅ Done! {len(df):,} rows uploaded.")
    return len(df)


# ─────────────────────────────────────────────────────────────────────────────
# SETUP SQL
# ─────────────────────────────────────────────────────────────────────────────

MYSQL_SETUP_SQL = """
-- Run this as MySQL root user ONCE before using the project:
-- mysql -u root -p < mysql_setup.sql

CREATE DATABASE IF NOT EXISTS fake_job_detection
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'jobdetect_user'@'localhost'
    IDENTIFIED BY 'jobdetect_pass';

GRANT ALL PRIVILEGES ON fake_job_detection.*
    TO 'jobdetect_user'@'localhost';

FLUSH PRIVILEGES;
"""


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake Job Detection – DB Helper v4.0")
    parser.add_argument("--test",   action="store_true", help="Test MySQL connection")
    parser.add_argument("--upload", action="store_true", help="Upload CSV to MySQL")
    parser.add_argument("--setup",  action="store_true", help="Print MySQL setup SQL")
    parser.add_argument("--csv",    default="outputs/cleaned_job_postings.csv",
                        help="CSV path (default: outputs/cleaned_job_postings.csv)")
    args = parser.parse_args()

    if args.setup:
        print(MYSQL_SETUP_SQL)
        sys.exit(0)

    if args.test:
        ok = test_connection()
        print(f"Connection: {'✅ SUCCESS' if ok else '❌ FAILED'}")
        sys.exit(0 if ok else 1)

    if args.upload:
        rows = upload_csv_to_mysql(args.csv)
        create_prediction_table()
        print(f"✅ Setup complete: {rows:,} rows + prediction_history table created.")
        sys.exit(0)

    print("Usage: python db_connection.py [--test | --upload | --setup]")
    print(f"       Config: {_get_config()}")
