-- =============================================================================
--  FAKE JOB DETECTION PROJECT
--  File 1: MySQL Schema & Table Creation  (v2.0 – upgraded from SQLite)
--  Author : Sumersing Patil | B.Tech (AI)
--  DB     : MySQL 8.0+
--  Run    : mysql -u jobdetect_user -p fake_job_detection < 01_fake_job_schema.sql
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- DROP existing objects (safe re-run)
-- ─────────────────────────────────────────────────────────────────────────────
DROP TABLE  IF EXISTS prediction_history;
DROP TABLE  IF EXISTS fraud_risk_rules;
DROP TABLE  IF EXISTS job_postings;
DROP VIEW   IF EXISTS vw_fraud_summary;
DROP VIEW   IF EXISTS vw_high_risk_jobs;
DROP VIEW   IF EXISTS vw_industry_risk;

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: job_postings  (main fact table)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE job_postings (

    -- Primary Key
    job_id                 INT            NOT NULL,

    -- Job Details
    title                  VARCHAR(500)   NOT NULL,
    location               VARCHAR(300),
    country                VARCHAR(100),
    department             VARCHAR(200),
    salary_range           VARCHAR(150),
    employment_type        VARCHAR(80),
    required_experience    VARCHAR(100),
    required_education     VARCHAR(100),
    industry               VARCHAR(200),
    `function`             VARCHAR(200),     -- backticks: 'function' is reserved in MySQL

    -- Original Text Fields (LONGTEXT for large job descriptions)
    company_profile        LONGTEXT,
    description            LONGTEXT,
    requirements           LONGTEXT,
    benefits               TEXT,

    -- Binary Flags (original dataset)
    telecommuting          TINYINT(1)     DEFAULT 0,
    has_company_logo       TINYINT(1)     DEFAULT 0,
    has_questions          TINYINT(1)     DEFAULT 0,

    -- Engineered Feature Flags
    has_salary             TINYINT(1)     DEFAULT 0  COMMENT '1 if salary_range is not null',
    has_company_profile    TINYINT(1)     DEFAULT 0,
    has_requirements       TINYINT(1)     DEFAULT 0,
    has_benefits           TINYINT(1)     DEFAULT 0,
    has_department         TINYINT(1)     DEFAULT 0,
    has_urgency_words      TINYINT(1)     DEFAULT 0,

    -- Numeric Features
    desc_length            INT            DEFAULT 0  COMMENT 'character count of description',
    req_length             INT            DEFAULT 0  COMMENT 'character count of requirements',
    title_length           INT            DEFAULT 0,
    profile_completeness   TINYINT        DEFAULT 0  COMMENT '0-6: sum of 6 binary flags',

    -- Target Label
    fraudulent             TINYINT(1)     NOT NULL DEFAULT 0,

    -- Audit timestamp
    inserted_at            DATETIME       DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (job_id)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Main fact table for Fake Job Detection project (17,880 records)';


-- ─────────────────────────────────────────────────────────────────────────────
-- INDEXES  (improve WHERE / GROUP BY / ORDER BY performance)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE INDEX idx_fraudulent           ON job_postings (fraudulent);
CREATE INDEX idx_country              ON job_postings (country);
CREATE INDEX idx_industry             ON job_postings (industry);
CREATE INDEX idx_employment_type      ON job_postings (employment_type);
CREATE INDEX idx_profile_completeness ON job_postings (profile_completeness);
CREATE INDEX idx_has_salary           ON job_postings (has_salary);
CREATE INDEX idx_has_urgency          ON job_postings (has_urgency_words);
CREATE INDEX idx_fraud_country        ON job_postings (fraudulent, country);
CREATE INDEX idx_fraud_industry       ON job_postings (fraudulent, industry);


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: fraud_risk_rules  (rule-based scoring reference)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE fraud_risk_rules (
    rule_id          INT           AUTO_INCREMENT PRIMARY KEY,
    rule_name        VARCHAR(100)  NOT NULL,
    condition_desc   VARCHAR(300)  NOT NULL,
    risk_score       TINYINT       NOT NULL COMMENT 'contribution to overall risk score (0-100)',
    description      VARCHAR(500),
    created_at       DATETIME      DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO fraud_risk_rules (rule_name, condition_desc, risk_score, description) VALUES
    ('No Salary',          'salary_range IS NULL',           30, 'Missing salary is the strongest single fraud indicator (95.5% of fraudulent jobs)'),
    ('No Company Logo',    'has_company_logo = 0',           20, 'Legitimate companies usually have logos (SHAP: -0.26 = strong legit signal)'),
    ('Urgency Language',   'has_urgency_words = 1',          25, 'Words like urgent/guaranteed/apply now are red flags (8x more common in fraud)'),
    ('No Company Profile', 'has_company_profile = 0',        15, 'No company description is suspicious (SHAP = 0.80 top predictor)'),
    ('No Requirements',    'has_requirements = 0',           10, 'No job requirements suggest low-effort fake posting'),
    ('Short Description',  'desc_length < 200',              10, 'Fraudulent jobs avg 857 chars vs legit 1890 chars'),
    ('Telecommuting',      'telecommuting = 1',               5, 'Remote jobs have slightly higher fraud rate'),
    ('Low Profile Score',  'profile_completeness <= 1',      15, 'Score ≤ 1/6 indicates very incomplete posting (fraud rate 26.8%)');


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: prediction_history  (logs all Streamlit app predictions)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE prediction_history (
    id             INT            AUTO_INCREMENT PRIMARY KEY,
    timestamp      DATETIME       DEFAULT CURRENT_TIMESTAMP,
    job_title      VARCHAR(500)   NOT NULL,
    company        VARCHAR(300),
    fraud_prob     DECIMAL(6,4)   NOT NULL  COMMENT 'raw probability 0.0–1.0',
    prediction     ENUM('FRAUD','LEGIT') NOT NULL,
    red_flag_count TINYINT        DEFAULT 0 COMMENT 'number of red flags triggered (0-6)',
    threshold_used DECIMAL(4,2)   DEFAULT 0.35,
    INDEX idx_timestamp  (timestamp),
    INDEX idx_prediction (prediction),
    INDEX idx_prob       (fraud_prob)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='Logs every prediction made via the Streamlit app Job Checker page';


-- ─────────────────────────────────────────────────────────────────────────────
-- VIEW: vw_fraud_summary  (overall KPI snapshot for Power BI)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE VIEW vw_fraud_summary AS
SELECT
    COUNT(*)                                                            AS total_jobs,
    SUM(fraudulent)                                                     AS total_fraud,
    SUM(CASE WHEN fraudulent = 0 THEN 1 ELSE 0 END)                    AS total_legit,
    ROUND(100.0 * SUM(fraudulent) / COUNT(*), 2)                       AS fraud_rate_pct,
    ROUND(AVG(CASE WHEN fraudulent = 1 THEN desc_length END), 0)       AS avg_fake_desc_len,
    ROUND(AVG(CASE WHEN fraudulent = 0 THEN desc_length END), 0)       AS avg_real_desc_len,
    SUM(CASE WHEN has_urgency_words = 1 AND fraudulent = 1 THEN 1 ELSE 0 END)  AS urgency_fraud_count,
    SUM(CASE WHEN has_salary = 0        AND fraudulent = 1 THEN 1 ELSE 0 END)  AS no_salary_fraud_count,
    ROUND(AVG(CASE WHEN fraudulent = 1 THEN profile_completeness END), 2)      AS avg_fraud_completeness,
    ROUND(AVG(CASE WHEN fraudulent = 0 THEN profile_completeness END), 2)      AS avg_legit_completeness
FROM job_postings;


-- ─────────────────────────────────────────────────────────────────────────────
-- VIEW: vw_high_risk_jobs  (all very-high and high-risk records)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE VIEW vw_high_risk_jobs AS
SELECT
    job_id,
    title,
    country,
    industry,
    employment_type,
    CASE WHEN has_salary = 0 THEN 'Not Disclosed' ELSE 'Disclosed' END  AS salary_status,
    has_urgency_words,
    has_company_logo,
    has_company_profile,
    profile_completeness,
    -- Computed risk category (MySQL-compatible, no generated columns)
    CASE
        WHEN has_salary = 0 AND has_company_logo = 0 AND has_urgency_words = 1 THEN 'VERY HIGH RISK'
        WHEN has_salary = 0 AND has_company_logo = 0                            THEN 'HIGH RISK'
        WHEN has_salary = 0 OR  has_urgency_words = 1                           THEN 'MEDIUM RISK'
        ELSE                                                                          'LOW RISK'
    END                                                                  AS risk_category,
    fraudulent
FROM job_postings;


-- ─────────────────────────────────────────────────────────────────────────────
-- VIEW: vw_industry_risk  (industry-level fraud rates for Power BI)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE VIEW vw_industry_risk AS
SELECT
    COALESCE(industry, 'Unknown')                           AS industry,
    COUNT(*)                                                AS total_jobs,
    SUM(fraudulent)                                         AS fraud_count,
    ROUND(100.0 * SUM(fraudulent) / COUNT(*), 2)           AS fraud_rate_pct,
    ROUND(AVG(has_urgency_words) * 100, 1)                 AS urgency_word_pct,
    ROUND(AVG(profile_completeness), 2)                    AS avg_completeness
FROM job_postings
WHERE industry IS NOT NULL
GROUP BY industry
HAVING total_jobs >= 30
ORDER BY fraud_rate_pct DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- VERIFY  – run after data import to confirm all tables populated
-- ─────────────────────────────────────────────────────────────────────────────
-- SELECT * FROM vw_fraud_summary;
-- SELECT risk_category, COUNT(*) AS cnt FROM vw_high_risk_jobs GROUP BY risk_category;
-- SELECT * FROM vw_industry_risk LIMIT 10;
-- SELECT COUNT(*) FROM prediction_history;
-- =============================================================================
