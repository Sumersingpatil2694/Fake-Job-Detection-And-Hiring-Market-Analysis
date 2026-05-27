-- =============================================================================
-- FAKE JOB DETECTION PROJECT
-- File 1: MySQL Schema Setup
-- Architecture: Notebook → Clean CSV → MySQL → SQL Analysis → Streamlit App
--
-- FIXES applied (v5 — FINAL):
--   1. CREATE DATABASE IF NOT EXISTS at top → file is self-bootstrapping.
--      You no longer need to run mysql_setup.sql first.
--   2. ALTER TABLE block upgrades an existing DB without DROP/re-create.
--   3. department VARCHAR(200) → VARCHAR(500) (root cause fix).
--   4. All VARCHAR limits increased to match db_connection.py v4.
--   5. vw_industry_risk: ORDER BY removed from view body (MySQL ignores it).
--   6. DROP VIEW statements appear before DROP TABLE (FK safety).
-- =============================================================================

CREATE DATABASE IF NOT EXISTS fake_job_detection
    DEFAULT CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE fake_job_detection;

-- =============================================================================
-- STEP 0: ALTER EXISTING TABLE (safe upgrade for already-deployed databases)
-- =============================================================================
ALTER TABLE IF EXISTS job_postings
    MODIFY COLUMN title                VARCHAR(500)    NOT NULL,
    MODIFY COLUMN location             VARCHAR(500),
    MODIFY COLUMN country              VARCHAR(200),
    MODIFY COLUMN department           VARCHAR(500),
    MODIFY COLUMN salary_range         VARCHAR(300),
    MODIFY COLUMN employment_type      VARCHAR(150),
    MODIFY COLUMN required_experience  VARCHAR(200),
    MODIFY COLUMN required_education   VARCHAR(200),
    MODIFY COLUMN industry             VARCHAR(500),
    MODIFY COLUMN `function`           VARCHAR(500);

-- =============================================================================
-- STEP 1: DROP EXISTING OBJECTS (SAFE RE-RUN / FRESH INSTALL)
-- =============================================================================
DROP VIEW IF EXISTS vw_salary_fraud_analysis;
DROP VIEW IF EXISTS vw_employment_type_fraud_analysis;
DROP VIEW IF EXISTS vw_country_fraud_analysis;
DROP VIEW IF EXISTS vw_fraud_summary;
DROP VIEW IF EXISTS vw_high_risk_jobs;
DROP VIEW IF EXISTS vw_industry_risk;
DROP TABLE IF EXISTS job_postings;

-- =============================================================================
-- STEP 2: MAIN TABLE
-- =============================================================================
CREATE TABLE job_postings (
    job_id                INT             NOT NULL,
    title                 VARCHAR(500)    NOT NULL,
    location              VARCHAR(500),
    country               VARCHAR(200),
    department            VARCHAR(500),
    salary_range          VARCHAR(300),
    employment_type       VARCHAR(150),
    required_experience   VARCHAR(200),
    required_education    VARCHAR(200),
    industry              VARCHAR(500),
    `function`            VARCHAR(500),

    company_profile       LONGTEXT,
    description           LONGTEXT,
    requirements          LONGTEXT,
    benefits              TEXT,

    telecommuting         TINYINT(1)  DEFAULT 0,
    has_company_logo      TINYINT(1)  DEFAULT 0,
    has_questions         TINYINT(1)  DEFAULT 0,
    has_salary            TINYINT(1)  DEFAULT 0,
    has_company_profile   TINYINT(1)  DEFAULT 0,
    has_requirements      TINYINT(1)  DEFAULT 0,
    has_benefits          TINYINT(1)  DEFAULT 0,
    has_department        TINYINT(1)  DEFAULT 0,
    has_urgency_words     TINYINT(1)  DEFAULT 0,

    desc_length           INT         DEFAULT 0,
    req_length            INT         DEFAULT 0,
    title_length          INT         DEFAULT 0,
    profile_completeness  TINYINT     DEFAULT 0,

    fraudulent            TINYINT(1)  NOT NULL DEFAULT 0,
    inserted_at           DATETIME    DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (job_id)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;

-- =============================================================================
-- STEP 3: INDEXES
-- =============================================================================
CREATE INDEX idx_fraudulent           ON job_postings (fraudulent);
CREATE INDEX idx_country              ON job_postings (country);
CREATE INDEX idx_industry             ON job_postings (industry);
CREATE INDEX idx_employment_type      ON job_postings (employment_type);
CREATE INDEX idx_profile_completeness ON job_postings (profile_completeness);
CREATE INDEX idx_has_salary           ON job_postings (has_salary);
CREATE INDEX idx_has_urgency          ON job_postings (has_urgency_words);
CREATE INDEX idx_fraud_country        ON job_postings (fraudulent, country);
CREATE INDEX idx_fraud_industry       ON job_postings (fraudulent, industry);

-- =============================================================================
-- STEP 4: VIEWS
-- =============================================================================

-- VIEW 1: OVERALL FRAUD SUMMARY
CREATE VIEW vw_fraud_summary AS
SELECT
    COUNT(*)                                                              AS total_jobs,
    SUM(fraudulent)                                                       AS total_fraud,
    SUM(CASE WHEN fraudulent = 0 THEN 1 ELSE 0 END)                      AS total_legit,
    ROUND(100.0 * SUM(fraudulent) / COUNT(*), 2)                         AS fraud_rate_pct,
    ROUND(AVG(CASE WHEN fraudulent = 1 THEN desc_length END), 0)         AS avg_fake_desc_len,
    ROUND(AVG(CASE WHEN fraudulent = 0 THEN desc_length END), 0)         AS avg_real_desc_len,
    SUM(CASE WHEN has_urgency_words = 1 AND fraudulent = 1 THEN 1 ELSE 0 END) AS urgency_fraud_count,
    SUM(CASE WHEN has_salary = 0        AND fraudulent = 1 THEN 1 ELSE 0 END) AS no_salary_fraud_count,
    ROUND(AVG(CASE WHEN fraudulent = 1 THEN profile_completeness END), 2) AS avg_fraud_completeness,
    ROUND(AVG(CASE WHEN fraudulent = 0 THEN profile_completeness END), 2) AS avg_legit_completeness
FROM job_postings;

-- VIEW 2: HIGH RISK JOBS
CREATE VIEW vw_high_risk_jobs AS
SELECT
    job_id, title, country, industry, employment_type,
    CASE WHEN has_salary = 0 THEN 'Not Disclosed' ELSE 'Disclosed' END AS salary_status,
    has_urgency_words, has_company_logo, has_company_profile,
    profile_completeness,
    CASE
        WHEN has_salary = 0 AND has_company_logo = 0 AND has_urgency_words = 1 THEN 'VERY HIGH RISK'
        WHEN has_salary = 0 AND has_company_logo = 0                           THEN 'HIGH RISK'
        WHEN has_salary = 0 OR  has_urgency_words = 1                          THEN 'MEDIUM RISK'
        ELSE 'LOW RISK'
    END AS risk_category,
    fraudulent
FROM job_postings;

-- VIEW 3: INDUSTRY RISK
CREATE VIEW vw_industry_risk AS
SELECT
    COALESCE(industry, 'Unknown')                        AS industry,
    COUNT(*)                                             AS total_jobs,
    SUM(fraudulent)                                      AS fraud_count,
    ROUND(100.0 * SUM(fraudulent) / COUNT(*), 2)        AS fraud_rate_pct,
    ROUND(AVG(has_urgency_words) * 100, 1)              AS urgency_word_pct,
    ROUND(AVG(profile_completeness), 2)                 AS avg_completeness
FROM job_postings
WHERE industry IS NOT NULL
GROUP BY industry
HAVING COUNT(*) >= 30;

-- VIEW 4: COUNTRY FRAUD ANALYSIS
CREATE VIEW vw_country_fraud_analysis AS
SELECT
    COALESCE(NULLIF(TRIM(country), ''), 'Unknown')       AS country,
    COUNT(*)                                             AS total_jobs,
    SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END)     AS fraud_jobs,
    ROUND(100.0 * SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS fraud_rate_pct
FROM job_postings
GROUP BY COALESCE(NULLIF(TRIM(country), ''), 'Unknown')
HAVING COUNT(*) >= 20;

-- VIEW 5: EMPLOYMENT TYPE FRAUD
CREATE VIEW vw_employment_type_fraud_analysis AS
SELECT
    COALESCE(NULLIF(TRIM(employment_type), ''), 'Not Specified') AS employment_type,
    COUNT(*)                                                      AS total_jobs,
    SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END)              AS fraud_jobs,
    ROUND(100.0 * SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS fraud_rate_pct
FROM job_postings
GROUP BY COALESCE(NULLIF(TRIM(employment_type), ''), 'Not Specified')
HAVING COUNT(*) >= 20;

-- VIEW 6: SALARY DISCLOSURE FRAUD
CREATE VIEW vw_salary_fraud_analysis AS
SELECT
    CASE WHEN has_salary = 1 THEN 'Salary Disclosed' ELSE 'Salary Not Disclosed' END AS salary_status,
    COUNT(*)                                             AS total_jobs,
    SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END)     AS fraud_jobs,
    ROUND(100.0 * SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS fraud_rate_pct
FROM job_postings
GROUP BY has_salary;

-- =============================================================================
-- STEP 5: VERIFY (uncomment after data import)
-- =============================================================================
-- SELECT COUNT(*) AS total_rows FROM job_postings;          -- expect 17,880
-- SELECT * FROM vw_fraud_summary;
-- SELECT risk_category, COUNT(*) FROM vw_high_risk_jobs GROUP BY risk_category;
-- SELECT * FROM vw_industry_risk   ORDER BY fraud_rate_pct DESC LIMIT 10;
