-- =============================================================================
-- Fake Job Detection Project
-- SQL Fraud Analysis Queries
-- Database: fake_job_detection
-- Table: job_postings
-- Dependency: Run after 01_fake_job_schema.sql and data import
-- Purpose: Business-ready SQL analysis for MySQL 8.0+ and Power BI
-- =============================================================================

USE fake_job_detection;

-- =============================================================================
-- Q1: Overall Fraud Summary
-- Business Goal: Measure total job volume, fake jobs, real jobs and fraud rate
-- =============================================================================
SELECT
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) AS total_fraud_jobs,
    SUM(CASE WHEN fraudulent = 0 THEN 1 ELSE 0 END) AS total_real_jobs,
    ROUND(
        100.0 * SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS fraud_rate_pct
FROM job_postings;


-- =============================================================================
-- Q2: Employment Type Fraud Analysis
-- Business Goal: Identify which employment types are more vulnerable to fraud
-- Note: Minimum record filter added for realistic analysis
-- =============================================================================
SELECT
    COALESCE(NULLIF(TRIM(employment_type), ''), 'Not Specified') AS employment_type,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) AS fraud_jobs,
    ROUND(
        100.0 * SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS fraud_rate_pct
FROM job_postings
GROUP BY COALESCE(NULLIF(TRIM(employment_type), ''), 'Not Specified')
HAVING COUNT(*) >= 20
ORDER BY fraud_rate_pct DESC, total_jobs DESC;


-- =============================================================================
-- Q3: Industry Fraud Analysis
-- Business Goal: Find high-risk industries for fraud monitoring and dashboarding
-- =============================================================================
SELECT
    COALESCE(NULLIF(TRIM(industry), ''), 'Unknown') AS industry,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) AS fraud_jobs,
    ROUND(
        100.0 * SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS fraud_rate_pct
FROM job_postings
GROUP BY COALESCE(NULLIF(TRIM(industry), ''), 'Unknown')
HAVING COUNT(*) >= 30
ORDER BY fraud_rate_pct DESC, total_jobs DESC;


-- =============================================================================
-- Q4: Country-wise Fraud Analysis
-- Business Goal: Support geo-level fraud tracking in Power BI maps/charts
-- =============================================================================
SELECT
    COALESCE(NULLIF(TRIM(country), ''), 'Unknown') AS country,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) AS fraud_jobs,
    ROUND(
        100.0 * SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS fraud_rate_pct
FROM job_postings
GROUP BY COALESCE(NULLIF(TRIM(country), ''), 'Unknown')
HAVING COUNT(*) >= 20
ORDER BY fraud_rate_pct DESC, total_jobs DESC;


-- =============================================================================
-- Q5: Salary Disclosure vs Fraud
-- Business Goal: Check whether missing salary information is a fraud indicator
-- =============================================================================
SELECT
    CASE
        WHEN has_salary = 1 THEN 'Salary Disclosed'
        ELSE 'Salary Not Disclosed'
    END AS salary_status,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) AS fraud_jobs,
    SUM(CASE WHEN fraudulent = 0 THEN 1 ELSE 0 END) AS real_jobs,
    ROUND(
        100.0 * SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS fraud_rate_pct
FROM job_postings
GROUP BY has_salary
ORDER BY fraud_rate_pct DESC;


-- =============================================================================
-- Q6: Company Profile & Posting Completeness Analysis
-- Business Goal: Evaluate whether incomplete company information increases fraud risk
-- =============================================================================
SELECT
    CASE
        WHEN has_company_profile = 1 THEN 'Company Profile Available'
        ELSE 'Company Profile Missing'
    END AS company_profile_status,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) AS fraud_jobs,
    ROUND(AVG(profile_completeness), 2) AS avg_profile_completeness,
    ROUND(AVG(desc_length), 0) AS avg_description_length,
    ROUND(
        100.0 * SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS fraud_rate_pct
FROM job_postings
GROUP BY has_company_profile
ORDER BY fraud_rate_pct DESC;


-- =============================================================================
-- Q7: Urgency Words Analysis
-- Business Goal: Test whether urgent language is a strong red flag for fraud
-- =============================================================================
SELECT
    CASE
        WHEN has_urgency_words = 1 THEN 'Urgency Words Present'
        ELSE 'No Urgency Words'
    END AS urgency_word_status,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) AS fraud_jobs,
    ROUND(
        100.0 * SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS fraud_rate_pct
FROM job_postings
GROUP BY has_urgency_words
ORDER BY fraud_rate_pct DESC;


-- =============================================================================
-- Q8: Remote Jobs vs Fraud
-- Business Goal: Compare telecommuting jobs with non-remote jobs
-- =============================================================================
SELECT
    CASE
        WHEN telecommuting = 1 THEN 'Remote / Telecommuting'
        ELSE 'Non-Remote / On-site'
    END AS remote_status,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) AS fraud_jobs,
    ROUND(
        100.0 * SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS fraud_rate_pct
FROM job_postings
GROUP BY telecommuting
ORDER BY fraud_rate_pct DESC;


-- =============================================================================
-- Q9: Top Fraudulent Job Titles
-- Business Goal: Highlight titles with repeated fraud exposure for visualization
-- Note: Minimum record filter avoids misleading low-volume titles
-- =============================================================================
SELECT
    COALESCE(NULLIF(TRIM(title), ''), 'Unknown Title') AS job_title,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) AS fraud_jobs,
    ROUND(
        100.0 * SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS fraud_rate_pct
FROM job_postings
GROUP BY COALESCE(NULLIF(TRIM(title), ''), 'Unknown Title')
HAVING COUNT(*) >= 5
ORDER BY fraud_jobs DESC, fraud_rate_pct DESC, total_jobs DESC
LIMIT 15;


-- =============================================================================
-- Q10: Fraud Risk Ranking (CTE + Window Function)
-- Business Goal: Create a realistic red-flag scoring model for interview discussion
-- Advanced SQL Used: CTE, CASE expressions, RANK() window function
-- =============================================================================
WITH fraud_stats AS (
    SELECT
        job_id,
        title,
        COALESCE(NULLIF(TRIM(country), ''), 'Unknown') AS country,
        COALESCE(NULLIF(TRIM(industry), ''), 'Unknown') AS industry,
        employment_type,
        fraudulent,
        profile_completeness,
        desc_length,
        (
            CASE WHEN has_salary = 0 THEN 30 ELSE 0 END +
            CASE WHEN has_company_profile = 0 THEN 20 ELSE 0 END +
            CASE WHEN has_company_logo = 0 THEN 15 ELSE 0 END +
            CASE WHEN has_requirements = 0 THEN 10 ELSE 0 END +
            CASE WHEN has_urgency_words = 1 THEN 15 ELSE 0 END +
            CASE WHEN telecommuting = 1 THEN 5 ELSE 0 END +
            CASE WHEN profile_completeness <= 1 THEN 10 ELSE 0 END +
            CASE WHEN desc_length < 200 THEN 10 ELSE 0 END
        ) AS fraud_risk_score
    FROM job_postings
),
risk_ranked AS (
    SELECT
        job_id,
        title,
        country,
        industry,
        COALESCE(NULLIF(TRIM(employment_type), ''), 'Not Specified') AS employment_type,
        fraudulent,
        profile_completeness,
        desc_length,
        fraud_risk_score,
        CASE
            WHEN fraud_risk_score >= 60 THEN 'High Risk'
            WHEN fraud_risk_score >= 35 THEN 'Medium Risk'
            ELSE 'Low Risk'
        END AS risk_level,
        RANK() OVER (
            ORDER BY fraud_risk_score DESC, profile_completeness ASC, job_id ASC
        ) AS risk_rank
    FROM fraud_stats
)
SELECT
    risk_rank,
    job_id,
    title,
    country,
    industry,
    employment_type,
    fraud_risk_score,
    risk_level,
    fraudulent
FROM risk_ranked
ORDER BY risk_rank, job_id
LIMIT 25;


-- =============================================================================
-- ADVANCED VIEW 1: Country Fraud View for Power BI
-- =============================================================================
DROP VIEW IF EXISTS vw_country_fraud_analysis;

CREATE VIEW vw_country_fraud_analysis AS
SELECT
    COALESCE(NULLIF(TRIM(country), ''), 'Unknown') AS country,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) AS fraud_jobs,
    ROUND(
        100.0 * SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS fraud_rate_pct
FROM job_postings
GROUP BY COALESCE(NULLIF(TRIM(country), ''), 'Unknown')
HAVING COUNT(*) >= 20;


-- =============================================================================
-- ADVANCED VIEW 2: Employment Type Fraud View for Power BI
-- =============================================================================
DROP VIEW IF EXISTS vw_employment_type_fraud_analysis;

CREATE VIEW vw_employment_type_fraud_analysis AS
SELECT
    COALESCE(NULLIF(TRIM(employment_type), ''), 'Not Specified') AS employment_type,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) AS fraud_jobs,
    ROUND(
        100.0 * SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS fraud_rate_pct
FROM job_postings
GROUP BY COALESCE(NULLIF(TRIM(employment_type), ''), 'Not Specified')
HAVING COUNT(*) >= 20;


-- =============================================================================
-- ADVANCED VIEW 3: Salary Disclosure Fraud View for Power BI
-- =============================================================================
DROP VIEW IF EXISTS vw_salary_fraud_analysis;

CREATE VIEW vw_salary_fraud_analysis AS
SELECT
    CASE
        WHEN has_salary = 1 THEN 'Salary Disclosed'
        ELSE 'Salary Not Disclosed'
    END AS salary_status,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) AS fraud_jobs,
    ROUND(
        100.0 * SUM(CASE WHEN fraudulent = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS fraud_rate_pct
FROM job_postings
GROUP BY has_salary;


-- =============================================================================
-- OPTIONAL CHECK QUERIES FOR REPORTING / DASHBOARD QA
-- =============================================================================
-- SELECT * FROM vw_fraud_summary;
-- SELECT * FROM vw_industry_risk ORDER BY fraud_rate_pct DESC;
-- SELECT * FROM vw_country_fraud_analysis ORDER BY fraud_rate_pct DESC;
-- SELECT * FROM vw_employment_type_fraud_analysis ORDER BY fraud_rate_pct DESC;
-- SELECT * FROM vw_salary_fraud_analysis ORDER BY fraud_rate_pct DESC;
