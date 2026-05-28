-- =============================================================================
-- FAKE JOB DETECTION PROJECT
-- File 0: MySQL initial setup — creates the database only.
--
-- Run order: mysql_setup.sql → 01_fake_job_schema.sql → upload CSV via notebook
--
-- Note: The notebook auto-creates the database and table via
-- db_connection.ensure_schema(). Running this file manually is optional —
-- it is provided for users who prefer to provision the database from
-- MySQL Workbench or the mysql CLI.
-- =============================================================================

CREATE DATABASE IF NOT EXISTS fake_job_detection
    DEFAULT CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE fake_job_detection;

-- Verify
SELECT
    SCHEMA_NAME            AS database_name,
    DEFAULT_CHARACTER_SET_NAME AS charset,
    DEFAULT_COLLATION_NAME AS collation
FROM information_schema.SCHEMATA
WHERE SCHEMA_NAME = 'fake_job_detection';
