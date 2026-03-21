# 🛡️ AI Based Fake Job Detection – Resume-Analyzer

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://ai-based-fake-job-detection-resume-analyzer-iatwkbf7soyytuqczt.streamlit.app//)
[![AUC](https://img.shields.io/badge/AUC--ROC-0.984-brightgreen)]()
[![MySQL](https://img.shields.io/badge/MySQL-8.0%2B-blue)](https://mysql.com)

> **Detect fraudulent job postings using NLP + Machine Learning + MySQL Analytics**
---

## 📋 Project Overview

Online job portals face a growing problem: **fake job postings** that exploit job seekers.
This end-to-end project builds a complete detection pipeline using:

- 🔬 **EDA** – Uncover patterns in fraudulent vs legitimate postings
- 🗄️ **MySQL Analytics** – Business intelligence queries in MySQL 8.0+
- 🤖 **ML Models** – TF-IDF + Logistic Regression + SMOTE achieving **AUC = 0.984, Recall = 88.3%**
- 🌐 **Streamlit App v2.0** – Real-time fraud detection with prediction history logging

---

## 📁 Project Structure

```
Fake_Job_Detection/
│
├── 📓 Fake_Job_Detection_Complete.ipynb   ← Complete notebook (EDA + ML + CV + SHAP)
├── 🗄️  01_fake_job_schema.sql             ← MySQL schema + views + indexes
├── 📊  02_fake_job_analysis.sql            ← 10 business analysis queries (MySQL)
├── 🌐  streamlit_app.py                    ← 5-page Streamlit web app v2.0
├── 🔗  db_connection.py                    ← MySQL connection + prediction logging
├── 📋  requirements.txt                    ← Python dependencies (incl. mysql)
├── .streamlit/
│   └── secrets.toml                        ← DB credentials (add to .gitignore!)
│
├── models/
│   ├── best_model.pkl                      ← Trained Logistic Regression (SMOTE)
│   ├── tfidf_vectorizer.pkl                ← Fitted TF-IDF (10k features)
│   ├── numeric_cols.pkl                    ← Feature column list
│   └── model_info.json                     ← ✅ Single source of truth for metrics
│
└── outputs/
    ├── cleaned_job_postings.csv            ← Engineered dataset (CSV)
    ├── top_fraud_features.csv              ← Top fraud-predicting words
    └── charts/                             ← All EDA + model charts (PNG)
```

---

## 📊 Dataset

| Attribute       | Value                              |
|-----------------|------------------------------------|
| Source          | Kaggle – Real or Fake Job Postings |
| Total Records   | 17,880                             |
| Fraudulent      | 866 (4.84%)                        |
| Legitimate      | 17,014 (95.16%)                    |
| Features        | 18 original + 14 engineered        |
| Class Imbalance | 95:5 ratio → handled with SMOTE    |

**Key Engineered Features:**
`has_salary`, `has_company_profile`, `has_requirements`, `has_benefits`,
`has_urgency_words`, `desc_length`, `req_length`, `profile_completeness (0–6)`,
`country`, `combined_text`

> **Note:** `profile_completeness` score = sum of 6 binary flags:
> `has_salary + has_company_profile + has_requirements + has_benefits + has_company_logo + has_questions`

---

## 🤖 Model Results

> ✅ **Single source of truth:** All metrics are from `models/model_info.json`

| Model               | Precision | Recall      | F1-Score    | Accuracy | AUC-ROC     |
|---------------------|-----------|-------------|-------------|----------|-------------|
| **Logistic Regression (v2.0 – SMOTE + threshold=0.35)** | **72.4%** | **88.3% 🏆** | **79.5% 🏆** | 96.9% | **0.984** |
| Logistic Regression (v1.0 – no SMOTE, threshold=0.50)  | 99.0% | 58.9%       | 73.9%       | 98.0% | 0.984 |
| Naive Bayes         | 82.3%     | 53.8% ❌    | 65.0%       | 97.2%    | 0.625 ❌  |
| Random Forest       | 99.0%     | 58.9% ⚠️    | 73.9%       | 98.0%    | 0.989     |

> 🏆 **Best Model for Production:** Logistic Regression v2.0
> - **Why not Random Forest?** RF has high precision (99%) but same recall (58.9%) as LR v1.0 — misses too much fraud
> - **Why not Naive Bayes?** AUC = 0.625 (barely better than random guessing)
> - **5-Fold Stratified CV:** AUC = 0.984 ± 0.003 (robust & reproducible)

---

## 🔑 Key Findings (EDA)

| Finding                              | Legitimate | Fraudulent  | Signal Strength |
|--------------------------------------|-----------|-------------|-----------------|
| Missing Salary                       | 81.7%     | **95.5%**   | ❌ Weak (both high) |
| Missing Company Profile              | 15.2%     | **66.8%**   | ✅ Strong        |
| Contains Urgency Words               | 2.3%      | **18.7%**   | ✅ Strong        |
| Avg Description Length               | 1,890 chars | **857 chars** | ✅ Medium      |


---

## 🗄️ MySQL Setup

### Prerequisites
- MySQL 8.0+ installed
- Python `mysql-connector-python` and `sqlalchemy` packages

### Quick Setup
```bash
# 1. Create database and user
mysql -u root -p < mysql_setup.sql

# 2. Import schema
mysql -u jobdetect_user -p fake_job_detection < 01_fake_job_schema.sql

# 3. Import data via Python
python db_connection.py --upload

# 4. Run analysis queries
mysql -u jobdetect_user -p fake_job_detection < 02_fake_job_analysis.sql
```

### Create `.streamlit/secrets.toml`
```toml
[mysql]
host     = "localhost"
user     = "jobdetect_user"
password = "your_password_here"
database = "fake_job_detection"
port     = 3306
```

> ⚠️ **Add `secrets.toml` to `.gitignore`!** Never commit credentials to GitHub.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Jupyter Notebook
```bash
jupyter notebook Fake_Job_Detection_Complete.ipynb
```

### 3. MySQL Database Setup
```bash
# Start MySQL, create DB, run schema, import data
python db_connection.py --upload
```

### 4. Launch Streamlit App
```bash
streamlit run streamlit_app.py
```

---

## 🌐 Streamlit App Pages

| Page                   | Description                                              |
|------------------------|----------------------------------------------------------|
| 🏠 Home                | Project overview, KPIs, architecture, v1→v2 uplift       |
| 📊 EDA Dashboard       | Interactive Plotly charts: class, geo, text, industry    |
| 🔍 Job Checker         | Real-time fraud prediction, gauge + red flags + history  |
| 🤖 Model Insights      | Comparison, SMOTE explainer, threshold tuning            |
| ⚠️ Limitations & Bias  | Known issues, industry bias, Houston anomaly, roadmap    |

---

## 🛠️ Tech Stack

| Category        | Tools                                          |
|-----------------|------------------------------------------------|
| Language        | Python 3.9+                                    |
| Data            | Pandas, NumPy                                  |
| NLP             | Scikit-learn TF-IDF, WordCloud                 |
| ML              | Logistic Regression, Naive Bayes, RandomForest |
| Imbalance       | imbalanced-learn (SMOTE)                       |
| Explainability  | SHAP LinearExplainer                           |
| Validation      | StratifiedKFold (5-fold CV)                    |
| Visualisation   | Matplotlib, Seaborn, Plotly                    |
| **Database**    | **MySQL 8.0** (upgraded from SQLite)           |
| Web App         | Streamlit v2.0                                 |
| Deployment      | Streamlit Cloud / GitHub                       |

---

## ⚠️ Known Limitations

1. **Industry Bias:** "oil", "gas" flagged as fraud words → legitimate O&G jobs may be falsely flagged
2. **English Only:** Non-English postings handled poorly by TF-IDF (future: multilingual BERT)
3. **Static Model:** No automated retraining pipeline (future: monthly cron job)
4. **Houston Anomaly:** Small sample (89 postings) inflates city fraud rate to 33.7%

---

## 📞 Contact

### Developer Information

**Sumersung Patil**
- 🐙 GitHub: [Sumersingpatil2694](https://github.com/Sumersingpatil2694)
- 💼 LinkedIn: [Sumersing Patil](linkedin.com/in/sumersing-patil-839674234)
- 📧 Email: sumerrajput0193@gmail.com
- 🐦 Twitter: [X](https://x.com/SumerRajput2694)

---
