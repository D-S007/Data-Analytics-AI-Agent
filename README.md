# 🧠 Data Analytics AI Agent

> An AI-powered data analytics assistant that automates the entire data lifecycle — from ingestion to transformation, querying, visualization, and reporting.

Built with Python, FastAPI, Streamlit, and PostgreSQL, this agent serves as an end-to-end intelligent companion for data analysts, enabling seamless data management through automation and natural-language interaction.

---

## 🧩 Tech Stack

| Category | Tools |
|----------|-------|
| Languages | Python |
| Frameworks | FastAPI, Streamlit |
| Libraries | Pandas, NumPy, SQLAlchemy, LangChain |
| Database | PostgreSQL |
| Key Concepts | ETL Automation, NLP Querying, Data Validation, Visualization, Reporting |

---

## 🚀 Core Features

- **🧮 Multi-Source Data Input** — Supports CSV/XLSX uploads, API connections, and database linking.
- **🧹 Automated ETL Pipeline** — Cleans, validates, and transforms datasets to ensure data quality and schema consistency.
- **🧠 NLP-Based SQL Querying** — Query datasets in plain English, automatically converted into SQL.
- **📊 Dynamic Visualization** — Generate interactive charts and plots instantly.
- **📄 Automated Reporting** — Exports summarized insights as downloadable PDF/CSV reports.
- **🔐 Governed Workflows** — Modular backend design for secure and auditable data operations.

---

## 🎥 Demo

> ▶️ Watch on YouTube: https://youtu.be/zXhgjuuniIU?si=GSOOFEOWkTKzLj1H

💬 This version demonstrates the complete end-to-end workflow on a local environment. Cloud deployment (Streamlit/Google Cloud) is under final testing for public release.

---

## 🏗️ System Architecture Overview

```

┌────────────────────────────────────────┐
│Frontend (UI)             │
│Streamlit Application          │
└────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────┐
│Backend (API Layer)              │
│FastAPI – Handles data flow & NLP SQL  │
└────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────┐
│Data Processing Engine           │
│ETL Pipeline • Validation • Reporting  │
└────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────┐
│PostgreSQL Database             │
│Structured Storage & Query Execution    │
└────────────────────────────────────────┘

```

---

## 📂 Project Structure

```

Data-Analytics-AI-Agent/
├──app.py          #streamlit app
├──modules
├── init.py
├── ai_services.py
├── data_cleaning.py
├── data_ingestion.py
├── database_manager.py
├── profiling.py
└── visualization.py
├──requirements.txt      #Project dependencies
└──utils
├── init.py
└── helpers.py
└──README.md
├── .devcontainer
└── devcontainer.json
├──.gitignore
├──LICENSE

```

---

## 🧠 How It Works

1. **Ingest** data from multiple sources (upload, API, or DB).
2. **Validate and Clean** via automated ETL pipelines.
3. **Query** using natural language → converted into SQL.
4. **Visualize** patterns and metrics instantly.
5. **Export Reports** in multiple formats (Interactive-Web-report/Cleaned-CSV-data-export).

---

## 📈 Future Enhancements

- ☁️ Streamlit Cloud / GCP Deployment for scalable access.
- 🧾 Role-Based Access Control (RBAC) for governed environments.
- ⏱ Automated Scheduling & Reporting Pipelines.
- 🗃 Integration with BigQuery or Hive for big-data scalability.

---

## 👨‍💻 Author

**David Singh**  
📍 Passionate Data Analyst | Focused on Big Data Analytics, AI Agents, and Workflow Automation

- LinkedIn: [Visit here](https://www.linkedin.com/in/david-singh-96830324a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
- GitHub: [See Projects here](https://github.com/D-S007)
- Email: singhdavid036@gmail.com

---

## 📜 License

MIT License © 2025 David S.  
Feel free to fork, contribute, or enhance this project.