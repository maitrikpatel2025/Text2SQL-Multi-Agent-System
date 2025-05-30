
# 🧠 Text2SQL Multi-Agent System

A powerful, modular, and interpretable system that converts **natural language queries into SQL**, executes them on a **PostgreSQL** database, and optionally performs **Python-based post-processing** and **graph visualization**—all orchestrated using **LangGraph agents** and exposed via a **FastAPI** backend.

---

## 🧩 System Architecture Diagram

Below is the Data Flow Diagram (DFD) that illustrates the multi-agent architecture of this system:

![DFD_AFTER _gRAHP](https://github.com/user-attachments/assets/9a7fbcbb-6272-4f4f-856d-458a5b08e119)


---

## 🚀 Features

- ✅ Natural Language → SQL Translation
- ✅ Metadata-aware query understanding (tables, columns, system info)
- ✅ SQL Query Generation & Validation
- ✅ Intelligent decision-making on when to post-process data
- ✅ Python code generation for complex tasks
- ✅ Graph generation from query results
- ✅ Chat history persistence
- ✅ Modular, agent-based architecture using LangGraph
- ✅ Fully REST API-based (FastAPI)
- ✅ Frontend ready (CORS enabled for `http://localhost:4200`)

---

## 🗂️ Project Structure

```
.
├── main.py                    # FastAPI + LangGraph orchestration
├── graph_agent.py            # GraphPlottingAgent for visualization
├── db_metadata.json          # Table/column metadata
├── static/images/            # Auto-generated graph images
└── requirements.txt          # Python dependencies
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/text2sql-agents.git
cd text2sql-agents
```

### 2. Create a virtual environment and activate

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
```

### 5. Ensure PostgreSQL is running

Make sure the following DB is available:

```ini
host=localhost
port=5432
dbname=APAR_KPI
user=postgres
password=admin
```

Also ensure there's a `Chat_History` table with appropriate columns.

### 6. Run the app

```bash
uvicorn main:app --reload
```

---

## 🔌 API Endpoints

### `POST /query`

Submit a natural language query.

```json
{
  "user_question": "How many machines were active last month?",
  "username": "john_doe"
}
```

### `GET /chat_history?username=john_doe&limit=5`

Returns recent query history.

### `GET /test_metadata`

Runs predefined metadata queries.

### `POST /enhanced-query/`

Directly test the GraphPlottingAgent.

---

## 🖼️ Graph Output

Generated plots are stored under:

```
/static/images/graph_<username>_<timestamp>.png
```

And served at:

```
http://localhost:8000/static/images/<filename>
```

---

## ✅ Sample Questions

- "What is this system?"
- "Show me power consumption for February"
- "How many machines are in each plant this year?"
- "Compare spindle speed across shifts"
- "What columns are in EMS_Daily?"

---

## 📸 Screenshots

### 🧠 Example: Help Query - "What is this Text2SQL tool and how can I use it?"
![image](https://github.com/user-attachments/assets/824f1403-28a7-49da-86c0-26445658aee5)


### 📋 Table Info Query - "How many Tables are there?"
![image](https://github.com/user-attachments/assets/bfde2c2b-28aa-4c2f-b442-7db6d6d4fff6)


### 📊 Top Machines by Power Consumption (Bar Graph)
![image](https://github.com/user-attachments/assets/ecf3e315-3965-4fa6-af86-b6dbb10ab6cf)


### 📈 Comparison Query - Axis Current (NCH1-NCH4)
![image](https://github.com/user-attachments/assets/6b3a87db-4b0d-4430-8516-8823a86bc8d4)


---

## 📦 Dependencies

- `FastAPI`
- `LangChain`, `LangGraph`
- `psycopg2`
- `pandas`, `numpy`, `matplotlib`
- `python-dotenv`
- `uuid`, `base64`, `re`, `json`, etc.

---

## 🧠 Powered by

- **LangGraph** – Multi-agent workflow orchestration  
- **OpenAI GPT-4o-mini** – Prompted agents  
- **PostgreSQL** – Query backend  
- **FastAPI** – RESTful API interface  
- **Matplotlib** – Graph rendering

---

## 📜 License

MIT License © Priyanshu Singh
