import os
import json
import re
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, date
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple, TypedDict, Annotated, Sequence, Optional, Literal, Union
import operator
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles # Added
from fastapi.responses import FileResponse # Added
from pydantic import BaseModel
import uuid # Added
import base64 # Added

from graph_agent import GraphPlottingAgent

# LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import matplotlib.pyplot as plt
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

app = FastAPI()
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # or "*" for all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static folder for storing images if it doesn't exist
STATIC_IMAGES_DIR = "static/images"
os.makedirs(STATIC_IMAGES_DIR, exist_ok=True) 

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Load metadata
with open('db_metadata.json', 'r') as f:
    metadata = json.load(f)

# Database configuration
db_config = {
    "host": "localhost",
    "port": "5432",
    "dbname": "APAR_KPI",
    "user": "postgres",
    "password": "admin"
}

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
graph_agent = GraphPlottingAgent(openai_api_key=OPENAI_API_KEY)

# Constants
CURRENT_DATE = datetime.today().date()
CURRENT_YEAR = CURRENT_DATE.year

class QueryRequest(BaseModel):
    user_question: str
    username: str = "guest"

class QueryResponse(BaseModel):
    Explanation: str
    Table: Any # Can be a list of dicts or "NA"
    Graph: str
    GraphImage: Optional[str] = None  # New field for image URL
    GraphDetails: Optional[Dict] = None  # New field for graph metadata

class ChatHistoryItem(BaseModel):
    User_Question: str
    # Timestamp: Optional[datetime] # Keep it if you parse it back to datetime
    # Table: Optional[Any] # Keep it if you parse it

class ChatHistoryResponse(BaseModel):
    success: bool
    data: Optional[List[Dict]] = None # Keeping as List[Dict] as per original parsing
    error: Optional[str] = None
    message: Optional[str] = None

class MetadataTestResponse(BaseModel):
    success: bool
    explanation: Optional[str] = None
    table: Optional[Any] = None
    graph: Optional[str] = None
    error: Optional[str] = None

class EnhancedQueryRequest(BaseModel):
    user_question: str
    explanation: str
    table_data: Any
    graph_suggestion: str
    username: str = "guest"

class EnhancedQueryResponse(BaseModel):
    success: bool
    explanation: str
    table: Any
    graph_suggestion: str
    plot_result: Optional[Dict] = None
    error_message: Optional[str] = None
    


# Helper functions
def execute_sql_query(query, db_config):
    """Execute SQL query and return results"""
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute(query)

        if cur.description:  # Check if query returns data
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            result = {"success": True, "columns": columns, "rows": rows}
        else:
            result = {"success": True, "message": f"Query executed. Rows affected: {cur.rowcount}"}

        conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()
    return result

def format_results_structured(results, max_rows=100):
    """
    Converts SQL result to a list of dictionaries (JSON-style),
    truncates to max_rows for safety.
    """
    if not results["success"] or "rows" not in results:
        return results

    columns = results["columns"]
    rows = results["rows"][:max_rows]  # limit number of rows
    structured_data = [dict(zip(columns, row)) for row in rows]
    return structured_data

def convert_datetime_to_str(data):
    """Convert datetime objects to strings in a list of dictionaries"""
    if not data or not isinstance(data, list) or not isinstance(data[0], dict):
        return data

    # Find which keys (if any) contain datetime or date objects in the first dict
    datetime_keys = [k for k, v in data[0].items() if isinstance(v, (datetime, date))]

    # If no datetime/date keys found, return data as-is
    if not datetime_keys:
        return data

    # Convert all datetime/date keys to strings in all dicts
    for entry in data:
        for key in datetime_keys:
            val = entry.get(key)
            if isinstance(val, datetime):
                entry[key] = val.strftime('%d-%m-%Y %H:%M:%S')
            elif isinstance(val, date):
                entry[key] = val.strftime('%d-%m-%Y')

    return data

def extract_sql_from_message(message):
    """Extract SQL query from message"""
    match = re.search(r"```sql\n(.*?)```", message, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_code_from_message(message):
    """Extract Python code from message"""
    match = re.search(r"```python\n(.*?)```", message, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def save_graph_image(graph_agent_result: Dict, username: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Save graph image to static folder and return URL and metadata."""
    try:
        if not graph_agent_result or not isinstance(graph_agent_result, dict):
            print("No graph agent result to save.")
            return None, None

        # plot_result might contain image_data or the figure itself
        plot_result = graph_agent_result.get("plot_result", {})
        if not plot_result:
            print("No plot_result in graph agent output.")
            return None, None

        image_data = plot_result.get('image_data') # Could be base64
        figure = plot_result.get('figure') # Could be a matplotlib figure or PIL Image

        if not image_data and not figure:
            print("No image data or figure found in plot_result.")
            return None, None

        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"graph_{username}_{timestamp}_{unique_id}.png"
        filepath = os.path.join(STATIC_IMAGES_DIR, filename)

        if image_data and isinstance(image_data, str):
            # Assuming base64 encoded string
            try:
                # Ensure correct padding for base64
                missing_padding = len(image_data) % 4
                if missing_padding:
                    image_data += '=' * (4 - missing_padding)
                image_bytes = base64.b64decode(image_data)
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                print(f"Image saved from base64 data to {filepath}")
            except base64.binascii.Error as e:
                print(f"Error decoding base64 image: {e}. Data: {image_data[:100]}...")
                return None, None
            except Exception as e:
                print(f"Error saving base64 image: {e}")
                return None, None
        elif figure:
            # If it's a PIL Image or matplotlib figure
            if hasattr(figure, 'save'): # For PIL Image
                figure.save(filepath)
                print(f"Image saved from PIL/figure object to {filepath}")
            elif hasattr(figure, 'savefig'): # For Matplotlib figure
                figure.savefig(filepath)
                plt.close(figure) # Close the figure to free memory
                print(f"Image saved from Matplotlib figure to {filepath}")
            else:
                print(f"Unknown figure type: {type(figure)}")
                return None, None
        else:
            print(f"Unsupported image data format. Neither image_data string nor figure object provided.")
            return None, None

        image_url = f"/static/images/{filename}"
        graph_metadata = {
            "filename": filename,
            "url": image_url,
            "timestamp": datetime.now().isoformat(),
            "plot_type": plot_result.get('plot_type', 'unknown'),
            "success": graph_agent_result.get('success', False),
            "message": graph_agent_result.get('message', plot_result.get('message', 'Graph generated.')),
            "x_label": plot_result.get('x_label'),
            "y_label": plot_result.get('y_label'),
            "title": plot_result.get('title')
        }
        return image_url, graph_metadata

    except Exception as e:
        print(f"❌ Error saving graph image: {e}")
        return None, None


def insert_chat_history(username, user_question, explanation, graph, table, db_config, graph_image_url: Optional[str] = None):
    """Insert chat interaction into history table, now including GraphImage."""
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        # Attempt to add GraphImage column if it doesn't exist
        try:
            cur.execute('''
                ALTER TABLE "Chat_History"
                ADD COLUMN IF NOT EXISTS "GraphImage" TEXT;
            ''')
            conn.commit()
        except Exception as alter_e:
            print(f"Notice: Could not ensure 'GraphImage' column exists (may require manual DB migration or permissions): {alter_e}")
            conn.rollback() # Rollback alter attempt if it failed

        insert_query = '''
            INSERT INTO "Chat_History" ("Username", "User_Question", "Explanation", "Graph", "Table", "GraphImage", "Timestamp")
            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        '''
        cur.execute(insert_query, (username, user_question, explanation, graph, json.dumps(table), graph_image_url))
        conn.commit()
    except Exception as e:
        print(f"❌ Error inserting chat history: {e}")
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

def get_metadata_context():
    """Get metadata context for prompts"""
    # Format metadata for better readability in prompts
    machines_info = []
    for item in metadata.get('machines_metadata', []):
        machines_info.append(f"- {item['column_name']}: {item['description']} (type: {item['column_datatype']})")

    schema_info = []
    for item in metadata.get('db_schema_metadata', []):
        schema_info.append(f"- {item['column_name']}: {item['description']} (type: {item['column_datatype']})")

    table_info = []
    for item in metadata.get('table_metadata', []):
        table_info.append(f"- {item['table_name']}")

    return {
        "machines_metadata": "\n".join(machines_info),
        "db_schema_metadata": "\n".join(schema_info),
        "table_metadata": "\n".join(table_info),
        "CURRENT_DATE": str(CURRENT_DATE),
        "CURRENT_YEAR": CURRENT_YEAR
    }

# Define the state structure as a TypedDict
class Text2SQLState(TypedDict):
    question: str
    username: str
    structured_intent: Optional[dict]
    sql_query: Optional[str]
    query_result: Optional[dict]
    post_processing_required: Optional[bool]
    python_code: Optional[str]
    final_result: Optional[dict] # Can be SQL result or Python processed result
    explanation: Optional[dict] # Contains Explanation, Table, Graph, and potentially GraphImage, GraphDetails

print("============Chat History================")

# Initialize LLM models
def get_llm(temperature=0):
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        api_key=OPENAI_API_KEY
    )

# Enhanced Planner Prompt with Metadata Query Detection
# Enhanced Planner Prompt with Time Range Detection
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a helpful assistant that breaks down user questions into structured intent.

    Your job is to identify and return a JSON object with the following fields:
    - action: (e.g., "metadata_query", "count_distinct", "show_trend", "compare", "find_average", "list_data", "aggregate_data")
    - data_source: table name based on granularity ("EMS_Daily", "EMS_Hourly", "EMS_Shiftwise", or "METADATA" for system queries)
    - filters: object with filter conditions
    - time_range: object with start_date and end_date in "YYYY-MM-DD" format
    - columns: array of exact column names needed from the schema
    - aggregation: object with aggregation details (if needed)
    - metadata_type: for metadata queries ("system_info", "tables_info", "columns_info", "help")

    TIME RANGE RULES (CRITICAL - MUST FOLLOW):
    * Always extract explicit dates in "YYYY-MM-DD" format.
    * For **month-only or vague references** like "February", "Jan", or "this month":
        * Always assume the **latest such month in the current year**
        * Use the current year: **{CURRENT_YEAR}**
        * Examples:
            * "February" → start_date = "{CURRENT_YEAR}-02-01", end_date = "{CURRENT_YEAR}-02-28" (or 29 if leap year)
            * "January" → start_date = "{CURRENT_YEAR}-01-01", end_date = "{CURRENT_YEAR}-01-31"
            * "March" → start_date = "{CURRENT_YEAR}-03-01", end_date = "{CURRENT_YEAR}-03-31"
    * For relative terms (based on today's date {CURRENT_DATE}):
        * "last week" → 7 days before {CURRENT_DATE} to {CURRENT_DATE}
        * "last month" → previous month in current year
        * "this month" → current month from 1st to {CURRENT_DATE}
        * "this year" → From "{CURRENT_YEAR}-01-01" to "{CURRENT_DATE}"
        * "yesterday" → 1 day before {CURRENT_DATE}
        * "today" → {CURRENT_DATE} to {CURRENT_DATE}
    * ❗ NEVER assume year 2023 or earlier unless explicitly stated.
    * ❗ Do NOT hallucinate dates; resolve based on {CURRENT_DATE} provided at runtime.
    * ❗ If no time reference is mentioned, set both start_date and end_date to null.

    TIME RANGE EXAMPLES:
    - "February" (when current year is {CURRENT_YEAR}): {{"start_date": "{CURRENT_YEAR}-02-01", "end_date": "{CURRENT_YEAR}-02-28"}}
    - "January 2024": {{"start_date": "2024-01-01", "end_date": "2024-01-31"}}
    - "last month" (when current date is {CURRENT_DATE}): Calculate previous month in current year
    - "this year": {{"start_date": "{CURRENT_YEAR}-01-01", "end_date": "{CURRENT_DATE}"}}
    - No time mentioned: {{"start_date": null, "end_date": null}}

    METADATA QUERY DETECTION:
    Detect when user asks about the system itself:
    - "what is this", "what is text2sql", "how does this work" → metadata_type: "system_info"
    - "what tables", "how many tables", "available tables" → metadata_type: "tables_info"
    - "what columns", "available columns", "schema", "structure" → metadata_type: "columns_info"
    - "help", "what can I ask", "examples" → metadata_type: "help"

    ACTION DETECTION RULES:
    - "metadata_query": When user asks about system, tables, columns, help
    - "count_distinct": When user asks "how many", "count of", "number of" for unique items
    - "aggregate_data": When user wants totals, sums, averages, grouped calculations
    - "show_trend": When user asks for trends over time, changes, patterns
    - "compare": When user wants to compare between different entities
    - "find_average": When user specifically asks for mean/average values
    - "list_data": When user wants to see raw data or simple listings

    METADATA KEYWORDS:
    - System: "what is this", "text2sql", "how does this work", "what can this do"
    - Tables: "tables", "how many tables", "available tables", "what tables"
    - Columns: "columns", "fields", "schema", "structure", "what columns", "available columns"
    - Help: "help", "examples", "what can I ask", "how to use", "guide"

    AGGREGATION DETECTION:
    When user asks questions like:
    - "how many machines for each division" → COUNT + GROUP BY
    - "total production by plant" → SUM + GROUP BY
    - "average speed per machine" → AVG + GROUP BY
    - "count of alarms by shift" → COUNT + GROUP BY

    For aggregation queries, include an "aggregation" object:
    {{
        "type": "COUNT" | "SUM" | "AVG" | "MAX" | "MIN",
        "column": "column_to_aggregate",
        "distinct": true/false,
        "group_by": ["column1", "column2"]
    }}

    TABLE SELECTION RULES:
    - Use "METADATA" for system/schema questions
    - Use "EMS_Daily" if the user mentions daily, per day, or general trends
    - Use "EMS_Hourly" if the user mentions hourly, hour-wise, or every hour
    - Use "EMS_Shiftwise" if the user mentions shifts, Shift A/B/C, or shift comparison
    - Default to "EMS_Daily" if not specified for data queries

    AVAILABLE COLUMNS:
    {db_schema_metadata}

    AVAILABLE TABLES:
    {table_metadata}

    EXAMPLES:

    User: "what is this text2sql system"
    Response:
    {{
        "action": "metadata_query",
        "data_source": "METADATA",
        "filters": {{}},
        "time_range": {{"start_date": null, "end_date": null}},
        "columns": [],
        "aggregation": null,
        "metadata_type": "system_info"
    }}

    User: "show me power data for February" (when current year is {CURRENT_YEAR})
    Response:
    {{
        "action": "list_data",
        "data_source": "EMS_Daily",
        "filters": {{}},
        "time_range": {{"start_date": "{CURRENT_YEAR}-02-01", "end_date": "{CURRENT_YEAR}-02-28"}},
        "columns": ["time", "MachineName", "NCH22", "NCH23"],
        "aggregation": null,
        "metadata_type": null
    }}

    User: "what was the maximum power consumption in January 2024"
    Response:
    {{
        "action": "find_average",
        "data_source": "EMS_Daily",
        "filters": {{}},
        "time_range": {{"start_date": "2024-01-01", "end_date": "2024-01-31"}},
        "columns": ["MachineName", "NCH22", "NCH23"],
        "aggregation": {{
            "type": "MAX",
            "column": "NCH23",
            "distinct": false,
            "group_by": ["MachineName"]
        }},
        "metadata_type": null
    }}

    User: "how many machines are there for each division this year"
    Response:
    {{
        "action": "count_distinct",
        "data_source": "EMS_Daily",
        "filters": {{}},
        "time_range": {{"start_date": "{CURRENT_YEAR}-01-01", "end_date": "{CURRENT_DATE}"}},
        "columns": ["MachineName", "DivisionName"],
        "aggregation": {{
            "type": "COUNT",
            "column": "MachineName",
            "distinct": true,
            "group_by": ["DivisionName"]
        }},
        "metadata_type": null
    }}

    User: "show trends for last month"
    Response:
    {{
        "action": "show_trend",
        "data_source": "EMS_Daily",
        "filters": {{}},
        "time_range": {{"start_date": "calculate_previous_month_start", "end_date": "calculate_previous_month_end"}},
        "columns": ["time", "MachineName", "NCH1", "NCH2"],
        "aggregation": null,
        "metadata_type": null
    }}

    IMPORTANT REMINDERS:
    - Current date: {CURRENT_DATE}
    - Current year: {CURRENT_YEAR}
    - Never use 2023 or earlier years unless explicitly mentioned
    - Always resolve relative dates based on current date/year
    - For month names without year, always use current year {CURRENT_YEAR}

    Return your response as a valid JSON object.
    """),
    ("human", "{question}"),
])

# Enhanced Generator Prompt with Aggregation Handling
generator_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a helpful assistant that generates accurate PostgreSQL queries from natural language questions using structured metadata.

    SQL GENERATION OBJECTIVE:
    - Generate valid PostgreSQL queries with correct syntax, data types, and filters.
    - Wrap all table names and column names in double quotes.
    - Only use column names and types defined in the schema below.
    - Use correct numeric, timestamp, and string types as defined in the metadata.

    TABLE USAGE RULES:
    - Use only these tables: "EMS_Daily", "EMS_Hourly", "EMS_Shiftwise".
    - Always use "EMS_Daily" as the default table.
    - Use "EMS_Hourly" only when the user's intent mentions "hour", "hourly", or per-hour analysis.
    - Use "EMS_Shiftwise" only if the user asks for shift-wise data or comparisons.
    - Never join across EMS tables. Use only one EMS table per query.

    AGGREGATION HANDLING:
    When the structured_intent contains an "aggregation" object:
    1. Use the specified aggregation function (COUNT, SUM, AVG, MAX, MIN)
    2. If "distinct": true, use COUNT(DISTINCT "column_name")
    3. If "group_by" is specified, add GROUP BY clause with those columns
    4. Include GROUP BY columns in SELECT clause
    5. Order results for better readability

    AGGREGATION EXAMPLES:
    - COUNT distinct machines by division:
      SELECT "DivisionName", COUNT(DISTINCT "MachineName") as machine_count
      FROM "EMS_Daily"
      GROUP BY "DivisionName"
      ORDER BY machine_count DESC;

    - COUNT distinct machines by division and plant:
      SELECT "DivisionName", "PlantName", COUNT(DISTINCT "MachineName") as machine_count
      FROM "EMS_Daily"
      GROUP BY "DivisionName", "PlantName"
      ORDER BY "DivisionName", "PlantName";

    FILTERING & MATCHING RULES:
    - For string fields like "MachineName", "ProcessName", "PlantName", "DivisionName", "SubPlantName":
    - Use case-insensitive filters: LOWER("column_name") = LOWER('value')
    - When user mentions specific names like "Godavari", "UNIT153", determine the appropriate column:
      * Machine names → "MachineName"
      * Plant names → "PlantName"
      * Division names → "DivisionName"
      * Process names → "ProcessName"

    COLUMN MAPPING RULES:
    - Use exact column names from schema, not descriptions
    - For metrics, map user requests to appropriate NCH columns:
      * Current/Amperage → NCH1, NCH2, NCH3, NCH4, NCH26, NCH27
      * Load → NCH5, NCH6, NCH7, NCH8, NCH28, NCH29
      * Speed → NCH9, NCH10, NCH11, NCH12, NCH30
      * Temperature → NCH13, NCH14, NCH15, NCH16, NCH21
      * Spindle related → NCH18, NCH19, NCH20, NCH21
      * Power → NCH22, NCH23
      * Status → NCH24
      * Alarms → NCH25

    DISTINCT vs NON-DISTINCT:
    - Use DISTINCT when counting unique items (machines, plants, divisions)
    - Don't use DISTINCT for sum/average calculations on metrics
    - For "how many machines" → COUNT(DISTINCT "MachineName")
    - For "total production" → SUM("ProductionColumn") [no DISTINCT]

    STRICT RULES:
    - Only use tables from table_metadata
    - Only use columns from db_schema_metadata
    - Do NOT invent new tables or columns
    - Always specify exact columns needed (no SELECT *)

    AVAILABLE COLUMNS:
    {db_schema_metadata}

    AVAILABLE TABLES:
    {table_metadata}
    """),
    ("human", """
    Based on the following structured intent, generate a PostgreSQL query.

    Original question: {question}
    Structured intent: {structured_intent}

    Pay special attention to the "aggregation" field in the structured intent. If it exists:
    - Use the specified aggregation type (COUNT, SUM, AVG, etc.)
    - Apply DISTINCT if "distinct": true
    - Add GROUP BY clause if "group_by" is specified
    - Include grouped columns in SELECT clause
    """),
])

validator_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a SQL Validator Agent in a Text2SQL system. Your job is to validate a PostgreSQL query generated by another agent.

    Your job is to:
    - Ensure the SQL is syntactically and semantically correct
    - Confirm that the SQL uses valid tables from the allowed list
    - Uses only known columns from the schema
    - Applies correct data types (as per db_schema_metadata)
    - Applies case-insensitive filtering using LOWER() for all name-based filters
    - Fix any column name issues or syntax errors

    SQL VALIDATION RULES:
    - Allowed tables: "EMS_Daily", "EMS_Hourly", "EMS_Shiftwise"
    - Use only columns defined in db_schema_metadata
    - Do not invent columns or use anything not listed in the metadata
    - Use only one EMS table per query. Never join multiple EMS tables.
    - All table and column names must be wrapped in double quotes
    - Use proper PostgreSQL syntax

    COLUMN VALIDATION:
    - Verify all column names exist in the schema
    - Check data types match (numeric, string, timestamp)
    - Ensure proper filtering syntax for different data types

    Return the corrected, executable SQL query.

    AVAILABLE COLUMNS:
    {db_schema_metadata}

    AVAILABLE TABLES:
    {table_metadata}
    """),
    ("human", """
    Original question: {question}
    Structured intent: {structured_intent}
    Generated SQL query: {sql_query}

    Please validate this query and return the final, corrected SQL query.
    """),
])

decision_maker_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a DecisionMaker agent. Your job is to determine if further Python processing is needed to fulfill the user's intent.

    Analyze the user's question and the SQL query results to decide if:
    1. The SQL results directly answer the user's question (NO_POST_PROCESSING_REQUIRED)
    2. Additional processing, calculations, or visualizations are needed (POST_PROCESSING_REQUIRED)

    Return a JSON response with a "decision" field containing either "POST_PROCESSING_REQUIRED" or "NO_POST_PROCESSING_REQUIRED".
    """),
    ("human", """
    User question: {question}
    SQL Query: {sql_query}
    Query results: {query_result}

    Should I perform post-processing on these results?
    """),
])

code_generator_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a Python Code Generator agent in a multi-agent Text2SQL system.
    Your role is to generate Python code (typically using pandas, matplotlib, seaborn, or numpy) to perform post-processing on data retrieved from a SQL query.

    Your responsibilities:
    - Create the required dataframe (df) from the provided SQL query result, so that no extra inputs are needed
    - Understand the user's intent from the question
    - Generate clean, readable Python code that processes the data accordingly
    - Use pandas for data manipulation
    - If visualization is requested (e.g., trend, outliers), return the parameters with actual data along with plot type which can be used to make plot on frontend
    - Instead of adding code lines for plotting graphs, just return the raw data which can be used to plot graphs
    - Ensure the code is self-contained and does not include file paths or external dependencies
    - Due to token limitations, only process and validate the rows which are in the token limit
    - If the data or code is too long, skip irrelevant parts and add a placeholder like "# ... skipped for brevity"

    Return your code enclosed in triple backticks like this: ```python
    # Your Python code here
    ```
    """),
    ("human", """
    User question: {question}
    SQL query: {sql_query}
    SQL query result: {query_result}

    Please generate Python code to process this data according to the user's intent.
    """),
])

code_validator_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a senior-level Python developer and data science expert tasked with reviewing and validating automatically generated Python code.

    Your goals:
    1. Correctness: Ensure the code is syntactically valid and runs without errors using the given DataFrame
    2. Context Awareness: Carefully analyze whether the Python code addresses the user's question appropriately
    3. Clean and Safe Code: Remove redundant steps, follow standard library usage conventions
    4. Plotting Rules: If the code generates a plot, don't write code for plot generation, instead create raw data such as x, y, type of plot which can be used to make graphs

    Return your validated/corrected code enclosed in triple backticks like this: ```python
    # Your Python code here
    ```
    """),
    ("human", """
    User question: {question}
    SQL query result: {query_result}
    Generated Python code: {python_code}

    Please validate and correct this code if necessary.
    """),
])

code_executor_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a Python Executor agent in a Text2SQL + Post-processing system.

    Your role is to:
    1. Execute the given Python code
    2. Capture any meaningful output, such as:
       - Final processed DataFrame
       - Aggregated metrics (e.g., mean, max)
       - Raw data needed for plotting graph e.g. x, y, df, type of plot etc.

    3. Return the results in a clear and structured format as a JSON object with these fields:
       - "execution_status": "success" or "error"
       - "result_data": The processed data or error message
       - "visualization_data": Raw data for plotting (if applicable)
       - "plot_type": Type of plot recommended (if applicable)

    Due to token limitations, only process and validate the rows which are in the token limit.
    """),
    ("human", """
    User question: {question}
    Python code to execute: {python_code}
    SQL query result (for df creation): {query_result}

    Please execute this code and return the results.
    """),
])

interpreter_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a result interpreter for a Text2SQL system. Your job is to transform raw query or processed results into clear, human-readable explanations.

    Number Formatting rules:
    - < 1 million: Show as-is (e.g., 12,345)
    - 1 million - 999 million: X.XX million (e.g., 123.45 million)
    - >= 1 billion: Use largest appropriate unit (billion, trillion, etc.)
    - ALWAYS round to 2 decimal places
    - NEVER use scientific notation

    Complete Data Display rules:
    - Show ALL machines/items if multiple exist
    - Never summarize with "etc." or similar
    - For tables: Include ALL rows in dictionary format

    Error Handling:
    - "No data available" for empty/null results
    - "Invalid query" for execution errors

    Return a valid JSON object with these fields:
    - "Explanation": A 1-2 line clear summary
    - "Table": Array of row objects OR "NA"
    - "Graph": Visualization suggestion OR "NA"
    """),
    ("human", """
    User question: {question}
    Final result: {final_result}

    Please interpret this result as a clear explanation for the user.
    """),
])

metadata_interpreter_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a result interpreter for metadata queries in a Text2SQL system.
    Your job is to transform metadata results into clear, human-readable explanations.

    For different metadata types:
    - system_info: Explain what the system does and its capabilities
    - tables_info: List available tables and their purposes
    - columns_info: Show column information in a structured way
    - help: Provide helpful guidance and examples

    Format guidelines:
    - Use clear, conversational language
    - For tables/columns: Present in organized format as an array of objects
    - Include practical examples where helpful
    - Keep explanations concise but informative

    IMPORTANT: Return data in the EXACT same format as regular queries:
    - "Explanation": Clear summary of the information
    - "Table": Array of objects (not "NA" - convert any dict data to array format)
    - "Graph": Always "NA" for metadata queries

    For table data, convert any dictionary/object data into an array of objects format.
    For example, if you have system info as a dict, convert it to:
    [
        {{"Property": "system_name", "Value": "Text2SQL System"}},
        {{"Property": "description", "Value": "AI-powered system..."}},
        etc.
    ]

    For tables info, format as:
    [
        {{"table_name": "EMS_Daily", "description": "Daily aggregated data"}},
        {{"table_name": "EMS_Hourly", "description": "Hourly data"}},
        etc.
    ]

    For columns info, format as:
    [
        {{"column_name": "MachineName", "data_type": "VARCHAR", "description": "Name of machine"}},
        {{"column_name": "PlantName", "data_type": "VARCHAR", "description": "Name of plant"}},
        etc.
    ]
    """),
    ("human", """
    User question: {question}
    Metadata result: {final_result}

    Please interpret this metadata result for the user and return in the standard format with Table as an array of objects.
    """),
])

# Define node functions for the LangGraph

def planner(state: Text2SQLState) -> dict:
    """Parse user question into structured intent"""
    print("\n🎯 PLANNER AGENT STARTED")
    print(f"📝 User Question: {state['question']}")

    llm = get_llm()
    chain = planner_prompt | llm

    metadata_context = get_metadata_context()

    result = chain.invoke({
        "question": state["question"],
        **metadata_context
    })

    print(f"🤖 Raw LLM Response: {result.content if hasattr(result, 'content') else result}")

    # Try to parse JSON from the response
    try:
        # First try to parse the entire content as JSON
        if hasattr(result, 'content'):
            content = result.content
        else:
            content = str(result)

        # Try to extract JSON from code blocks if present
        json_match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
        if json_match:
            content = json_match.group(1).strip()

        # Parse JSON
        structured_intent = json.loads(content)
        print(f"✅ Structured Intent Parsed: {json.dumps(structured_intent, indent=2)}")

    except (json.JSONDecodeError, AttributeError) as e:
        # Fallback: create a basic structured intent
        print(f"❌ JSON parsing failed: {e}. Creating fallback structured intent.")
        structured_intent = {
            "action": "list_data",
            "data_source": "EMS_Daily",
            "filters": {},
            "time_range": {"start_date": None, "end_date": None},
            "columns": ["time", "MachineName", "PlantName"]
        }
        print(f"🔄 Fallback Intent: {json.dumps(structured_intent, indent=2)}")

    print("🎯 PLANNER AGENT COMPLETED\n")
    return {"structured_intent": structured_intent}

def metadata_handler(state: Text2SQLState) -> dict:
    """Handle metadata queries about the system, tables, and columns"""
    print("\n📋 METADATA HANDLER AGENT STARTED")
    print(f"🔍 Metadata Query Type: {state['structured_intent'].get('metadata_type')}")

    metadata_type = state['structured_intent'].get('metadata_type')
    filters = state['structured_intent'].get('filters', {})

    if metadata_type == "system_info":
        result = {
            "system_name": "Text2SQL Multi-Agent System",
            "description": "An AI-powered system that converts natural language questions into SQL queries and provides intelligent data analysis",
            "capabilities": [
                "Convert natural language to SQL queries",
                "Execute queries on EMS database",
                "Perform data aggregation and analysis",
                "Generate visualizations and insights",
                "Handle complex multi-step data processing"
            ],
            "available_tables": ["EMS_Daily", "EMS_Hourly", "EMS_Shiftwise"],
            "supported_queries": [
                "Data retrieval and filtering",
                "Aggregations (count, sum, average)",
                "Time-based analysis",
                "Machine and plant comparisons",
                "Trend analysis"
            ]
        }

    elif metadata_type == "tables_info":
        tables_info = []
        for item in metadata.get('table_metadata', []):
            tables_info.append({
                "table_name": item['table_name'],
                "description": f"Contains {item['table_name'].replace('EMS_', '').lower()} level data for machine monitoring"
            })

        result = {
            "total_tables": len(tables_info),
            "tables": tables_info,
            "table_descriptions": {
                "EMS_Daily": "Daily aggregated machine data - use for general trends and daily analysis",
                "EMS_Hourly": "Hourly machine data - use for detailed hourly patterns",
                "EMS_Shiftwise": "Shift-based machine data - use for shift comparisons and analysis"
            }
        }

    elif metadata_type == "columns_info":
        table_name = filters.get('table_name')
        if table_name:
            # Filter columns for specific table
            columns_info = [
                {
                    "column_name": item['column_name'],
                    "data_type": item['column_datatype'],
                    "description": item['description']
                }
                for item in metadata.get('db_schema_metadata', [])
                if table_name.lower() in item.get('table_name', '').lower()
            ]
            result = {
                "table_name": table_name,
                "total_columns": len(columns_info),
                "columns": columns_info
            }
        else:
            # All columns info
            columns_info = [
                {
                    "column_name": item['column_name'],
                    "data_type": item['column_datatype'],
                    "description": item['description']
                }
                for item in metadata.get('db_schema_metadata', [])
            ]
            result = {
                "total_columns": len(columns_info),
                "columns": columns_info[:20],  # Limit for readability
                "note": f"Showing first 20 out of {len(columns_info)} columns. Ask about specific table for complete list."
            }

    elif metadata_type == "help":
        result = {
            "help_guide": "Text2SQL System - How to Use",
            "example_questions": [
                "How many machines are there for each division?",
                "Show me the trend of machine NCH1 values over time",
                "What is the average temperature for machines in Godavari plant?",
                "List all machines in UNIT153 division",
                "Compare production between different shifts",
                "What tables are available in the database?",
                "What columns are in EMS_Daily table?"
            ],
            "query_types": {
                "Data Queries": "Ask about machine data, production metrics, comparisons",
                "Aggregations": "Use 'how many', 'total', 'average', 'for each'",
                "Time Analysis": "Mention dates, trends, time periods",
                "System Info": "Ask 'what tables', 'what columns', 'help'"
            },
            "tips": [
                "Be specific about time ranges (e.g., 'last month', '2024-01-01 to 2024-01-31')",
                "Mention specific machine/plant names if known",
                "Use keywords like 'trend', 'compare', 'average', 'total' for better results"
            ]
        }
    else:
        result = {"error": "Unknown metadata query type"}

    print(f"📊 Metadata Result: {json.dumps(result, indent=2)[:500]}...")
    print("📋 METADATA HANDLER AGENT COMPLETED\n")

    return {"final_result": result}

def route_after_planner(state: Text2SQLState) -> Literal["metadata_handler", "generator"]:
    """Route to metadata handler or SQL generator based on intent"""
    if state["structured_intent"].get("action") == "metadata_query":
        return "metadata_handler"
    else:
        return "generator"

def generator(state: Text2SQLState) -> dict:
    """Generate SQL query from structured intent"""
    print("\n🔧 SQL GENERATOR AGENT STARTED")
    print(f"📊 Structured Intent: {json.dumps(state['structured_intent'], indent=2)}")

    llm = get_llm()
    chain = generator_prompt | llm

    metadata_context = get_metadata_context()

    result = chain.invoke({
        "question": state["question"],
        "structured_intent": state["structured_intent"],
        **metadata_context
    })

    print(f"🤖 Raw LLM Response: {result.content}")

    sql_query = extract_sql_from_message(result.content)
    if not sql_query:
        sql_query = result.content

    print(f"📋 Generated SQL Query:")
    print(f"{'='*50}")
    print(sql_query)
    print(f"{'='*50}")
    print("🔧 SQL GENERATOR AGENT COMPLETED\n")

    return {"sql_query": sql_query}

def validator(state: Text2SQLState) -> dict:
    """Validate and correct SQL query"""
    print("\n✅ SQL VALIDATOR AGENT STARTED")
    print(f"🔍 Input SQL Query:")
    print(f"{'='*50}")
    print(state["sql_query"])
    print(f"{'='*50}")

    llm = get_llm()
    chain = validator_prompt | llm

    metadata_context = get_metadata_context()

    result = chain.invoke({
        "question": state["question"],
        "structured_intent": state["structured_intent"],
        "sql_query": state["sql_query"],
        **metadata_context
    })

    print(f"🤖 Validator LLM Response: {result.content}")

    final_sql = extract_sql_from_message(result.content)
    if not final_sql:
        final_sql = result.content

    print(f"📋 Final Validated SQL Query:")
    print(f"{'='*50}")
    print(final_sql)
    print(f"{'='*50}")

    # Execute SQL
    print("🚀 Executing SQL Query...")
    query_result = execute_sql_query(final_sql, db_config)

    if query_result["success"]:
        if "rows" in query_result:
            print(f"✅ Query executed successfully. Rows returned: {len(query_result['rows'])}")
            print(f"📊 Columns: {query_result['columns']}")
            structured_data = format_results_structured(query_result)
            structured_data = convert_datetime_to_str(structured_data)
            print(f"📈 Sample data (first 3 rows): {structured_data[:3] if len(structured_data) > 3 else structured_data}")
        else:
            print(f"✅ Query executed successfully. {query_result.get('message', 'No rows returned')}")
            structured_data = query_result # This will be like {"success": True, "message": "..."}
    else:
        print(f"❌ Query execution failed: {query_result.get('error', 'Unknown error')}") # Ensure error key exists or provide default
        structured_data = query_result


    print("✅ SQL VALIDATOR AGENT COMPLETED\n")

    return {
        "sql_query": final_sql,
        "query_result": structured_data # This is now a list of dicts or the success/error message
    }

def decision_maker(state: Text2SQLState) -> dict:
    """Decide if post-processing is needed"""
    print("\n🤔 DECISION MAKER AGENT STARTED")
    print(f"❓ Analyzing if post-processing is needed for: {state['question']}")

    llm = get_llm()
    chain = decision_maker_prompt | llm | JsonOutputParser()

    # Query result might be a success message if no rows were returned (e.g. DDL/DML)
    # Ensure query_result sent to LLM is appropriate
    current_query_result = state["query_result"]
    if isinstance(current_query_result, dict) and "rows" not in current_query_result and "columns" not in current_query_result:
        # It's likely a message like {"success": True, "message": "Query executed..."}
        # For decision making, it's better to represent this as no data.
        display_query_result = "Query executed, no data returned."
    else:
        display_query_result = current_query_result


    result = chain.invoke({
        "question": state["question"],
        "sql_query": state["sql_query"],
        "query_result": display_query_result
    })

    print(f"🤖 Decision Maker Response: {result}")

    post_processing_required = result["decision"] == "POST_PROCESSING_REQUIRED"

    if post_processing_required:
        print("🔄 Decision: POST_PROCESSING_REQUIRED - Will generate Python code")
    else:
        print("✅ Decision: NO_POST_PROCESSING_REQUIRED - Will go directly to interpretation")

    print("🤔 DECISION MAKER AGENT COMPLETED\n")

    # If no post-processing, the final_result is the query_result
    return {
        "post_processing_required": post_processing_required,
        "final_result": state["query_result"] if not post_processing_required else None
    }

def route_to_next(state: Text2SQLState) -> Literal["generate_code", "interpret"]:
    """Route to code generation or directly to interpretation"""
    if state["post_processing_required"]:
        return "generate_code"
    else:
        return "interpret"

def code_generator_node(state: Text2SQLState) -> dict:
    """Generate Python code for post-processing"""
    print("\n🐍 PYTHON CODE GENERATOR AGENT STARTED")
    print(f"💻 Generating Python code for: {state['question']}")

    llm = get_llm(temperature=0.2)  # Slightly higher temperature for creative code generation
    chain = code_generator_prompt | llm

    result = chain.invoke({
        "question": state["question"],
        "sql_query": state["sql_query"],
        "query_result": state["query_result"] # This is list of dicts from validator
    })

    print(f"🤖 Code Generator LLM Response: {result.content[:500]}{'...' if len(result.content) > 500 else ''}")

    python_code = extract_code_from_message(result.content)
    if not python_code:
        python_code = result.content

    print(f"📝 Generated Python Code:")
    print(f"{'='*50}")
    print(python_code)
    print(f"{'='*50}")
    print("🐍 PYTHON CODE GENERATOR AGENT COMPLETED\n")

    return {"python_code": python_code}

def code_validator_node(state: Text2SQLState) -> dict:
    """Validate and correct Python code"""
    print("\n🔍 PYTHON CODE VALIDATOR AGENT STARTED")
    print(f"🧪 Validating Python code...")

    llm = get_llm()
    chain = code_validator_prompt | llm

    result = chain.invoke({
        "question": state["question"],
        "query_result": state["query_result"], # This is list of dicts
        "python_code": state["python_code"]
    })

    print(f"🤖 Code Validator LLM Response: {result.content[:500]}{'...' if len(result.content) > 500 else ''}")

    validated_code = extract_code_from_message(result.content)
    if not validated_code:
        validated_code = result.content

    print(f"✅ Validated Python Code:")
    print(f"{'='*50}")
    print(validated_code)
    print(f"{'='*50}")
    print("🔍 PYTHON CODE VALIDATOR AGENT COMPLETED\n")

    return {"python_code": validated_code}

def code_executor_node(state: Text2SQLState) -> dict:
    """Execute Python code and return results"""
    print("\n⚡ PYTHON CODE EXECUTOR AGENT STARTED")
    print(f"🏃‍♂️ Executing Python code...")

    llm = get_llm() # Using LLM to "execute" and interpret results
    chain = code_executor_prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({
            "question": state["question"],
            "python_code": state["python_code"],
            "query_result": state["query_result"] # This is list of dicts
        })
        print(f"🤖 Code Executor LLM Response: {json.dumps(result, indent=2) if isinstance(result, dict) else result}")

        if isinstance(result, dict) and result.get("execution_status") == "success":
            print("✅ Python code 'executed' successfully (simulated by LLM)")
            final_data = result.get("result_data", state["query_result"]) # Fallback to original if LLM doesn't give data
            # Potentially add visualization data to final_data if provided by LLM
            if "visualization_data" in result:
                # This part assumes your graph_agent might use this later or interpreter
                final_data = {"processed_data": final_data, "visualization_params": result["visualization_data"]}


        elif isinstance(result, dict) and result.get("execution_status") == "error":
            print(f"❌ Python code 'execution' failed (simulated by LLM): {result.get('result_data', 'Unknown error')}")
            final_data = {"error": result.get("result_data", "Python processing error")}
        else: # Fallback if LLM output is not as expected
            print(f"⚠️ Unexpected LLM output from code executor. Result: {result}")
            final_data = {"error": "Unexpected output from Python processing step.", "raw_llm_output": result}

    except Exception as e:
        print(f"❌ Exception during code executor LLM call: {e}")
        final_data = {"error": f"Exception in code executor: {str(e)}"}

    print("⚡ PYTHON CODE EXECUTOR AGENT COMPLETED\n")

    return {"final_result": final_data}


def interpreter_node(state: Text2SQLState) -> dict:
    """Generate final explanation for the user"""
    print("\n📖 INTERPRETER AGENT STARTED")
    print(f"📚 Generating final explanation for: {state['question']}")
    print(f"🔍 Final result to interpret: {state['final_result']}")


    llm = get_llm()
    chain = interpreter_prompt | llm | JsonOutputParser()

    try:
        # Ensure final_result is serializable for the LLM
        final_result_for_llm = state["final_result"]
        if isinstance(final_result_for_llm, pd.DataFrame): # Just in case
            final_result_for_llm = final_result_for_llm.to_dict(orient='records')


        result = chain.invoke({
            "question": state["question"],
            "final_result": final_result_for_llm
        })
        print(f"🤖 Interpreter LLM Response: {json.dumps(result, indent=2) if isinstance(result, dict) else result}")

        if not isinstance(result, dict) or not all(k in result for k in ["Explanation", "Table", "Graph"]):
             print(f"⚠️ Interpreter response is not in the expected format: {result}")
             # Fallback or error structure
             fallback_explanation = {
                 "Explanation": "Could not generate a full interpretation. Please check the raw data.",
                 "Table": final_result_for_llm if isinstance(final_result_for_llm, list) else "NA",
                 "Graph": "NA",
                 "Error": "Interpreter format error"
             }
             result = fallback_explanation


    except Exception as e:
        print(f"❌ Exception during interpreter LLM call: {e}")
        result = {
            "Explanation": f"Error during interpretation: {str(e)}",
            "Table": "NA",
            "Graph": "NA"
        }


    print(f"💬 Final Explanation (from interpreter): {result.get('Explanation', 'No explanation provided')}")
    print("📖 INTERPRETER AGENT COMPLETED\n")
    return {"explanation": result} # This 'explanation' dict now has Explanation, Table, Graph


def metadata_interpreter(state: Text2SQLState) -> dict:
    """Generate explanation for metadata queries"""
    print("\n📖 METADATA INTERPRETER AGENT STARTED")

    llm = get_llm()
    chain = metadata_interpreter_prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({
            "question": state["question"],
            "final_result": state["final_result"] # This is the dict from metadata_handler
        })
        print(f"📋 Metadata Interpretation LLM Response: {json.dumps(result, indent=2)}")

        if not isinstance(result, dict) or not all(k in result for k in ["Explanation", "Table", "Graph"]):
             print(f"⚠️ Metadata Interpreter response is not in the expected format: {result}")
             # Fallback or error structure
             fallback_explanation = {
                 "Explanation": "Could not generate a full metadata interpretation.",
                 "Table": state["final_result"] if isinstance(state["final_result"], list) else [{"data": state["final_result"]}], # Ensure table is list of dicts
                 "Graph": "NA", # Always NA for metadata
                 "Error": "Metadata Interpreter format error"
             }
             result = fallback_explanation

    except Exception as e:
        print(f"❌ Exception during metadata interpreter LLM call: {e}")
        result = {
            "Explanation": f"Error during metadata interpretation: {str(e)}",
            "Table": [{"error": str(e)}], # Ensure table is list of dicts
            "Graph": "NA"
        }


    print("📖 METADATA INTERPRETER AGENT COMPLETED\n")
    return {"explanation": result} # This 'explanation' dict has Explanation, Table, Graph


def graph_agent_node(state: Text2SQLState) -> dict:
    """Generate a plot using the GraphPlottingAgent, save it, and update state."""
    print("\n📊 GRAPH AGENT STARTED")

    # Agent for plotting (already initialized globally)
    # graph_agent = GraphPlottingAgent() # Not needed here if global

    question = state["question"]
    username = state.get("username", "guest") # Get username from state

    # Get data from the 'explanation' field set by the interpreter_node
    explanation_content = state.get("explanation", {})
    explanation_text = explanation_content.get("Explanation", "")
    table_data = explanation_content.get("Table", []) # This should be list of dicts or "NA"
    graph_suggestion = explanation_content.get("Graph", "") # This is a suggestion string

    # Ensure table_data is in a usable format for the agent (list of dicts)
    if table_data == "NA" or not isinstance(table_data, list):
        print("⚠️ No valid table data for graph agent, attempting to use raw final_result if available.")
        # Try to get table data from final_result if interpreter didn't provide it
        # This can happen if interpretation failed or if it's a direct path after code execution
        raw_final_result = state.get("final_result")
        if isinstance(raw_final_result, dict) and "processed_data" in raw_final_result and isinstance(raw_final_result["processed_data"], list):
            table_data = raw_final_result["processed_data"]
        elif isinstance(raw_final_result, list): # e.g. direct from SQL validator
            table_data = raw_final_result
        else:
            table_data = [] # Fallback to empty list if no suitable data found


    if not table_data and graph_suggestion != "NA":
        print(f"📉 Graph suggestion is '{graph_suggestion}' but no table data is available. Skipping graph generation.")
        # Keep existing explanation, but explicitly set graph details to indicate no graph was made
        updated_explanation = explanation_content.copy()
        updated_explanation["GraphImage"] = None
        updated_explanation["GraphDetails"] = {
            "success": False,
            "message": "Graph generation skipped: No data available for plotting.",
            "plot_type": "none"
        }
        return {
            "explanation": updated_explanation,
            # "final_result": updated_explanation # final_result should ideally hold raw data or processed data, not UI explanation
        }
    elif graph_suggestion == "NA":
        print("📉 No graph suggested by interpreter. Skipping graph generation.")
        updated_explanation = explanation_content.copy()
        updated_explanation["GraphImage"] = None
        updated_explanation["GraphDetails"] = {
            "success": False,
            "message": "No graph was suggested by the interpreter.",
            "plot_type": "none"
        }
        return {
            "explanation": updated_explanation,
        }


    print(f"🖼️ Calling GraphPlottingAgent with: Question='{question}', Suggestion='{graph_suggestion}', TableData (sample)='{str(table_data[:2]) if table_data else 'empty'}'")
    # Generate plot analysis using the global graph_agent instance
    graph_agent_output = graph_agent.analyze_and_plot(
        user_question=question,
        explanation=explanation_text, # The textual explanation
        table_data=table_data,       # The actual data for plotting
        graph_suggestion=graph_suggestion # The textual suggestion for what kind of graph
    )

    print(f"📈 Graph Agent Output: {graph_agent_output}")

    # Save the generated image and get its URL and metadata
    image_url, graph_meta = save_graph_image(graph_agent_output, username)

    # Merge the plot result (URL and metadata) into the existing explanation
    updated_explanation = explanation_content.copy() # Start with what interpreter gave
    updated_explanation["GraphImage"] = image_url if image_url else None # URL of the saved image
    updated_explanation["GraphDetails"] = graph_meta if graph_meta else { # Metadata about the graph
        "success": False,
        "message": graph_agent_output.get("error", "Graph generation failed or no image produced."),
        "plot_type": "error"
    }
    if image_url:
         updated_explanation["Graph"] = f"Visualized as {graph_meta.get('plot_type', 'chart')}. See image." # Update graph string

    print(f"🖼️ Updated explanation with graph info: {updated_explanation}")
    print("📊 GRAPH AGENT NODE COMPLETED\n")

    return {
        "explanation": updated_explanation,
        # "final_result": state.get("final_result") # Keep final_result as is (data, not UI explanation)
    }


# Build the enhanced graph
def build_graph():
    """Build the enhanced LangGraph workflow with metadata handling"""
    graph = StateGraph(Text2SQLState)

    # Add all nodes
    graph.add_node("planner", planner)
    graph.add_node("metadata_handler", metadata_handler)
    graph.add_node("metadata_interpreter", metadata_interpreter) # Interprets metadata output
    graph.add_node("generator", generator)
    graph.add_node("validator", validator)
    graph.add_node("decision_maker", decision_maker)
    graph.add_node("generate_code", code_generator_node)
    graph.add_node("validate_code", code_validator_node)
    graph.add_node("execute_code", code_executor_node)
    graph.add_node("interpret", interpreter_node) # Interprets SQL/Python output
    graph.add_node("graph_agent", graph_agent_node) # Generates graph based on interpreter's output


    # Define edges with conditional routing after planner
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "metadata_handler": "metadata_handler",
            "generator": "generator"
        }
    )

    # Metadata path
    graph.add_edge("metadata_handler", "metadata_interpreter")
    # Metadata interpreter's output does not go to graph_agent by default, ends.
    # If you want graphs for metadata (e.g. a chart of table counts), this would change.
    graph.add_edge("metadata_interpreter", END)


    # Regular SQL/Python path
    graph.add_edge("generator", "validator")
    graph.add_edge("validator", "decision_maker")
    graph.add_conditional_edges(
        "decision_maker",
        route_to_next,
        {
            "generate_code": "generate_code", # If post-processing needed
            "interpret": "interpret"          # If no post-processing, directly interpret SQL results
        }
    )
    graph.add_edge("generate_code", "validate_code")
    graph.add_edge("validate_code", "execute_code")
    graph.add_edge("execute_code", "interpret") # Interpret Python processed results

    # After interpretation (for non-metadata queries), generate graph
    graph.add_edge("interpret", "graph_agent")
    graph.add_edge("graph_agent", END) # End of the flow after graph generation

    # Set entry point
    graph.set_entry_point("planner")

    return graph.compile()

# Initialize the enhanced workflow
workflow = build_graph()

# API routes
@app.post('/query', response_model=QueryResponse) # Ensure response model matches actual output
async def handle_query(request: QueryRequest):
    try:
        user_question = request.user_question
        username = request.username

        print(f"\n🚀 NEW QUERY REQUEST")
        print(f"👤 Username: {username}")
        print(f"❓ Question: {user_question}")
        print(f"{'='*80}")

        if not user_question:
            raise HTTPException(status_code=400, detail="No question provided")

        # Initialize state
        initial_state: Text2SQLState = { # Explicitly type for clarity
            "question": user_question,
            "username": username,
            "structured_intent": None,
            "sql_query": None,
            "query_result": None, # result from SQL execution (list of dicts or error/message)
            "post_processing_required": None,
            "python_code": None,
            "final_result": None, # result after SQL or Python processing (data for interpretation)
            "explanation": None   # dict from interpreter/metadata_interpreter/graph_agent
        }

        # Execute the graph
        print("🔄 Starting LangGraph workflow execution...")
        final_state = workflow.invoke(initial_state)
        print("✅ LangGraph workflow execution completed!")
        print(f" 최종 상태 (Final State): {final_state}")


        # Extract final explanation details from the state
        # The 'explanation' field should now contain all necessary parts
        explanation_output = final_state.get("explanation", {})
        if not explanation_output: # Should not happen if graph ends properly
            print("⚠️ final_state['explanation'] is empty or None. This indicates an issue in the graph flow.")
            explanation_output = {
                "Explanation": "An unexpected error occurred in the processing pipeline.",
                "Table": "NA",
                "Graph": "NA",
                "GraphImage": None,
                "GraphDetails": {"success": False, "message": "Pipeline error."}
            }


        # Prepare response using QueryResponse model structure
        api_response = QueryResponse(
            Explanation=explanation_output.get("Explanation", "NA"),
            Table=explanation_output.get("Table", "NA"),
            Graph=explanation_output.get("Graph", "NA"), # Textual suggestion or updated info
            GraphImage=explanation_output.get("GraphImage"), # URL
            GraphDetails=explanation_output.get("GraphDetails") # Metadata
        )

        # Save to chat history
        print("💾 Saving to chat history...")
        insert_chat_history(
            username=username,
            user_question=user_question,
            explanation=api_response.Explanation,
            graph=api_response.Graph, # Textual graph info
            table=api_response.Table,
            db_config=db_config,
            graph_image_url=api_response.GraphImage # Pass the image URL
        )


        print(f"📤 FINAL RESPONSE TO CLIENT:")
        print(f"💬 Explanation: {api_response.Explanation}")
        print(f"📊 Table: {'Available' if api_response.Table != 'NA' else 'Not Available'}")
        print(f"📈 Graph Suggestion: {api_response.Graph}")
        print(f"🖼️ Graph Image URL: {api_response.GraphImage if api_response.GraphImage else 'Not Available'}")
        if api_response.GraphDetails:
            print(f"🔧 Graph Details: Success: {api_response.GraphDetails.get('success')}, Type: {api_response.GraphDetails.get('plot_type')}, Msg: {api_response.GraphDetails.get('message')}")
        print(f"{'='*80}\n")

        return api_response

    except Exception as e:
        print(f"❌ ERROR in handle_query: {e}")
        # Log the full traceback for detailed debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get('/chat_history', response_model=ChatHistoryResponse) # Added response_model
async def get_chat_history(username: str = "guest", limit: int = 10):
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        # Include GraphImage in select
        query = '''
            SELECT "User_Question", "Timestamp", "Table", "Explanation", "Graph", "GraphImage"
            FROM "Chat_History"
            WHERE "Username" = %s
            ORDER BY "Timestamp" DESC
            LIMIT %s
        '''

        cur.execute(query, (username, limit))
        columns = [desc[0] for desc in cur.description]
        history_rows = cur.fetchall()

        history = []
        for row in history_rows:
            history_item = dict(zip(columns, row))
            if isinstance(history_item.get("Timestamp"), datetime):
                history_item["Timestamp"] = history_item["Timestamp"].isoformat()
            if isinstance(history_item.get("Table"), str):
                try:
                    history_item["Table"] = json.loads(history_item["Table"])
                except json.JSONDecodeError:
                    pass # Keep as string if not valid JSON
            history.append(history_item)

        return ChatHistoryResponse(success=True, data=history) # Use response model

    except Exception as e:
        print(f"❌ Error retrieving chat history: {e}") # More context for logging
        # Log traceback
        import traceback
        traceback.print_exc()
        return ChatHistoryResponse(success=False, error=f"Error retrieving chat history: {str(e)}") # Use response model

    finally:
        if 'cur' in locals() and cur: # Check if cur exists and is not None
            cur.close()
        if 'conn' in locals() and conn: # Check if conn exists and is not None
            conn.close()


# Test endpoint for metadata queries
@app.get('/test_metadata', response_model=Dict[str, MetadataTestResponse]) # Added response_model
async def test_metadata():
    """Test endpoint to verify metadata queries work"""
    test_queries = [
        "What is this text2sql system?",
        "What tables are available?",
        "Help me understand what I can ask",
        # "How many machines are there for each division?" # This is not a metadata query
    ]

    results: Dict[str, MetadataTestResponse] = {} # Ensure type for results
    for query in test_queries:
        try:
            initial_state: Text2SQLState = {
                "question": query,
                "username": "test_user_metadata",
                "structured_intent": None, "sql_query": None, "query_result": None,
                "post_processing_required": None, "python_code": None,
                "final_result": None, "explanation": None
            }

            final_state = workflow.invoke(initial_state)
            explanation_output = final_state.get("explanation", {})
            results[query] = MetadataTestResponse( # Use response model
                success=True,
                explanation=explanation_output.get("Explanation", "NA"),
                table=explanation_output.get("Table", "NA"),
                graph=explanation_output.get("Graph", "NA") # Should be NA for metadata
            )
        except Exception as e:
            # Log traceback
            import traceback
            traceback.print_exc()
            results[query] = MetadataTestResponse(success=False, error=str(e)) # Use response model
            # Optionally, re-raise or handle more gracefully depending on desired behavior for a test endpoint
            # For now, we'll collect errors and return them.
            # raise HTTPException(status_code=500, detail=f"Error processing metadata query '{query}': {str(e)}")

    return results


@app.post("/enhanced-query/", response_model=EnhancedQueryResponse)
async def enhanced_query_endpoint(request: EnhancedQueryRequest):
    """
    This endpoint is primarily for testing the GraphPlottingAgent directly
    or for scenarios where you already have the data and want just the plot.
    It does NOT go through the full LangGraph workflow.
    """
    try:
        # Note: The global 'graph_agent' is used here.
        result_from_agent = graph_agent.analyze_and_plot(
            user_question=request.user_question,
            explanation=request.explanation,
            table_data=request.table_data,
            graph_suggestion=request.graph_suggestion
        )

        # The result_from_agent should contain 'plot_result' which itself might contain
        # 'image_data', 'figure', 'plot_type', 'success', 'message', etc.
        # We need to save this image if produced.

        image_url, graph_meta = save_graph_image(result_from_agent, request.username)

        # Construct the plot_result for the response
        # It should ideally contain the URL and other metadata.
        response_plot_result = {
            "success": result_from_agent.get("success", False),
            "message": result_from_agent.get("message", graph_meta.get("message") if graph_meta else "Analysis complete."),
            "plot_type": graph_meta.get("plot_type") if graph_meta else result_from_agent.get("plot_result", {}).get("plot_type", "unknown"),
            "image_url": image_url,
            "graph_details": graph_meta # Contains more detailed metadata including filename, timestamp etc.
        }


        return EnhancedQueryResponse(
            success=result_from_agent.get("success", False),
            explanation=request.explanation,
            table=request.table_data,
            graph_suggestion=request.graph_suggestion, # This was the input suggestion
            plot_result=response_plot_result, # This contains the outcome, including image_url
            error_message=result_from_agent.get("error", None)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return EnhancedQueryResponse(
            success=False,
            explanation=request.explanation,
            table=request.table_data,
            graph_suggestion=request.graph_suggestion,
            plot_result=None,
            error_message=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    # Ensure you have 'python-multipart' installed if you plan to upload files directly to FastAPI
    # pip install python-multipart
    # Ensure 'graph_agent.py' is in the same directory or accessible via PYTHONPATH
    # Ensure 'db_metadata.json' is present
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)