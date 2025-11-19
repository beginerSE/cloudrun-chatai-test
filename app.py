# -*- coding: utf-8 -*-
import os
import re
import json
import ast
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack
import queue
import threading

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import bcrypt
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
# LOCATION = os.getenv("LOCATION", "US")
# DEFAULT_DATASET = os.getenv("DEFAULT_DATASET", "your_dataset")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# GCP_SA_JSON = os.getenv("GCP_SA_JSON", "")
# DATABASE_URL = os.getenv("DATABASE_URL", "")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SESSION_SECRET', 'dev-secret-key-change-in-production')

# Configure Flask to use DateTimeEncoder for all JSON responses
from flask.json.provider import DefaultJSONProvider
class DateTimeJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

app.json = DateTimeJSONProvider(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'このページにアクセスするにはログインが必要です。'

# Progress tracking for chat tasks
chat_tasks = {}  # {task_id: {'status': 'running'|'completed'|'error', 'steps': [], 'result': {}, 'error': ''}}
task_lock = threading.Lock()

# Database connection
def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

# Safe redirect helper
def is_safe_url(target):
    """Check if the target URL is safe to redirect to"""
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc

def get_redirect_target():
    """Get a safe redirect target from request args or referrer"""
    for target in request.args.get('next'), request.referrer:
        if not target:
            continue
        if is_safe_url(target):
            return target
    return None

# User class
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute('SELECT id, username, email FROM users WHERE id = %s', (user_id,))
    user_data = cur.fetchone()
    cur.close()
    conn.close()
    
    if user_data:
        return User(user_data['id'], user_data['username'], user_data['email'])
    return None

def get_active_project_config(user_id):
    """Get configuration from active project or fall back to env variables"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute('''
        SELECT openai_api_key, bigquery_project_id, bigquery_dataset_id, service_account_json
        FROM projects
        WHERE user_id = %s AND is_active = true
        LIMIT 1
    ''', (user_id,))
    project = cur.fetchone()
    cur.close()
    conn.close()
    
    if project:
        return {
            'api_key': project['openai_api_key'] or OPENAI_API_KEY,
            'project_id': project['bigquery_project_id'] or PROJECT_ID,
            'dataset_id': project['bigquery_dataset_id'] or DEFAULT_DATASET,
            'service_account_json': project['service_account_json'] or GCP_SA_JSON
        }
    else:
        # Fall back to environment variables
        return {
            'api_key': OPENAI_API_KEY,
            'project_id': PROJECT_ID,
            'dataset_id': DEFAULT_DATASET,
            'service_account_json': GCP_SA_JSON
        }

async def extract_payload_text(res) -> str:
    """Extract text from MCP response"""
    for p in res.content:
        if getattr(p, "type", None) == "application/json" and getattr(p, "text", None):
            return p.text
    for p in res.content:
        if hasattr(p, "text") and p.text:
            return p.text
    return ""

async def call_tool_text(session: ClientSession, tool: str, args: Dict[str, Any]) -> str:
    """Call MCP tool and return text response"""
    res = await session.call_tool(tool, arguments=args)
    text = await extract_payload_text(res)
    if not text:
        raise RuntimeError(f"{tool} returned empty payload.")
    return text

async def list_tables(session: ClientSession, project_id: str) -> List[str]:
    """List all tables in BigQuery project"""
    text = await call_tool_text(session, "list-tables", {"project": project_id})
    try:
        js = json.loads(text)
        if isinstance(js, dict):
            arr = js.get("tables") or js.get("data") or []
            if isinstance(arr, list):
                return [str(t) if isinstance(t, str) else t.get("table", "") for t in arr]
    except:
        pass
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines

def parse_ddl_schema(ddl: str) -> List[Dict[str, str]]:
    """Parse BigQuery DDL to extract schema information"""
    import re
    schema = []
    
    # DDL から カラム定義部分を抽出（括弧内）
    match = re.search(r'\((.*)\)', ddl, re.DOTALL)
    if not match:
        return []
    
    columns_section = match.group(1)
    
    # 各行を処理
    for line in columns_section.split('\n'):
        line = line.strip()
        if not line or line.startswith('--'):
            continue
        
        # カラム名と型を抽出: "column_name TYPE" または "column_name TYPE,"
        # オプション句も考慮
        match = re.match(r'^([a-zA-Z0-9_]+)\s+([A-Z0-9_<>]+)', line)
        if match:
            col_name = match.group(1)
            col_type = match.group(2)
            schema.append({"name": col_name, "type": col_type})
    
    return schema

async def describe_table(session: ClientSession, project_id: str, dataset: str, table: str) -> List[Dict[str, str]]:
    """Get table schema"""
    variants = [
        {"project": project_id, "dataset": dataset, "table_name": table},
        {"project": project_id, "table_name": f"{dataset}.{table}"},
        {"dataset": dataset, "table_name": table},
        {"table_name": f"{dataset}.{table}"},
        {"table_name": f"{project_id}.{dataset}.{table}"},  # フルパス形式も試す
    ]
    
    for args in variants:
        try:
            text = await call_tool_text(session, "describe-table", args)
            print(f"DEBUG: describe-table response (first 200 chars): {text[:200]}")
            
            # まずJSONとして解析を試みる
            try:
                js = json.loads(text)
            except json.JSONDecodeError:
                # Pythonリテラル形式として評価
                print("DEBUG: describe-table JSON parse failed, trying Python literal eval...")
                try:
                    import datetime as dt_module
                    safe_dict = {
                        "datetime": dt_module,
                        "None": None,
                        "True": True,
                        "False": False,
                        "__builtins__": {}
                    }
                    exec(f"result = {text}", safe_dict)
                    js = safe_dict["result"]
                    print(f"DEBUG: Successfully evaluated describe-table Python literal")
                except Exception as eval_error:
                    print(f"DEBUG: describe-table Python literal eval failed: {eval_error}")
                    continue
            
            if isinstance(js, dict):
                # DDL形式の場合（'ddl'キーがある）
                ddl = js.get("ddl")
                if ddl and isinstance(ddl, str):
                    print(f"DEBUG: Parsing DDL: {ddl[:200]}...")
                    schema = parse_ddl_schema(ddl)
                    if schema:
                        return schema
                
                # 通常のスキーマ形式
                cols = js.get("columns") or js.get("schema") or js.get("fields")
                if isinstance(cols, list) and cols:
                    return [{"name": str(c.get("name", "")), "type": str(c.get("type", "UNKNOWN"))} for c in cols]
            elif isinstance(js, list):
                # リストが直接返される場合
                if js and isinstance(js[0], dict):
                    # DDL形式がリスト内にある場合
                    ddl = js[0].get("ddl")
                    if ddl and isinstance(ddl, str):
                        print(f"DEBUG: Parsing DDL from list: {ddl[:200]}...")
                        schema = parse_ddl_schema(ddl)
                        if schema:
                            return schema
                    # 通常のスキーマリスト
                    return [{"name": str(c.get("name", "")), "type": str(c.get("type", "UNKNOWN"))} for c in js if isinstance(c, dict)]
        except Exception as e:
            print(f"DEBUG: describe-table failed with args {args}: {e}")
            continue
    return []

async def execute_query(session: ClientSession, project_id: str, location: str, sql: str, timeout: int = 300) -> List[Dict[str, Any]]:
    """Execute BigQuery SQL and return results with timeout (default 5 minutes)"""
    lower = sql.lower()
    forbidden = ("insert ", "update ", "delete ", "merge ", "create ", "drop ", "alter ", "truncate ")
    if any(tok in lower for tok in forbidden):
        raise RuntimeError("Non read-only statements are blocked. Only SELECT queries are allowed.")
    
    variants = [
        {"project": project_id, "location": location, "query": sql},
        {"project": project_id, "query": sql},
    ]
    
    last_error = None
    for args in variants:
        try:
            # タイムアウト付きでクエリを実行
            text = await asyncio.wait_for(
                call_tool_text(session, "execute-query", args),
                timeout=timeout
            )
            print(f"DEBUG: Raw response from execute-query: {text[:200]}...")  # 最初の200文字を表示
            
            # エラーレスポンスのチェック
            if text.startswith("Error:") or "error" in text.lower()[:100]:
                # BigQueryエラーメッセージをパース
                error_msg = text
                if "Access Denied" in error_msg:
                    raise RuntimeError("BigQuery access denied. Please check: 1) Service account has 'BigQuery User' and 'BigQuery Data Viewer' roles, 2) Project ID is correct, 3) Dataset exists and service account has permission to access it.")
                elif "403" in error_msg:
                    raise RuntimeError("BigQuery permission error (403). Service account needs 'BigQuery Job User' role to execute queries.")
                elif "404" in error_msg:
                    raise RuntimeError("BigQuery resource not found (404). Check if the project ID, dataset, or table name is correct.")
                else:
                    raise RuntimeError(f"BigQuery error: {error_msg[:500]}")
            
            # まずJSONとして解析を試みる
            try:
                js = json.loads(text)
            except json.JSONDecodeError:
                # JSONとして解析できない場合、Pythonリテラル形式として評価
                print("DEBUG: JSON parse failed, trying Python literal eval...")
                # datetime等の複雑なオブジェクトを含む場合は、安全な評価のためにast.literal_evalは使えない
                # 代わりに、evalを制限的に使用（リスクを承知の上で）
                try:
                    # シンプルな方法: execで安全な環境で評価
                    import datetime as dt_module
                    safe_dict = {
                        "datetime": dt_module,
                        "None": None, 
                        "True": True, 
                        "False": False,
                        "__builtins__": {}  # セキュリティのため組み込み関数を制限
                    }
                    exec(f"result = {text}", safe_dict)
                    js = safe_dict["result"]
                    print(f"DEBUG: Successfully evaluated Python literal, type: {type(js)}")
                except Exception as eval_error:
                    print(f"DEBUG: Python literal eval also failed: {eval_error}")
                    raise
            
            # MCPサーバーは配列を直接返す場合がある
            if isinstance(js, list):
                # datetime オブジェクトを文字列に変換
                def serialize_row(row):
                    if isinstance(row, dict):
                        return {k: (v.isoformat() if hasattr(v, 'isoformat') else v) for k, v in row.items()}
                    return row
                return [serialize_row(row) for row in js]
            
            # または辞書形式で返す場合
            if isinstance(js, dict):
                rows = js.get("rows") or js.get("data") or js.get("result") or []
                if isinstance(rows, list):
                    return rows
                print(f"DEBUG: execute-query response keys: {list(js.keys())}, args: {args}")
        except asyncio.TimeoutError:
            error_msg = f"Query execution timed out after {timeout} seconds. The query may be too complex or processing too much data. Try: 1) Adding LIMIT clause, 2) Filtering data with WHERE, 3) Using aggregation."
            print(f"TIMEOUT: {error_msg}")
            raise RuntimeError(error_msg)
        except Exception as e:
            last_error = e
            print(f"DEBUG: execute-query failed with args {args}: {e}")
            continue
    
    error_msg = f"execute-query failed. Last error: {last_error}"
    print(f"ERROR: {error_msg}")
    raise RuntimeError(error_msg)

def execute_python_code(code: str, dataframe_dict: Dict[str, pd.DataFrame] = None, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute Python code in a restricted sandbox environment with access to BigQuery results as DataFrames
    
    Args:
        code: Python code to execute
        dataframe_dict: Dictionary of DataFrames to make available (e.g., {'df': query_result_dataframe})
        timeout: Execution timeout in seconds
        
    Returns:
        Dictionary with 'output' (stdout), 'result' (last expression value), 'plots' (base64 encoded images)
    """
    import io
    import sys
    import base64
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import sklearn
    from sklearn import (
        linear_model, tree, ensemble, svm, naive_bayes, 
        neighbors, cluster, decomposition, preprocessing,
        model_selection, metrics
    )
    import scipy
    from scipy import stats
    import statsmodels.api as sm
    
    # Capture stdout
    captured_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_output
    
    # Safe import function that only allows specific modules
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        allowed_modules = {
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'scipy', 'statsmodels',
            'matplotlib.pyplot', 'sklearn.ensemble', 'sklearn.model_selection', 
            'sklearn.metrics', 'sklearn.linear_model', 'sklearn.tree', 'sklearn.svm',
            'sklearn.naive_bayes', 'sklearn.neighbors', 'sklearn.cluster',
            'sklearn.decomposition', 'sklearn.preprocessing', 'scipy.stats'
        }
        # Allow submodule imports
        if name in allowed_modules or any(name.startswith(mod + '.') for mod in allowed_modules):
            return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f"Import of '{name}' is not allowed in sandbox")
    
    # Prepare restricted globals
    safe_globals = {
        '__builtins__': {
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'type': type,
            'isinstance': isinstance,
            'hasattr': hasattr,
            'getattr': getattr,
            'True': True,
            'False': False,
            'None': None,
            '__import__': safe_import,
        },
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns,
        'DataFrame': pd.DataFrame,
        'Series': pd.Series,
        # Scikit-learn modules
        'sklearn': sklearn,
        'linear_model': linear_model,
        'tree': tree,
        'ensemble': ensemble,
        'svm': svm,
        'naive_bayes': naive_bayes,
        'neighbors': neighbors,
        'cluster': cluster,
        'decomposition': decomposition,
        'preprocessing': preprocessing,
        'model_selection': model_selection,
        'metrics': metrics,
        # Scipy
        'scipy': scipy,
        'stats': stats,
        # Statsmodels
        'sm': sm,
        'statsmodels': sm,
    }
    
    # Add DataFrames to globals
    if dataframe_dict:
        safe_globals.update(dataframe_dict)
    
    result = None
    plots = []
    
    try:
        # Clear any existing plots
        plt.close('all')
        
        # Compile and execute code
        compiled_code = compile(code, '<string>', 'exec')
        exec(compiled_code, safe_globals)
        
        # Check if there's a result variable
        if 'result' in safe_globals:
            result = safe_globals['result']
        
        # Capture any matplotlib plots
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plots.append(img_base64)
            buf.close()
        
        plt.close('all')
        
    except Exception as e:
        sys.stdout = old_stdout
        raise RuntimeError(f"Python execution error: {str(e)}")
    finally:
        sys.stdout = old_stdout
    
    output_text = captured_output.getvalue()
    
    return {
        'output': output_text,
        'result': str(result) if result is not None else None,
        'plots': plots
    }

def build_openai_tools_schema() -> List[Dict[str, Any]]:
    """Build OpenAI function calling schema"""
    return [
        {
            "type": "function",
            "function": {
                "name": "list_tables",
                "description": "List all tables in BigQuery project",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project": {"type": "string", "description": "GCP project ID"}
                    },
                    "required": ["project"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "describe_table",
                "description": "Get the complete schema (column names, data types, and descriptions) of a BigQuery table. ALWAYS call this before writing SQL to ensure you use correct column names and understand data types. Pay special attention to: 1) Exact column name spelling (case-sensitive), 2) Data types (STRING, INTEGER, FLOAT64, TIMESTAMP, DATE, etc.), 3) Which columns are suitable for aggregation vs grouping.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project": {"type": "string", "description": "GCP project ID"},
                        "dataset": {"type": "string", "description": "Dataset name"},
                        "table": {"type": "string", "description": "Table name (exact spelling from list_tables)"}
                    },
                    "required": ["project", "dataset", "table"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "execute_query",
                "description": "Execute a read-only BigQuery SQL SELECT query. IMPORTANT: 1) Always use backticks for table names: `project.dataset.table` or `dataset.table`, 2) Use exact column names from describe_table (case-sensitive), 3) Use SAFE_CAST() for type conversions to avoid errors, 4) Include ORDER BY for visualization data, 5) Use LIMIT to prevent large result sets, 6) Handle NULLs with IFNULL() or WHERE column IS NOT NULL. Only SELECT queries are allowed - no INSERT, UPDATE, DELETE, CREATE, DROP.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project": {"type": "string", "description": "GCP project ID"},
                        "location": {"type": "string", "description": "BigQuery location (e.g., 'US', 'asia-northeast1')"},
                        "query": {"type": "string", "description": "Complete SQL SELECT query with proper BigQuery syntax. Must use backticks for table names, exact column names, and include ORDER BY for sorted results."}
                    },
                    "required": ["project", "location", "query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "suggest_chart",
                "description": "Suggest the best chart visualization based on query results. Analyze the data structure and column types to determine the most appropriate chart type (bar, line, pie, doughnut, scatter) and axis configuration.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chart_type": {
                            "type": "string", 
                            "enum": ["bar", "line", "pie", "doughnut", "scatter", "none"],
                            "description": "Chart type: 'bar' for categorical comparisons, 'line' for time series/trends, 'pie' or 'doughnut' for part-to-whole relationships (max 10 categories), 'scatter' for correlations, 'none' if data is not suitable for visualization"
                        },
                        "x_axis": {"type": "string", "description": "Column name for X-axis (labels). Use date/time columns for line charts, categories for bar charts"},
                        "y_axis": {"type": "string", "description": "Column name for Y-axis (values). Must be numeric"},
                        "title": {"type": "string", "description": "Chart title in Japanese describing what the chart shows"}
                    },
                    "required": ["chart_type", "x_axis", "y_axis", "title"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": "Execute Python code in a secure sandbox environment with access to BigQuery query results as Pandas DataFrames. Use this for advanced data analysis, machine learning, custom visualizations, statistical calculations, or data transformations that cannot be done with SQL alone. IMPORTANT: You CAN import and use external libraries in this sandbox. Available and importable libraries include: pandas, numpy, matplotlib, seaborn, sklearn (scikit-learn with all submodules like ensemble.RandomForestClassifier), scipy, and statsmodels. You can use import statements like 'from sklearn.ensemble import RandomForestClassifier' or 'import pandas as pd'. Pre-imported aliases available: pd (pandas), np (numpy), plt (matplotlib.pyplot), sns (seaborn), sklearn modules (linear_model, tree, ensemble, svm, naive_bayes, neighbors, cluster, decomposition, preprocessing, model_selection, metrics), scipy.stats (stats), statsmodels.api (sm). Query results are available as 'df' DataFrame. Use 'result' variable to return values. Use plt for creating custom charts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute. The most recent BigQuery query result is available as 'df' (a pandas DataFrame). Use print() for output, set 'result' variable for return values, and use plt.figure()/plt.plot()/etc for visualizations. Example: 'print(df.describe())' or 'result = df.groupby(\"category\")[\"value\"].mean()' or 'plt.figure(); plt.hist(df[\"column\"]); plt.title(\"Distribution\")'"
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    ]

async def run_agent(user_question: str, conversation_history: List[Dict[str, str]], 
                    api_key: str = None, project_id: str = None, dataset_id: str = None, 
                    service_account_json: str = None) -> Dict[str, Any]:
    """Main agent logic with MCP and OpenAI"""
    # Use provided parameters or fall back to global variables
    api_key = api_key or OPENAI_API_KEY
    project_id = project_id or PROJECT_ID
    dataset_id = dataset_id or DEFAULT_DATASET
    service_account_json = service_account_json or GCP_SA_JSON
    
    env = os.environ.copy()
    if service_account_json:
        env["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_json
    
    server_params = StdioServerParameters(
        command="mcp-server-bigquery",
        args=["--project", project_id, "--location", LOCATION],
        env=env
    )
    
    steps = []
    result_data = None
    result_charts = []  # 複数のグラフ設定を保持
    
    async with AsyncExitStack() as stack:
        stdio_transport = await stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await stack.enter_async_context(ClientSession(stdio, write))
        
        await session.initialize()
        
        client = OpenAI(api_key=api_key)
        
        messages = [
            {"role": "system", "content": f"""You are an expert BigQuery data analyst assistant with deep knowledge of SQL optimization and data analysis.

## ENVIRONMENT
- BigQuery Project: {project_id}
- Default Dataset: {dataset_id}

## STEP-BY-STEP REASONING FRAMEWORK
Follow this systematic approach for every user query:

### 1. UNDERSTAND THE QUESTION
- Identify the key metrics, dimensions, and filters requested
- Determine the time period if applicable
- Clarify any ambiguous terms by analyzing context

### 2. DISCOVER RELEVANT TABLES
- Use list_tables to see all available tables
- Identify tables that likely contain the requested data based on naming patterns
- Consider multiple tables if joins might be needed

### 3. EXAMINE TABLE SCHEMAS
- Use describe_table for each relevant table
- Carefully note:
  * Column names and their exact spelling
  * Data types (STRING, INTEGER, FLOAT, TIMESTAMP, DATE, BOOL, etc.)
  * Which columns can be used for aggregation vs grouping
  * Potential join keys between tables

### 4. CONSTRUCT SQL QUERY WITH BEST PRACTICES

**BigQuery-Specific Syntax:**
- Always use backticks for table references: `project.dataset.table` or `dataset.table`
- Use DATETIME/TIMESTAMP functions: CURRENT_DATETIME(), DATETIME_DIFF(), FORMAT_DATETIME()
- For dates: CURRENT_DATE(), DATE_DIFF(), FORMAT_DATE()
- Use SAFE_CAST() instead of CAST() to avoid errors
- Leverage IFNULL() or COALESCE() for NULL handling

**SQL Quality Guidelines:**
- Write explicit column names (avoid SELECT *)
- Use meaningful aliases with AS keyword
- Apply appropriate WHERE clauses for filtering
- Use proper GROUP BY with all non-aggregated columns
- Add ORDER BY for sorted results (essential for visualizations)
- Limit results appropriately (use LIMIT for large datasets)
- Handle NULLs explicitly to avoid incorrect aggregations

**Common Patterns:**
```sql
-- Aggregation with grouping
SELECT 
  column1,
  SUM(SAFE_CAST(column2 AS FLOAT64)) AS total_value,
  COUNT(*) AS count
FROM `{PROJECT_ID}.{DEFAULT_DATASET}.table_name`
WHERE column3 IS NOT NULL
GROUP BY column1
ORDER BY total_value DESC
LIMIT 100

-- Time-based analysis
SELECT 
  FORMAT_DATE('%Y-%m', date_column) AS month,
  AVG(numeric_column) AS avg_value
FROM `{PROJECT_ID}.{DEFAULT_DATASET}.table_name`
WHERE date_column >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)
GROUP BY month
ORDER BY month ASC

-- String matching
SELECT * 
FROM `{PROJECT_ID}.{DEFAULT_DATASET}.table_name`
WHERE LOWER(text_column) LIKE LOWER('%search_term%')
```

### 5. EXECUTE AND VALIDATE
- Run execute_query with the constructed SQL
- Check if results make sense (row count, value ranges)
- If error occurs, analyze the error message and fix:
  * Column name typos → recheck schema
  * Type mismatches → use SAFE_CAST()
  * NULL issues → add NULL handling
  * Syntax errors → verify BigQuery syntax

### 6. ADVANCED ANALYSIS WITH PYTHON (Optional)
For complex analysis beyond SQL capabilities, use execute_python with query results:

**When to Use Python:**
- Statistical analysis (correlation, regression, hypothesis testing)
- Custom data transformations or complex calculations
- Advanced visualizations (histograms, heatmaps, custom plots)
- Machine learning or predictive analytics
- Data cleaning or preprocessing

**Python Environment:**
- Query results available as 'df' (pandas DataFrame)
- Available libraries: pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns)
- Use print() for text output, set 'result' variable for return values
- Use plt for custom visualizations (automatically captured)

**Example Usage:**
```python
# Statistical analysis
print(df.describe())
print(df.corr())

# Custom visualization
plt.figure(figsize=(10, 6))
plt.hist(df['column_name'], bins=30)
plt.title('Distribution of Column')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Advanced calculation
result = df.groupby('category')['value'].agg(['mean', 'std', 'count'])
```

### 7. VISUALIZE INTELLIGENTLY
After getting query results, choose between suggest_chart (simple) or execute_python (advanced):

**suggest_chart - Use for standard visualizations:**
- **Line chart**: Time series data (x=date/time, y=metric), trends over time
- **Bar chart**: Categorical comparisons (x=category, y=value), rankings
- **Pie/Doughnut**: Part-to-whole with ≤10 categories, percentages, distributions
- **Scatter**: Correlation between two numeric variables (x=variable1, y=variable2)
- **None**: Single values, text-heavy data, or >50 data points without clear pattern

**execute_python - Use for custom visualizations:**
- Histograms, box plots, heatmaps, pair plots
- Multiple subplots or complex chart combinations
- Statistical plots (QQ plots, distribution fits)
- Custom styling or annotations

**Axis Selection (for suggest_chart):**
- x_axis: Use the dimension/category column (dates for trends, categories for comparisons)
- y_axis: Use the numeric metric column (always numeric values)
- Ensure columns match exactly with query result column names

### 8. PROVIDE INSIGHTS
- Explain what the data shows
- Highlight key findings (trends, outliers, patterns)
- Answer the user's original question clearly

## CRITICAL RULES
1. **Always check schemas before writing SQL** - Never assume column names
2. **Use exact column names from describe_table** - BigQuery is case-sensitive
3. **Test one table at a time** - Don't assume relationships exist
4. **Handle NULLs explicitly** - They affect aggregations
5. **Use appropriate data types** - SAFE_CAST when unsure
6. **Fully qualify table names** - Use `project.dataset.table` or `dataset.table`
7. **Always provide Japanese explanations** - Users expect Japanese responses

Think step-by-step and show your reasoning process."""}
        ]
        
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_question})
        
        tools = build_openai_tools_schema()
        
        for iteration in range(10):
            # gpt-5の場合は専用パラメーターを使用
            api_params = {
                "model": OPENAI_MODEL,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto"
            }
            
            # gpt-5の場合は reasoning_effort を追加
            if OPENAI_MODEL.startswith("gpt-5"):
                api_params["reasoning_effort"] = "high"  # 深い推論を実行
            
            response = client.chat.completions.create(**api_params)
            
            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump())
            
            if not assistant_message.tool_calls:
                return {
                    "answer": assistant_message.content,
                    "steps": steps,
                    "data": result_data,
                    "charts": result_charts  # 配列で返す
                }
            
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                steps.append(f"🔧 {func_name}({json.dumps(func_args, ensure_ascii=False)})")
                
                try:
                    if func_name == "list_tables":
                        result = await list_tables(session, func_args.get("project", PROJECT_ID))
                        func_result = json.dumps({"tables": result}, ensure_ascii=False)
                        
                    elif func_name == "describe_table":
                        result = await describe_table(
                            session,
                            func_args.get("project", PROJECT_ID),
                            func_args.get("dataset", DEFAULT_DATASET),
                            func_args["table"]
                        )
                        func_result = json.dumps({"columns": result}, ensure_ascii=False)
                        
                    elif func_name == "execute_query":
                        result = await execute_query(
                            session,
                            func_args.get("project", PROJECT_ID),
                            func_args.get("location", LOCATION),
                            func_args["query"]
                        )
                        result_data = result
                        steps.append(f"📊 Query returned {len(result)} rows")
                        
                        # If data is large, return summary to avoid token limit
                        if len(result) > 100:
                            summary = {
                                "row_count": len(result),
                                "columns": list(result[0].keys()) if result else [],
                                "sample_rows": result[:5],
                                "message": f"Large dataset with {len(result)} rows. Data stored for analysis."
                            }
                            func_result = json.dumps(summary, ensure_ascii=False, cls=DateTimeEncoder)
                        else:
                            func_result = json.dumps({"rows": result}, ensure_ascii=False, cls=DateTimeEncoder)
                    else:
                        func_result = json.dumps({"error": "Unknown function"})
                        
                except Exception as e:
                    error_msg = f"Error executing {func_name}: {str(e)}"
                    steps.append(f"❌ {error_msg}")
                    func_result = json.dumps({"error": error_msg}, ensure_ascii=False)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": func_result
                })
        
        return {
            "answer": "Maximum iterations reached.",
            "steps": steps,
            "data": result_data,
            "charts": result_charts  # 配列で返す
        }

async def run_agent_streaming(user_question: str, conversation_history: List[Dict[str, str]], msg_queue: queue.Queue, 
                              api_key: str = None, project_id: str = None, dataset_id: str = None, 
                              service_account_json: str = None) -> Dict[str, Any]:
    """Main agent logic with MCP and OpenAI - with streaming progress"""
    # Use provided parameters or fall back to global variables
    api_key = api_key or OPENAI_API_KEY
    project_id = project_id or PROJECT_ID
    dataset_id = dataset_id or DEFAULT_DATASET
    service_account_json = service_account_json or GCP_SA_JSON
    
    env = os.environ.copy()
    if service_account_json:
        env["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_json
    
    server_params = StdioServerParameters(
        command="mcp-server-bigquery",
        args=["--project", project_id, "--location", LOCATION],
        env=env
    )
    
    steps = []
    result_data = None
    result_charts = []  # 複数のグラフ設定を保持
    
    async with AsyncExitStack() as stack:
        stdio_transport = await stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await stack.enter_async_context(ClientSession(stdio, write))
        
        await session.initialize()
        
        client = OpenAI(api_key=api_key)
        
        messages = [
            {"role": "system", "content": f"""You are an expert BigQuery data analyst assistant with deep knowledge of SQL optimization and data analysis.

## ENVIRONMENT
- BigQuery Project: {project_id}
- Default Dataset: {dataset_id}

## STEP-BY-STEP REASONING FRAMEWORK
Follow this systematic approach for every user query:

### 1. UNDERSTAND THE QUESTION
- Identify the key metrics, dimensions, and filters requested
- Determine the time period if applicable
- Clarify any ambiguous terms by analyzing context

### 2. DISCOVER RELEVANT TABLES
- Use list_tables to see all available tables
- Identify tables that likely contain the requested data based on naming patterns
- Consider multiple tables if joins might be needed

### 3. EXAMINE TABLE SCHEMAS
- Use describe_table for each relevant table
- Carefully note:
  * Column names and their exact spelling
  * Data types (STRING, INTEGER, FLOAT, TIMESTAMP, DATE, BOOL, etc.)
  * Which columns can be used for aggregation vs grouping
  * Potential join keys between tables

### 4. CONSTRUCT SQL QUERY WITH BEST PRACTICES

**BigQuery-Specific Syntax:**
- Always use backticks for table references: `project.dataset.table` or `dataset.table`
- Use DATETIME/TIMESTAMP functions: CURRENT_DATETIME(), DATETIME_DIFF(), FORMAT_DATETIME()
- For dates: CURRENT_DATE(), DATE_DIFF(), FORMAT_DATE()
- Use SAFE_CAST() instead of CAST() to avoid errors
- Leverage IFNULL() or COALESCE() for NULL handling

**SQL Quality Guidelines:**
- Write explicit column names (avoid SELECT *)
- Use meaningful aliases with AS keyword
- Apply appropriate WHERE clauses for filtering
- Use proper GROUP BY with all non-aggregated columns
- Add ORDER BY for sorted results (essential for visualizations)
- Limit results appropriately (use LIMIT for large datasets)
- Handle NULLs explicitly to avoid incorrect aggregations

**Common Patterns:**
```sql
-- Aggregation with grouping
SELECT 
  column1,
  SUM(SAFE_CAST(column2 AS FLOAT64)) AS total_value,
  COUNT(*) AS count
FROM `{PROJECT_ID}.{DEFAULT_DATASET}.table_name`
WHERE column3 IS NOT NULL
GROUP BY column1
ORDER BY total_value DESC
LIMIT 100

-- Time-based analysis
SELECT 
  FORMAT_DATE('%Y-%m', date_column) AS month,
  AVG(numeric_column) AS avg_value
FROM `{PROJECT_ID}.{DEFAULT_DATASET}.table_name`
WHERE date_column >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)
GROUP BY month
ORDER BY month ASC

-- String matching
SELECT * 
FROM `{PROJECT_ID}.{DEFAULT_DATASET}.table_name`
WHERE LOWER(text_column) LIKE LOWER('%search_term%')
```

### 5. EXECUTE AND VALIDATE
- Run execute_query with the constructed SQL
- Check if results make sense (row count, value ranges)
- If error occurs, analyze the error message and fix:
  * Column name typos → recheck schema
  * Type mismatches → use SAFE_CAST()
  * NULL issues → add NULL handling
  * Syntax errors → verify BigQuery syntax

### 6. ADVANCED ANALYSIS WITH PYTHON (Optional)
For complex analysis beyond SQL capabilities, use execute_python with query results:

**When to Use Python:**
- Statistical analysis (correlation, regression, hypothesis testing)
- Custom data transformations or complex calculations
- Advanced visualizations (histograms, heatmaps, custom plots)
- Machine learning or predictive analytics
- Data cleaning or preprocessing

**Python Environment:**
- Query results available as 'df' (pandas DataFrame)
- Available libraries: pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns)
- Use print() for text output, set 'result' variable for return values
- Use plt for custom visualizations (automatically captured)

**Example Usage:**
```python
# Statistical analysis
print(df.describe())
print(df.corr())

# Custom visualization
plt.figure(figsize=(10, 6))
plt.hist(df['column_name'], bins=30)
plt.title('Distribution of Column')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Advanced calculation
result = df.groupby('category')['value'].agg(['mean', 'std', 'count'])
```

### 7. VISUALIZE INTELLIGENTLY
After getting query results, choose between suggest_chart (simple) or execute_python (advanced):

**suggest_chart - Use for standard visualizations:**
- **Line chart**: Time series data (x=date/time, y=metric), trends over time
- **Bar chart**: Categorical comparisons (x=category, y=value), rankings
- **Pie/Doughnut**: Part-to-whole with ≤10 categories, percentages, distributions
- **Scatter**: Correlation between two numeric variables (x=variable1, y=variable2)
- **None**: Single values, text-heavy data, or >50 data points without clear pattern

**execute_python - Use for custom visualizations:**
- Histograms, box plots, heatmaps, pair plots
- Multiple subplots or complex chart combinations
- Statistical plots (QQ plots, distribution fits)
- Custom styling or annotations

**Axis Selection (for suggest_chart):**
- x_axis: Use the dimension/category column (dates for trends, categories for comparisons)
- y_axis: Use the numeric metric column (always numeric values)
- Ensure columns match exactly with query result column names

### 8. PROVIDE INSIGHTS
- Explain what the data shows
- Highlight key findings (trends, outliers, patterns)
- Answer the user's original question clearly

## CRITICAL RULES
1. **Always check schemas before writing SQL** - Never assume column names
2. **Use exact column names from describe_table** - BigQuery is case-sensitive
3. **Test one table at a time** - Don't assume relationships exist
4. **Handle NULLs explicitly** - They affect aggregations
5. **Use appropriate data types** - SAFE_CAST when unsure
6. **Fully qualify table names** - Use `project.dataset.table` or `dataset.table`
7. **Always provide Japanese explanations** - Users expect Japanese responses

Think step-by-step and show your reasoning process."""}
        ]
        
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_question})
        
        tools = build_openai_tools_schema()
        
        for iteration in range(10):
            msg_queue.put({"type": "thinking", "message": "次のアクションを考えています..."})
            
            # gpt-5の場合は専用パラメーターを使用
            api_params = {
                "model": OPENAI_MODEL,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "stream": True
            }
            
            # gpt-5の場合は reasoning_effort を追加
            if OPENAI_MODEL.startswith("gpt-5"):
                api_params["reasoning_effort"] = "high"  # 深い推論を実行
            
            # GPT-5でストリーミングエラーが発生する場合は非ストリーミングにフォールバック
            try:
                response = client.chat.completions.create(**api_params)
            except Exception as e:
                if "stream" in str(e).lower() and OPENAI_MODEL.startswith("gpt-5"):
                    # ストリーミングなしで再試行
                    msg_queue.put({"type": "info", "message": "GPT-5: 非ストリーミングモードで実行中..."})
                    api_params["stream"] = False
                    response_obj = client.chat.completions.create(**api_params)
                    
                    # 非ストリーミングレスポンスをストリーミング形式に変換
                    assistant_message = response_obj.choices[0].message
                    if assistant_message.content:
                        for char in assistant_message.content:
                            msg_queue.put({"type": "assistant_text", "text": char})
                    
                    # メッセージを構築してツール呼び出しを処理
                    messages.append(assistant_message.model_dump())
                    
                    if not assistant_message.tool_calls:
                        return {
                            "answer": assistant_message.content,
                            "steps": steps,
                            "data": result_data,
                            "charts": result_charts  # 配列で返す
                        }
                    
                    # ツール呼び出し処理（既存のロジックと同じ）
                    for tool_call in assistant_message.tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        
                        steps.append(f"🔧 {func_name}({json.dumps(func_args, ensure_ascii=False)})")
                        msg_queue.put({"type": "tool_call", "name": func_name, "args": func_args})
                        
                        try:
                            if func_name == "list_tables":
                                msg_queue.put({"type": "tool_start", "tool": "list_tables", "message": "テーブル一覧を取得中..."})
                                result = await list_tables(session, func_args.get("project", PROJECT_ID))
                                func_result = json.dumps({"tables": result}, ensure_ascii=False)
                                msg_queue.put({"type": "tool_done", "tool": "list_tables", "message": f"{len(result)}個のテーブルを発見しました"})
                                
                            elif func_name == "describe_table":
                                # GPT-5とGPT-4で異なるパラメータ名を使う可能性があるため両方をチェック
                                table_name = func_args.get("table") or func_args.get("table_name")
                                if not table_name:
                                    raise ValueError(f"Missing required parameter: 'table'. Received: {func_args}")
                                msg_queue.put({"type": "tool_start", "tool": "describe_table", "message": f"テーブル '{table_name}' のスキーマを取得中..."})
                                result = await describe_table(
                                    session,
                                    func_args.get("project", PROJECT_ID),
                                    func_args.get("dataset", DEFAULT_DATASET),
                                    table_name
                                )
                                func_result = json.dumps({"columns": result}, ensure_ascii=False)
                                msg_queue.put({"type": "tool_done", "tool": "describe_table", "message": f"{len(result)}個のカラムを発見しました"})
                                
                            elif func_name == "execute_query":
                                query = func_args.get("query") or func_args.get("sql")
                                if not query:
                                    raise ValueError(f"Missing required parameter: 'query'. Received: {func_args}")
                                msg_queue.put({"type": "tool_start", "tool": "execute_query", "message": f"クエリを実行中...", "query": query})
                                result = await execute_query(
                                    session,
                                    func_args.get("project", PROJECT_ID),
                                    func_args.get("location", LOCATION),
                                    query
                                )
                                result_data = result
                                msg = f"📊 {len(result)}行のデータを取得しました"
                                steps.append(msg)
                                msg_queue.put({"type": "tool_done", "tool": "execute_query", "message": msg})
                                
                                # If data is large, return summary to avoid token limit
                                if len(result) > 100:
                                    summary = {
                                        "row_count": len(result),
                                        "columns": list(result[0].keys()) if result else [],
                                        "sample_rows": result[:5],
                                        "message": f"Large dataset with {len(result)} rows. Data stored for analysis."
                                    }
                                    func_result = json.dumps(summary, ensure_ascii=False, cls=DateTimeEncoder)
                                else:
                                    func_result = json.dumps({"result": result}, ensure_ascii=False, cls=DateTimeEncoder)
                                
                            elif func_name == "suggest_chart":
                                msg_queue.put({"type": "tool_start", "tool": "suggest_chart", "message": "グラフ設定を提案中..."})
                                chart_config = {
                                    "chart_type": func_args.get("chart_type", "bar"),
                                    "x_axis": func_args.get("x_axis"),
                                    "y_axis": func_args.get("y_axis"),
                                    "title": func_args.get("title", "")
                                }
                                result_charts.append(chart_config)  # 配列に追加
                                func_result = json.dumps({"chart": chart_config}, ensure_ascii=False)
                                chart_type_ja = {
                                    "bar": "棒グラフ",
                                    "line": "折れ線グラフ",
                                    "pie": "円グラフ",
                                    "doughnut": "ドーナツグラフ",
                                    "scatter": "散布図",
                                    "none": "グラフなし"
                                }.get(chart_config["chart_type"], chart_config["chart_type"])
                                msg = f"📈 {chart_type_ja}を提案しました"
                                steps.append(msg)
                                msg_queue.put({"type": "tool_done", "tool": "suggest_chart", "message": msg})
                                
                            else:
                                func_result = json.dumps({"error": "Unknown function"})
                                
                        except Exception as e:
                            error_msg = str(e)
                            error_details = {
                                "error": error_msg,
                                "function": func_name,
                                "arguments": func_args
                            }
                            
                            # エラーの種類に応じた修正提案を追加
                            if func_name == "execute_query":
                                if "column" in error_msg.lower() or "field" in error_msg.lower():
                                    error_details["hint"] = "Column name error detected. Use describe_table to get exact column names (case-sensitive)."
                                elif "table" in error_msg.lower():
                                    error_details["hint"] = "Table reference error. Use backticks: `project.dataset.table` or `dataset.table`"
                                elif "type" in error_msg.lower() or "cast" in error_msg.lower():
                                    error_details["hint"] = "Data type error. Use SAFE_CAST(column AS FLOAT64) for type conversion."
                                elif "syntax" in error_msg.lower():
                                    error_details["hint"] = "SQL syntax error. Check BigQuery syntax: use backticks for tables, proper GROUP BY, etc."
                                else:
                                    error_details["hint"] = "Query execution failed. Review the error message and check: 1) Column names (case-sensitive), 2) Table references (use backticks), 3) Data types, 4) BigQuery syntax."
                            
                            steps.append(f"❌ {func_name} エラー: {error_msg}")
                            msg_queue.put({"type": "error", "message": f"{func_name} でエラーが発生しました: {error_msg}"})
                            func_result = json.dumps(error_details, ensure_ascii=False)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": func_result
                        })
                    continue  # 次のイテレーションへ
                else:
                    raise  # その他のエラーは再スロー
            
            # ストリーミングレスポンスを収集
            full_content = ""
            reasoning_content = ""
            tool_calls_data = []
            current_tool_call = None
            
            for chunk in response:
                delta = chunk.choices[0].delta
                
                # 推論過程のストリーミング（GPT-5など）
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                    msg_queue.put({"type": "reasoning", "text": delta.reasoning_content})
                
                # コンテンツのストリーミング
                if delta.content:
                    full_content += delta.content
                    msg_queue.put({"type": "assistant_text", "text": delta.content})
                
                # ツール呼び出しの処理
                if delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        if tc_chunk.index is not None:
                            while len(tool_calls_data) <= tc_chunk.index:
                                tool_calls_data.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                            current_tool_call = tool_calls_data[tc_chunk.index]
                            
                            if tc_chunk.id:
                                current_tool_call["id"] = tc_chunk.id
                            if tc_chunk.function:
                                if tc_chunk.function.name:
                                    current_tool_call["function"]["name"] = tc_chunk.function.name
                                if tc_chunk.function.arguments:
                                    current_tool_call["function"]["arguments"] += tc_chunk.function.arguments
            
            # メッセージを構築
            assistant_message_dict = {"role": "assistant", "content": full_content or None}
            
            if tool_calls_data:
                # ツール呼び出しをOpenAI形式に変換
                from openai.types.chat import ChatCompletionMessageToolCall
                from openai.types.chat.chat_completion_message_tool_call import Function
                
                tool_calls_objs = []
                for tc in tool_calls_data:
                    tool_calls_objs.append(
                        ChatCompletionMessageToolCall(
                            id=tc["id"],
                            type="function",
                            function=Function(
                                name=tc["function"]["name"],
                                arguments=tc["function"]["arguments"]
                            )
                        )
                    )
                assistant_message_dict["tool_calls"] = [tc.model_dump() for tc in tool_calls_objs]
                
                messages.append(assistant_message_dict)
                
                # ツール実行
                for tool_call in tool_calls_objs:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    step_msg = f"🔧 {func_name}({json.dumps(func_args, ensure_ascii=False)})"
                    steps.append(step_msg)
                    
                    try:
                        if func_name == "list_tables":
                            msg_queue.put({"type": "tool_start", "tool": "list_tables", "message": "テーブル一覧を取得中..."})
                            result = await list_tables(session, func_args.get("project", PROJECT_ID))
                            func_result = json.dumps({"tables": result}, ensure_ascii=False)
                            msg_queue.put({"type": "tool_done", "tool": "list_tables", "message": f"{len(result)}個のテーブルを発見しました"})
                            
                        elif func_name == "describe_table":
                            # GPT-5とGPT-4で異なるパラメータ名を使う可能性があるため両方をチェック
                            table_name = func_args.get("table") or func_args.get("table_name")
                            if not table_name:
                                raise ValueError(f"Missing required parameter: 'table'. Received: {func_args}")
                            msg_queue.put({"type": "tool_start", "tool": "describe_table", "message": f"テーブル '{table_name}' のスキーマを取得中..."})
                            result = await describe_table(
                                session,
                                func_args.get("project", PROJECT_ID),
                                func_args.get("dataset", DEFAULT_DATASET),
                                table_name
                            )
                            func_result = json.dumps({"columns": result}, ensure_ascii=False)
                            msg_queue.put({"type": "tool_done", "tool": "describe_table", "message": f"{len(result)}個のカラムを発見しました"})
                            
                        elif func_name == "execute_query":
                            query = func_args.get("query") or func_args.get("sql")
                            if not query:
                                raise ValueError(f"Missing required parameter: 'query'. Received: {func_args}")
                            msg_queue.put({"type": "tool_start", "tool": "execute_query", "message": f"クエリを実行中...", "query": query})
                            result = await execute_query(
                                session,
                                func_args.get("project", PROJECT_ID),
                                func_args.get("location", LOCATION),
                                query
                            )
                            result_data = result
                            msg = f"📊 {len(result)}行のデータを取得しました"
                            steps.append(msg)
                            msg_queue.put({"type": "tool_done", "tool": "execute_query", "message": msg})
                            
                            # If data is large, return summary to avoid token limit
                            if len(result) > 100:
                                summary = {
                                    "row_count": len(result),
                                    "columns": list(result[0].keys()) if result else [],
                                    "sample_rows": result[:5],
                                    "message": f"Large dataset with {len(result)} rows. Data stored for analysis."
                                }
                                func_result = json.dumps(summary, ensure_ascii=False, cls=DateTimeEncoder)
                            else:
                                func_result = json.dumps({"rows": result}, ensure_ascii=False, cls=DateTimeEncoder)
                            
                        elif func_name == "suggest_chart":
                            msg_queue.put({"type": "tool_start", "tool": "suggest_chart", "message": "グラフ設定を提案中..."})
                            chart_config = {
                                "chart_type": func_args.get("chart_type", "bar"),
                                "x_axis": func_args.get("x_axis"),
                                "y_axis": func_args.get("y_axis"),
                                "title": func_args.get("title", "")
                            }
                            result_charts.append(chart_config)  # 配列に追加
                            func_result = json.dumps({"chart": chart_config}, ensure_ascii=False)
                            chart_type_ja = {
                                "bar": "棒グラフ",
                                "line": "折れ線グラフ",
                                "pie": "円グラフ",
                                "doughnut": "ドーナツグラフ",
                                "scatter": "散布図",
                                "none": "グラフなし"
                            }.get(chart_config["chart_type"], chart_config["chart_type"])
                            msg = f"📈 {chart_type_ja}を提案しました"
                            steps.append(msg)
                            msg_queue.put({"type": "tool_done", "tool": "suggest_chart", "message": msg})
                            
                        else:
                            func_result = json.dumps({"error": "Unknown function"})
                            
                    except Exception as e:
                        error_msg = str(e)
                        error_details = {
                            "error": error_msg,
                            "function": func_name,
                            "arguments": func_args
                        }
                        
                        # エラーの種類に応じた修正提案を追加
                        if func_name == "execute_query":
                            if "column" in error_msg.lower() or "field" in error_msg.lower():
                                error_details["hint"] = "Column name error detected. Use describe_table to get exact column names (case-sensitive)."
                            elif "table" in error_msg.lower():
                                error_details["hint"] = "Table reference error. Use backticks: `project.dataset.table` or `dataset.table`"
                            elif "type" in error_msg.lower() or "cast" in error_msg.lower():
                                error_details["hint"] = "Data type error. Use SAFE_CAST(column AS FLOAT64) for type conversion."
                            elif "syntax" in error_msg.lower():
                                error_details["hint"] = "SQL syntax error. Check BigQuery syntax: use backticks for tables, proper GROUP BY, etc."
                            else:
                                error_details["hint"] = "Query execution failed. Review the error message and check: 1) Column names (case-sensitive), 2) Table references (use backticks), 3) Data types, 4) BigQuery syntax."
                        
                        steps.append(f"❌ {func_name} エラー: {error_msg}")
                        msg_queue.put({"type": "error", "message": f"{func_name} でエラーが発生しました: {error_msg}"})
                        func_result = json.dumps(error_details, ensure_ascii=False)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": func_result
                    })
            else:
                # ツール呼び出しなし = 最終回答
                messages.append(assistant_message_dict)
                return {
                    "answer": full_content,
                    "steps": steps,
                    "data": result_data,
                    "charts": result_charts  # 配列で返す
                }
        
        return {
            "answer": "Maximum iterations reached.",
            "steps": steps,
            "data": result_data,
            "charts": result_charts  # 配列で返す
        }

async def run_agent_with_steps(task_id: str, user_question: str, conversation_history: List[Dict[str, str]], 
                              api_key: str = None, project_id: str = None, dataset_id: str = None, 
                              service_account_json: str = None) -> Dict[str, Any]:
    """Main agent logic with progress tracking via task_id"""
    
    def add_step(message: str):
        """Helper to add step to task"""
        with task_lock:
            if task_id in chat_tasks:
                chat_tasks[task_id]['steps'].append(message)
    
    api_key = api_key or OPENAI_API_KEY
    project_id = project_id or PROJECT_ID
    dataset_id = dataset_id or DEFAULT_DATASET
    service_account_json = service_account_json or GCP_SA_JSON
    
    env = os.environ.copy()
    if service_account_json:
        env["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_json
    
    server_params = StdioServerParameters(
        command="mcp-server-bigquery",
        args=["--project", project_id, "--location", LOCATION],
        env=env
    )
    
    steps = []
    result_data = None
    result_charts = []
    latest_dataframe = None  # Store latest query result as DataFrame for Python execution
    python_results = []  # Store Python execution results
    
    async with AsyncExitStack() as stack:
        stdio_transport = await stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await stack.enter_async_context(ClientSession(stdio, write))
        
        await session.initialize()
        
        client = OpenAI(api_key=api_key)
        
        messages = [
            {"role": "system", "content": f"""You are an expert BigQuery data analyst assistant with deep knowledge of SQL optimization and data analysis.

## ENVIRONMENT
- BigQuery Project: {project_id}
- Default Dataset: {dataset_id}

## STEP-BY-STEP REASONING FRAMEWORK
Follow this systematic approach for every user query:

### 1. UNDERSTAND THE QUESTION
- Identify the key metrics, dimensions, and filters requested
- Determine the time period if applicable
- Clarify any ambiguous terms by analyzing context

### 2. DISCOVER RELEVANT TABLES
- Use list_tables to see all available tables
- Identify tables that likely contain the requested data based on naming patterns
- Consider multiple tables if joins might be needed

### 3. EXAMINE TABLE SCHEMAS
- Use describe_table for each relevant table
- Carefully note:
  * Column names and their exact spelling
  * Data types (STRING, INTEGER, FLOAT, TIMESTAMP, DATE, BOOL, etc.)
  * Which columns can be used for aggregation vs grouping
  * Potential join keys between tables

### 4. CONSTRUCT SQL QUERY WITH BEST PRACTICES

**BigQuery-Specific Syntax:**
- Always use backticks for table references: `project.dataset.table` or `dataset.table`
- Use DATETIME/TIMESTAMP functions: CURRENT_DATETIME(), DATETIME_DIFF(), FORMAT_DATETIME()
- For dates: CURRENT_DATE(), DATE_DIFF(), FORMAT_DATE()
- Use SAFE_CAST() instead of CAST() to avoid errors
- Leverage IFNULL() or COALESCE() for NULL handling

**SQL Quality Guidelines:**
- Write explicit column names (avoid SELECT *)
- Use meaningful aliases with AS keyword
- Apply appropriate WHERE clauses for filtering
- Use proper GROUP BY with all non-aggregated columns
- Add ORDER BY for sorted results (essential for visualizations)
- Limit results appropriately (use LIMIT for large datasets)
- Handle NULLs explicitly to avoid incorrect aggregations

**Common Patterns:**
```sql
-- Aggregation with grouping
SELECT 
  column1,
  SUM(SAFE_CAST(column2 AS FLOAT64)) AS total_value,
  COUNT(*) AS count
FROM `{PROJECT_ID}.{DEFAULT_DATASET}.table_name`
WHERE column3 IS NOT NULL
GROUP BY column1
ORDER BY total_value DESC
LIMIT 100

-- Time-based analysis
SELECT 
  FORMAT_DATE('%Y-%m', date_column) AS month,
  AVG(numeric_column) AS avg_value
FROM `{PROJECT_ID}.{DEFAULT_DATASET}.table_name`
WHERE date_column >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)
GROUP BY month
ORDER BY month ASC

-- String matching
SELECT * 
FROM `{PROJECT_ID}.{DEFAULT_DATASET}.table_name`
WHERE LOWER(text_column) LIKE LOWER('%search_term%')
```

### 5. EXECUTE AND VALIDATE
- Run execute_query with the constructed SQL
- Check if results make sense (row count, value ranges)
- If error occurs, analyze the error message and fix:
  * Column name typos → recheck schema
  * Type mismatches → use SAFE_CAST()
  * NULL issues → add NULL handling
  * Syntax errors → verify BigQuery syntax

### 6. ADVANCED ANALYSIS WITH PYTHON (Optional)
For complex analysis beyond SQL capabilities, use execute_python with query results:

**When to Use Python:**
- Statistical analysis (correlation, regression, hypothesis testing)
- Custom data transformations or complex calculations
- Advanced visualizations (histograms, heatmaps, custom plots)
- Machine learning or predictive analytics
- Data cleaning or preprocessing

**Python Environment:**
- Query results available as 'df' (pandas DataFrame)
- Available libraries: pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns)
- Use print() for text output, set 'result' variable for return values
- Use plt for custom visualizations (automatically captured)

**Example Usage:**
```python
# Statistical analysis
print(df.describe())
print(df.corr())

# Custom visualization
plt.figure(figsize=(10, 6))
plt.hist(df['column_name'], bins=30)
plt.title('Distribution of Column')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Advanced calculation
result = df.groupby('category')['value'].agg(['mean', 'std', 'count'])
```

### 7. VISUALIZE INTELLIGENTLY
After getting query results, choose between suggest_chart (simple) or execute_python (advanced):

**suggest_chart - Use for standard visualizations:**
- **Line chart**: Time series data (x=date/time, y=metric), trends over time
- **Bar chart**: Categorical comparisons (x=category, y=value), rankings
- **Pie/Doughnut**: Part-to-whole with ≤10 categories, percentages, distributions
- **Scatter**: Correlation between two numeric variables (x=variable1, y=variable2)
- **None**: Single values, text-heavy data, or >50 data points without clear pattern

**execute_python - Use for custom visualizations:**
- Histograms, box plots, heatmaps, pair plots
- Multiple subplots or complex chart combinations
- Statistical plots (QQ plots, distribution fits)
- Custom styling or annotations

**Axis Selection (for suggest_chart):**
- x_axis: Use the dimension/category column (dates for trends, categories for comparisons)
- y_axis: Use the numeric metric column (always numeric values)
- Ensure columns match exactly with query result column names

### 8. PROVIDE INSIGHTS
- Explain what the data shows
- Highlight key findings (trends, outliers, patterns)
- Answer the user's original question clearly

## CRITICAL RULES
1. **Always check schemas before writing SQL** - Never assume column names
2. **Use exact column names from describe_table** - BigQuery is case-sensitive
3. **Test one table at a time** - Don't assume relationships exist
4. **Handle NULLs explicitly** - They affect aggregations
5. **Use appropriate data types** - SAFE_CAST when unsure
6. **Fully qualify table names** - Use `project.dataset.table` or `dataset.table`
7. **Always provide Japanese explanations** - Users expect Japanese responses

Think step-by-step and show your reasoning process."""}
        ]
        
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_question})
        
        tools = build_openai_tools_schema()
        
        for iteration in range(10):
            add_step("🤔 次のアクションを考えています...")
            
            api_params = {
                "model": OPENAI_MODEL,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto"
            }
            
            if OPENAI_MODEL.startswith("gpt-5"):
                api_params["reasoning_effort"] = "high"
            
            response = client.chat.completions.create(**api_params)
            assistant_message = response.choices[0].message
            
            assistant_message_dict = {
                "role": "assistant", 
                "content": assistant_message.content or None
            }
            
            if assistant_message.tool_calls:
                assistant_message_dict["tool_calls"] = [tc.model_dump() for tc in assistant_message.tool_calls]
                messages.append(assistant_message_dict)
                
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    try:
                        if func_name == "list_tables":
                            add_step("📋 テーブル一覧を取得中...")
                            result = await list_tables(session, func_args.get("project", PROJECT_ID))
                            func_result = json.dumps({"tables": result}, ensure_ascii=False)
                            add_step(f"✅ {len(result)}個のテーブルを発見しました")
                            
                        elif func_name == "describe_table":
                            table_name = func_args.get("table") or func_args.get("table_name")
                            if not table_name:
                                raise ValueError(f"Missing required parameter: 'table'")
                            add_step(f"🔍 テーブル '{table_name}' のスキーマを取得中...")
                            result = await describe_table(
                                session,
                                func_args.get("project", PROJECT_ID),
                                func_args.get("dataset", DEFAULT_DATASET),
                                table_name
                            )
                            func_result = json.dumps({"columns": result}, ensure_ascii=False)
                            add_step(f"✅ {len(result)}個のカラムを発見しました")
                            
                        elif func_name == "execute_query":
                            query = func_args.get("query") or func_args.get("sql")
                            if not query:
                                raise ValueError(f"Missing required parameter: 'query'")
                            add_step("⚡ クエリを実行中...")
                            result = await execute_query(
                                session,
                                func_args.get("project", PROJECT_ID),
                                func_args.get("location", LOCATION),
                                query
                            )
                            result_data = result
                            
                            # Convert to DataFrame for Python execution
                            if result:
                                latest_dataframe = pd.DataFrame(result)
                                add_step(f"📊 {len(result)}行のデータを取得しました（Python処理可能）")
                                
                                # If data is large, return summary to avoid token limit
                                if len(result) > 100:
                                    summary = {
                                        "row_count": len(result),
                                        "columns": list(result[0].keys()) if result else [],
                                        "sample_rows": result[:5],
                                        "message": f"Large dataset with {len(result)} rows. Full data available in Python as 'df'. Use execute_python for analysis."
                                    }
                                    func_result = json.dumps(summary, ensure_ascii=False, cls=DateTimeEncoder)
                                else:
                                    func_result = json.dumps({"rows": result}, ensure_ascii=False, cls=DateTimeEncoder)
                            else:
                                latest_dataframe = None
                                add_step("📊 0行のデータを取得しました")
                                func_result = json.dumps({"rows": []}, ensure_ascii=False)
                            
                        elif func_name == "execute_python":
                            code = func_args.get("code")
                            if not code:
                                raise ValueError(f"Missing required parameter: 'code'")
                            
                            add_step("🐍 Pythonコードを実行中...")
                            
                            # Prepare DataFrame dict
                            df_dict = {}
                            if latest_dataframe is not None:
                                df_dict['df'] = latest_dataframe
                            
                            # Execute Python code
                            python_result = execute_python_code(code, df_dict, timeout=30)
                            
                            # Store result
                            python_results.append(python_result)
                            
                            # Build result message
                            result_parts = []
                            if python_result.get('output'):
                                result_parts.append(f"出力:\n{python_result['output']}")
                            if python_result.get('result'):
                                result_parts.append(f"結果: {python_result['result']}")
                            if python_result.get('plots'):
                                result_parts.append(f"{len(python_result['plots'])}個のグラフを生成しました")
                            
                            func_result = json.dumps({
                                "output": python_result.get('output', ''),
                                "result": python_result.get('result'),
                                "plots_count": len(python_result.get('plots', []))
                            }, ensure_ascii=False)
                            
                            add_step(f"✅ Python実行完了: {', '.join(result_parts) if result_parts else 'エラーなし'}")
                            
                        elif func_name == "suggest_chart":
                            add_step("📈 グラフ設定を提案中...")
                            chart_config = {
                                "chart_type": func_args.get("chart_type", "bar"),
                                "x_axis": func_args.get("x_axis"),
                                "y_axis": func_args.get("y_axis"),
                                "title": func_args.get("title", "")
                            }
                            result_charts.append(chart_config)
                            func_result = json.dumps({"chart": chart_config}, ensure_ascii=False)
                            chart_type_ja = {
                                "bar": "棒グラフ",
                                "line": "折れ線グラフ",
                                "pie": "円グラフ",
                                "doughnut": "ドーナツグラフ",
                                "scatter": "散布図",
                                "none": "グラフなし"
                            }.get(chart_config["chart_type"], chart_config["chart_type"])
                            add_step(f"✅ {chart_type_ja}を提案しました")
                            
                        else:
                            func_result = json.dumps({"error": "Unknown function"})
                            
                    except Exception as e:
                        error_msg = str(e)
                        error_details = {
                            "error": error_msg,
                            "function": func_name,
                            "arguments": func_args
                        }
                        
                        if func_name == "execute_query":
                            if "column" in error_msg.lower() or "field" in error_msg.lower():
                                error_details["hint"] = "Column name error detected. Use describe_table to get exact column names (case-sensitive)."
                            elif "table" in error_msg.lower():
                                error_details["hint"] = "Table reference error. Use backticks: `project.dataset.table` or `dataset.table`"
                            elif "type" in error_msg.lower() or "cast" in error_msg.lower():
                                error_details["hint"] = "Data type error. Use SAFE_CAST(column AS FLOAT64) for type conversion."
                            elif "syntax" in error_msg.lower():
                                error_details["hint"] = "SQL syntax error. Check BigQuery syntax: use backticks for tables, proper GROUP BY, etc."
                            else:
                                error_details["hint"] = "Query execution failed. Review the error message and check: 1) Column names (case-sensitive), 2) Table references (use backticks), 3) Data types, 4) BigQuery syntax."
                        
                        add_step(f"❌ {func_name} エラー: {error_msg}")
                        func_result = json.dumps(error_details, ensure_ascii=False)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": func_result
                    })
            else:
                messages.append(assistant_message_dict)
                add_step("✅ 分析完了")
                return {
                    "answer": assistant_message.content,
                    "steps": steps,
                    "data": result_data,
                    "charts": result_charts,
                    "python_results": python_results
                }
        
        add_step("⚠️ 最大反復回数に達しました")
        return {
            "answer": "Maximum iterations reached.",
            "steps": steps,
            "data": result_data,
            "charts": result_charts,
            "python_results": python_results
        }

# @app.route('/health')
# def health_check():
#     """Health check endpoint for Cloud Run and deployment monitoring"""
#     return jsonify({
#         "status": "healthy",
#         "service": "bigquery-ai-agent",
#         "timestamp": datetime.now().isoformat()
#     }), 200

@app.route('/')
@login_required
def index():
    """Render dashboard page"""
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle login"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('SELECT * FROM users WHERE username = %s', (username,))
        user_data = cur.fetchone()
        cur.close()
        conn.close()
        
        if user_data and bcrypt.checkpw(password.encode('utf-8'), user_data['password_hash'].encode('utf-8')):
            user = User(user_data['id'], user_data['username'], user_data['email'])
            login_user(user, remember=request.form.get('remember'))
            
            # Safe redirect - only allow same-origin redirects
            next_page = get_redirect_target()
            if not next_page or not is_safe_url(next_page):
                next_page = url_for('index')
            return redirect(next_page)
        else:
            flash('ユーザー名またはパスワードが正しくありません。', 'error')
    
    return render_template('login.html')

# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     """Handle user registration"""
#     if current_user.is_authenticated:
#         return redirect(url_for('index'))
    
#     if request.method == 'POST':
#         username = request.form.get('username')
#         password = request.form.get('password')
#         email = request.form.get('email', '')
        
#         # Validation
#         if not username or not password:
#             flash('ユーザー名とパスワードは必須です。', 'error')
#             return render_template('signup.html')
        
#         if len(password) < 6:
#             flash('パスワードは6文字以上にしてください。', 'error')
#             return render_template('signup.html')
        
#         # Check if username exists
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
#         cur.execute('SELECT id FROM users WHERE username = %s', (username,))
#         existing_user = cur.fetchone()
        
#         if existing_user:
#             flash('このユーザー名は既に使用されています。', 'error')
#             cur.close()
#             conn.close()
#             return render_template('signup.html')
        
#         # Hash password
#         password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
#         # Insert new user
#         cur.execute(
#             'INSERT INTO users (username, password_hash, email) VALUES (%s, %s, %s) RETURNING id',
#             (username, password_hash, email)
#         )
#         user_id = cur.fetchone()['id']
#         conn.commit()
#         cur.close()
#         conn.close()
        
#         flash('アカウントが作成されました。ログインしてください。', 'success')
#         return redirect(url_for('login'))
    
#     return render_template('signup.html')

# @app.route('/logout')
# @login_required
# def logout():
#     """Handle logout"""
#     logout_user()
#     return redirect(url_for('login'))

# @app.route('/account')
# @login_required
# def account_settings():
#     """Render account settings page"""
#     return render_template('account_settings.html')

# @app.route('/agent-chat')
# @login_required
# def agent_chat_redirect():
#     """Redirect to latest chat session or create new one"""
#     try:
#         # Get active project
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
        
#         cur.execute('''
#             SELECT id FROM projects
#             WHERE user_id = %s AND is_active = true
#             LIMIT 1
#         ''', (current_user.id,))
#         project = cur.fetchone()
        
#         if not project:
#             # No active project - show page with warning
#             cur.close()
#             conn.close()
#             return render_template('agent_chat.html', session_id=None)
        
#         # Get latest session for this project
#         cur.execute('''
#             SELECT id FROM chat_sessions
#             WHERE project_id = %s
#             ORDER BY updated_at DESC
#             LIMIT 1
#         ''', (project['id'],))
#         latest_session = cur.fetchone()
        
#         if latest_session:
#             # Redirect to latest session
#             session_id = latest_session['id']
#             cur.close()
#             conn.close()
#             return redirect(url_for('agent_chat', session_id=session_id))
        
#         # No existing sessions - create new one
#         cur.execute('''
#             INSERT INTO chat_sessions (project_id, title)
#             VALUES (%s, %s)
#             RETURNING id
#         ''', (project['id'], 'New Chat'))
        
#         session = cur.fetchone()
#         conn.commit()
#         cur.close()
#         conn.close()
        
#         # Redirect to session URL
#         return redirect(url_for('agent_chat', session_id=session['id']))
#     except Exception as e:
#         print(f"Error handling chat redirect: {e}")
#         return render_template('agent_chat.html', session_id=None)

# @app.route('/agent-chat/<int:session_id>')
# @login_required
# def agent_chat(session_id):
#     """Render agent chat page for specific session"""
#     try:
#         # Verify session belongs to user
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
        
#         cur.execute('''
#             SELECT s.id 
#             FROM chat_sessions s
#             JOIN projects p ON p.id = s.project_id
#             WHERE s.id = %s AND p.user_id = %s
#             LIMIT 1
#         ''', (session_id, current_user.id))
        
#         session = cur.fetchone()
#         cur.close()
#         conn.close()
        
#         if not session:
#             # Session not found or doesn't belong to user
#             flash('セッションが見つかりませんでした。', 'error')
#             return redirect(url_for('agent_chat_redirect'))
        
#         return render_template('agent_chat.html', session_id=session_id)
#     except Exception as e:
#         print(f"Error loading session: {e}")
#         return redirect(url_for('agent_chat_redirect'))

# @app.route('/api/chat', methods=['POST'])
# @login_required
# def chat():
#     """Start chat task and return task_id for polling"""
#     try:
#         data = request.json
#         question = data.get('question', '')
#         history = data.get('history', [])
#         session_id = data.get('session_id')
        
#         if not question:
#             return jsonify({"error": "Question is required"}), 400
        
#         # Get active project configuration
#         config = get_active_project_config(current_user.id)
        
#         if not config['api_key']:
#             return jsonify({"error": "OpenAI API key not configured"}), 500
        
#         if not config['project_id']:
#             return jsonify({"error": "BigQuery project not configured"}), 500
        
#         # Get active project ID
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
#         cur.execute('SELECT id FROM projects WHERE user_id = %s AND is_active = true LIMIT 1', (current_user.id,))
#         active_project = cur.fetchone()
#         active_project_id = active_project['id'] if active_project else None
        
#         # Create or verify session
#         if session_id:
#             cur.execute('''
#                 SELECT id FROM chat_sessions 
#                 WHERE id = %s AND project_id = %s
#             ''', (session_id, active_project_id))
#             if not cur.fetchone():
#                 session_id = None
        
#         if not session_id and active_project_id:
#             cur.execute('''
#                 INSERT INTO chat_sessions (project_id, title)
#                 VALUES (%s, %s)
#                 RETURNING id
#             ''', (active_project_id, 'New Chat'))
#             new_session = cur.fetchone()
#             session_id = new_session['id']
#             conn.commit()
        
#         cur.close()
#         conn.close()
        
#         # Generate task ID
#         import uuid
#         task_id = str(uuid.uuid4())
        
#         # Initialize task state
#         with task_lock:
#             chat_tasks[task_id] = {
#                 'status': 'running',
#                 'steps': ['🔧 処理を開始しました...'],
#                 'result': None,
#                 'error': None
#             }
        
#         # Run task in background thread
#         def run_task():
#             try:
#                 result = asyncio.run(run_agent_with_steps(
#                     task_id, question, history,
#                     api_key=config['api_key'],
#                     project_id=config['project_id'],
#                     dataset_id=config['dataset_id'],
#                     service_account_json=config['service_account_json']
#                 ))
                
#                 # Save to database
#                 if active_project_id and session_id:
#                     conn = get_db_connection()
#                     cur = conn.cursor(cursor_factory=RealDictCursor)
#                     cur.execute('''
#                         INSERT INTO chat_history (project_id, session_id, user_message, ai_response, query_result)
#                         VALUES (%s, %s, %s, %s, %s)
#                     ''', (
#                         active_project_id,
#                         session_id,
#                         question,
#                         result.get('answer', ''),
#                         json.dumps(result, ensure_ascii=False, cls=DateTimeEncoder)
#                     ))
                    
#                     # Update session timestamp and title
#                     cur.execute('''
#                         UPDATE chat_sessions 
#                         SET updated_at = CURRENT_TIMESTAMP 
#                         WHERE id = %s
#                     ''', (session_id,))
                    
#                     cur.execute('''
#                         SELECT title, 
#                                (SELECT COUNT(*) FROM chat_history WHERE session_id = %s) as msg_count
#                         FROM chat_sessions 
#                         WHERE id = %s
#                     ''', (session_id, session_id))
#                     session_info = cur.fetchone()
                    
#                     if session_info and session_info['title'] == 'New Chat' and session_info['msg_count'] == 1:
#                         auto_title = question[:50] + ('...' if len(question) > 50 else '')
#                         cur.execute('''
#                             UPDATE chat_sessions 
#                             SET title = %s 
#                             WHERE id = %s
#                         ''', (auto_title, session_id))
                    
#                     conn.commit()
#                     cur.close()
#                     conn.close()
                
#                 with task_lock:
#                     chat_tasks[task_id]['status'] = 'completed'
#                     chat_tasks[task_id]['result'] = result
#                     chat_tasks[task_id]['session_id'] = session_id
                    
#             except Exception as e:
#                 import traceback
#                 error_traceback = traceback.format_exc()
#                 print(f"ERROR in run_task: {error_traceback}", flush=True)
#                 with task_lock:
#                     chat_tasks[task_id]['status'] = 'error'
#                     chat_tasks[task_id]['error'] = str(e)
#                     chat_tasks[task_id]['steps'].append(f"❌ エラー: {str(e)}")
#                     chat_tasks[task_id]['traceback'] = error_traceback
        
#         thread = threading.Thread(target=run_task)
#         thread.daemon = True
#         thread.start()
        
#         return jsonify({
#             "task_id": task_id,
#             "session_id": session_id
#         })
    
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# @app.route('/api/chat/status/<task_id>', methods=['GET'])
# @login_required
# def chat_status(task_id):
#     """Get chat task status and progress"""
#     with task_lock:
#         task = chat_tasks.get(task_id)
#         if not task:
#             return jsonify({"error": "Task not found"}), 404
        
#         return jsonify({
#             "status": task['status'],
#             "steps": task['steps'],
#             "result": task['result'],
#             "error": task['error'],
#             "session_id": task.get('session_id')
#         })

# @app.route('/api/config', methods=['GET'])
# def get_config():
#     """Get configuration status"""
#     return jsonify({
#         "project_id": PROJECT_ID,
#         "dataset": DEFAULT_DATASET,
#         "openai_configured": bool(OPENAI_API_KEY),
#         "gcp_configured": bool(GCP_SA_JSON)
#     })

# @app.route('/projects')
# @login_required
# def projects():
#     """Render projects page"""
#     return render_template('projects.html')

# @app.route('/settings')
# @login_required
# def settings():
#     """Render settings page"""
#     return render_template('settings.html')

# @app.route('/api/settings', methods=['GET'])
# @login_required
# def get_settings():
#     """Get active project settings"""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
#         cur.execute('''
#             SELECT id, name, description, bigquery_project_id, bigquery_dataset_id,
#                    openai_api_key IS NOT NULL as has_api_key,
#                    service_account_json IS NOT NULL as has_service_account
#             FROM projects
#             WHERE user_id = %s AND is_active = true
#             LIMIT 1
#         ''', (current_user.id,))
#         project = cur.fetchone()
#         cur.close()
#         conn.close()
        
#         if not project:
#             return jsonify({
#                 "success": False,
#                 "error": "アクティブなプロジェクトが選択されていません。プロジェクトを作成・選択してください。",
#                 "no_project": True
#             }), 404
        
#         return jsonify({
#             "success": True,
#             "project_id": project['bigquery_project_id'] or '',
#             "dataset": project['bigquery_dataset_id'] or '',
#             "has_api_key": project['has_api_key'],
#             "has_service_account": project['has_service_account'],
#             "project_name": project['name'],
#             "project_description": project['description'] or '',
#             "openai_model": OPENAI_MODEL,  # モデルはグローバル設定
#             "location": LOCATION  # ロケーションはグローバル設定
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# @app.route('/api/settings', methods=['POST'])
# @login_required
# def save_settings():
#     """Save settings to active project"""
#     try:
#         # Get active project
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
#         cur.execute('''
#             SELECT id FROM projects
#             WHERE user_id = %s AND is_active = true
#             LIMIT 1
#         ''', (current_user.id,))
#         project = cur.fetchone()
        
#         if not project:
#             cur.close()
#             conn.close()
#             return jsonify({
#                 "success": False,
#                 "error": "アクティブなプロジェクトが選択されていません"
#             }), 404
        
#         project_id = project['id']
        
#         # Build update fields
#         update_fields = []
#         params = []
        
#         # Handle project name
#         if request.form.get('project_name'):
#             update_fields.append('name = %s')
#             params.append(request.form.get('project_name'))
        
#         # Handle project description
#         if 'project_description' in request.form:
#             update_fields.append('description = %s')
#             params.append(request.form.get('project_description'))
        
#         # Handle OpenAI API key
#         if request.form.get('openai_key'):
#             update_fields.append('openai_api_key = %s')
#             params.append(request.form.get('openai_key'))
        
#         # Handle BigQuery project ID
#         if request.form.get('project_id'):
#             update_fields.append('bigquery_project_id = %s')
#             params.append(request.form.get('project_id'))
        
#         # Handle BigQuery dataset
#         if request.form.get('default_dataset'):
#             update_fields.append('bigquery_dataset_id = %s')
#             params.append(request.form.get('default_dataset'))
        
#         # Handle GCP JSON file upload
#         if 'gcp_json' in request.files:
#             json_file = request.files['gcp_json']
#             if json_file.filename:
#                 # Save the JSON file with project-specific name
#                 json_path = os.path.join(os.getcwd(), f'gcp_credentials_project_{project_id}.json')
#                 json_file.save(json_path)
#                 update_fields.append('service_account_json = %s')
#                 params.append(json_path)
        
#         if not update_fields:
#             cur.close()
#             conn.close()
#             return jsonify({
#                 "success": False,
#                 "error": "更新する設定がありません"
#             }), 400
        
#         # Update project
#         update_fields.append('updated_at = CURRENT_TIMESTAMP')
#         params.extend([project_id, current_user.id])
        
#         query = f'''
#             UPDATE projects
#             SET {', '.join(update_fields)}
#             WHERE id = %s AND user_id = %s
#         '''
        
#         cur.execute(query, params)
#         conn.commit()
#         cur.close()
#         conn.close()
        
#         return jsonify({
#             "success": True,
#             "message": "設定を保存しました"
#         })
    
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# @app.route('/api/get-api-key')
# @login_required
# def get_api_key():
#     """Get OpenAI API key for active project"""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
#         cur.execute('''
#             SELECT openai_api_key FROM projects
#             WHERE user_id = %s AND is_active = true
#             LIMIT 1
#         ''', (current_user.id,))
#         project = cur.fetchone()
#         cur.close()
#         conn.close()
        
#         if not project:
#             return jsonify({"api_key": ""})
        
#         return jsonify({
#             "api_key": project['openai_api_key'] or ""
#         })
#     except Exception as e:
#         return jsonify({"api_key": ""}), 500

# @app.route('/api/get-json-file-info')
# @login_required
# def get_json_file_info():
#     """Get information about uploaded JSON file for active project"""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
#         cur.execute('''
#             SELECT service_account_json FROM projects
#             WHERE user_id = %s AND is_active = true
#             LIMIT 1
#         ''', (current_user.id,))
#         project = cur.fetchone()
#         cur.close()
#         conn.close()
        
#         if not project:
#             return jsonify({"exists": False})
        
#         json_path = project['service_account_json']
#         if json_path and os.path.exists(json_path):
#             filename = os.path.basename(json_path)
#             return jsonify({
#                 "exists": True,
#                 "filename": filename,
#                 "path": json_path
#             })
        
#         return jsonify({"exists": False})
#     except Exception as e:
#         return jsonify({"exists": False}), 500

# @app.route('/api/delete-json-file', methods=['POST'])
# @login_required
# def delete_json_file():
#     """Delete uploaded JSON file for active project"""
#     try:
#         # Get active project
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
#         cur.execute('''
#             SELECT id, service_account_json FROM projects
#             WHERE user_id = %s AND is_active = true
#             LIMIT 1
#         ''', (current_user.id,))
#         project = cur.fetchone()
        
#         if not project:
#             cur.close()
#             conn.close()
#             return jsonify({
#                 "success": False,
#                 "error": "アクティブなプロジェクトが選択されていません"
#             }), 404
        
#         # Delete physical file if it exists
#         json_path = project['service_account_json']
#         if json_path and os.path.exists(json_path):
#             os.remove(json_path)
        
#         # Update database to remove JSON path
#         cur.execute('''
#             UPDATE projects
#             SET service_account_json = NULL, updated_at = CURRENT_TIMESTAMP
#             WHERE id = %s AND user_id = %s
#         ''', (project['id'], current_user.id))
        
#         conn.commit()
#         cur.close()
#         conn.close()
        
#         return jsonify({
#             "success": True,
#             "message": "JSON file deleted successfully"
#         })
    
#     except Exception as e:
#         return jsonify({
#             "error": str(e)
#         }), 500

# @app.route('/api/test-connection', methods=['POST'])
# @login_required
# def test_connection():
#     """Test OpenAI and BigQuery connection for active project"""
#     try:
#         # Get active project configuration
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
#         cur.execute('''
#             SELECT openai_api_key, bigquery_project_id, bigquery_dataset_id, service_account_json
#             FROM projects
#             WHERE user_id = %s AND is_active = true
#             LIMIT 1
#         ''', (current_user.id,))
#         project = cur.fetchone()
#         cur.close()
#         conn.close()
        
#         if not project:
#             return jsonify({
#                 "success": False,
#                 "error": "アクティブなプロジェクトが選択されていません"
#             }), 404
        
#         api_key = project['openai_api_key']
#         project_id = project['bigquery_project_id']
#         gcp_json = project['service_account_json']
        
#         # Test OpenAI
#         if not api_key:
#             return jsonify({"error": "OpenAI API key が設定されていません"}), 400
        
#         try:
#             client = OpenAI(api_key=api_key)
#             client.models.list()
#             openai_ok = True
#         except Exception as e:
#             return jsonify({"error": f"OpenAI 接続エラー: {str(e)}"}), 400
        
#         # Test BigQuery
#         if not project_id or not gcp_json:
#             return jsonify({"error": "BigQuery が完全に設定されていません"}), 400
        
#         if not os.path.exists(gcp_json):
#             return jsonify({"error": f"GCP 認証ファイルが見つかりません: {gcp_json}"}), 400
        
#         return jsonify({
#             "success": True,
#             "message": "すべての接続が成功しました！OpenAIとBigQueryが正しく設定されています。"
#         })
    
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# # ============================================
# # Project Management API
# # ============================================

# @app.route('/api/projects', methods=['GET'])
# @login_required
# def get_projects():
#     """Get all projects for current user"""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
#         cur.execute('''
#             SELECT id, name, description, bigquery_project_id, bigquery_dataset_id,
#                    is_active, created_at, updated_at
#             FROM projects
#             WHERE user_id = %s
#             ORDER BY created_at DESC
#         ''', (current_user.id,))
#         projects = cur.fetchall()
#         cur.close()
#         conn.close()
        
#         return jsonify({
#             "success": True,
#             "projects": [dict(p) for p in projects]
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# @app.route('/api/projects', methods=['POST'])
# @login_required
# def create_project():
#     """Create a new project"""
#     try:
#         data = request.json
#         name = data.get('name', '').strip()
#         description = data.get('description', '').strip()
        
#         if not name:
#             return jsonify({"error": "Project name is required"}), 400
        
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
        
#         # Create project
#         cur.execute('''
#             INSERT INTO projects (user_id, name, description)
#             VALUES (%s, %s, %s)
#             RETURNING id, name, description, is_active, created_at, updated_at
#         ''', (current_user.id, name, description))
        
#         project = cur.fetchone()
#         conn.commit()
#         cur.close()
#         conn.close()
        
#         return jsonify({
#             "success": True,
#             "project": dict(project),
#             "message": "Project created successfully"
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# @app.route('/api/projects/<int:project_id>', methods=['GET'])
# @login_required
# def get_project(project_id):
#     """Get project details"""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
#         cur.execute('''
#             SELECT id, name, description, bigquery_project_id, bigquery_dataset_id,
#                    is_active, created_at, updated_at,
#                    openai_api_key IS NOT NULL as has_api_key,
#                    service_account_json IS NOT NULL as has_service_account
#             FROM projects
#             WHERE id = %s AND user_id = %s
#         ''', (project_id, current_user.id))
        
#         project = cur.fetchone()
#         cur.close()
#         conn.close()
        
#         if not project:
#             return jsonify({"error": "Project not found"}), 404
        
#         return jsonify({
#             "success": True,
#             "project": dict(project)
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# @app.route('/api/projects/<int:project_id>', methods=['PUT'])
# @login_required
# def update_project(project_id):
#     """Update project"""
#     try:
#         data = request.json
        
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
        
#         # Check if project exists and belongs to user
#         cur.execute('SELECT id FROM projects WHERE id = %s AND user_id = %s', 
#                    (project_id, current_user.id))
#         if not cur.fetchone():
#             cur.close()
#             conn.close()
#             return jsonify({"error": "Project not found"}), 404
        
#         # Build update query dynamically
#         update_fields = []
#         params = []
        
#         if 'name' in data:
#             update_fields.append('name = %s')
#             params.append(data['name'])
#         if 'description' in data:
#             update_fields.append('description = %s')
#             params.append(data['description'])
#         if 'openai_api_key' in data:
#             update_fields.append('openai_api_key = %s')
#             params.append(data['openai_api_key'])
#         if 'bigquery_project_id' in data:
#             update_fields.append('bigquery_project_id = %s')
#             params.append(data['bigquery_project_id'])
#         if 'bigquery_dataset_id' in data:
#             update_fields.append('bigquery_dataset_id = %s')
#             params.append(data['bigquery_dataset_id'])
#         if 'service_account_json' in data:
#             update_fields.append('service_account_json = %s')
#             params.append(data['service_account_json'])
        
#         if not update_fields:
#             return jsonify({"error": "No fields to update"}), 400
        
#         update_fields.append('updated_at = CURRENT_TIMESTAMP')
#         params.extend([project_id, current_user.id])
        
#         query = f'''
#             UPDATE projects
#             SET {', '.join(update_fields)}
#             WHERE id = %s AND user_id = %s
#             RETURNING id, name, description, bigquery_project_id, bigquery_dataset_id,
#                       is_active, created_at, updated_at
#         '''
        
#         cur.execute(query, params)
#         project = cur.fetchone()
#         conn.commit()
#         cur.close()
#         conn.close()
        
#         return jsonify({
#             "success": True,
#             "project": dict(project),
#             "message": "Project updated successfully"
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# @app.route('/api/projects/<int:project_id>', methods=['DELETE'])
# @login_required
# def delete_project(project_id):
#     """Delete project"""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor()
        
#         # Check if project exists and belongs to user
#         cur.execute('SELECT id FROM projects WHERE id = %s AND user_id = %s', 
#                    (project_id, current_user.id))
#         if not cur.fetchone():
#             cur.close()
#             conn.close()
#             return jsonify({"error": "Project not found"}), 404
        
#         # Delete project (cascade will delete chat history)
#         cur.execute('DELETE FROM projects WHERE id = %s AND user_id = %s', 
#                    (project_id, current_user.id))
#         conn.commit()
#         cur.close()
#         conn.close()
        
#         return jsonify({
#             "success": True,
#             "message": "Project deleted successfully"
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# @app.route('/api/projects/<int:project_id>/activate', methods=['POST'])
# @login_required
# def activate_project(project_id):
#     """Set project as active"""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
        
#         # Check if project exists and belongs to user
#         cur.execute('SELECT id FROM projects WHERE id = %s AND user_id = %s', 
#                    (project_id, current_user.id))
#         if not cur.fetchone():
#             cur.close()
#             conn.close()
#             return jsonify({"error": "Project not found"}), 404
        
#         # Deactivate all projects for this user
#         cur.execute('UPDATE projects SET is_active = false WHERE user_id = %s', 
#                    (current_user.id,))
        
#         # Activate selected project
#         cur.execute('''
#             UPDATE projects SET is_active = true 
#             WHERE id = %s AND user_id = %s
#             RETURNING id, name, description, bigquery_project_id, bigquery_dataset_id,
#                       is_active, created_at, updated_at
#         ''', (project_id, current_user.id))
        
#         project = cur.fetchone()
#         conn.commit()
#         cur.close()
#         conn.close()
        
#         return jsonify({
#             "success": True,
#             "project": dict(project),
#             "message": f"Project '{project['name']}' is now active"
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# @app.route('/api/projects/active', methods=['GET'])
# @login_required
# def get_active_project():
#     """Get currently active project"""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
#         cur.execute('''
#             SELECT id, name, description, bigquery_project_id, bigquery_dataset_id,
#                    openai_api_key, service_account_json, is_active, created_at, updated_at
#             FROM projects
#             WHERE user_id = %s AND is_active = true
#             LIMIT 1
#         ''', (current_user.id,))
        
#         project = cur.fetchone()
#         cur.close()
#         conn.close()
        
#         if not project:
#             return jsonify({
#                 "success": True,
#                 "project": None
#             })
        
#         return jsonify({
#             "success": True,
#             "project": dict(project)
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# # ============================================
# # Chat Session Management API
# # ============================================

# @app.route('/api/chat-sessions', methods=['POST'])
# @login_required
# def create_chat_session():
#     """Create a new chat session for active project"""
#     try:
#         # Get active project
#         config = get_active_project_config(current_user.id)
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
        
#         cur.execute('''
#             SELECT id FROM projects
#             WHERE user_id = %s AND is_active = true
#             LIMIT 1
#         ''', (current_user.id,))
#         project = cur.fetchone()
        
#         if not project:
#             cur.close()
#             conn.close()
#             return jsonify({"error": "No active project"}), 404
        
#         # Create new session
#         cur.execute('''
#             INSERT INTO chat_sessions (project_id, title)
#             VALUES (%s, %s)
#             RETURNING id, project_id, title, created_at, updated_at
#         ''', (project['id'], 'New Chat'))
        
#         session = cur.fetchone()
#         conn.commit()
#         cur.close()
#         conn.close()
        
#         return jsonify({
#             "success": True,
#             "session": dict(session)
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# @app.route('/api/chat-sessions', methods=['GET'])
# @login_required
# def get_chat_sessions():
#     """Get all chat sessions for active project"""
#     try:
#         # Get active project
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
        
#         cur.execute('''
#             SELECT id FROM projects
#             WHERE user_id = %s AND is_active = true
#             LIMIT 1
#         ''', (current_user.id,))
#         project = cur.fetchone()
        
#         if not project:
#             cur.close()
#             conn.close()
#             return jsonify({
#                 "success": True,
#                 "sessions": []
#             })
        
#         # Get sessions with message count
#         cur.execute('''
#             SELECT 
#                 s.id, 
#                 s.project_id, 
#                 s.title, 
#                 s.created_at, 
#                 s.updated_at,
#                 COUNT(h.id) as message_count
#             FROM chat_sessions s
#             LEFT JOIN chat_history h ON h.session_id = s.id
#             WHERE s.project_id = %s
#             GROUP BY s.id, s.project_id, s.title, s.created_at, s.updated_at
#             ORDER BY s.updated_at DESC
#         ''', (project['id'],))
        
#         sessions = cur.fetchall()
#         cur.close()
#         conn.close()
        
#         return jsonify({
#             "success": True,
#             "sessions": [dict(s) for s in sessions]
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# @app.route('/api/chat-sessions/<int:session_id>/messages', methods=['GET'])
# @login_required
# def get_session_messages(session_id):
#     """Get all messages for a specific chat session"""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
        
#         # Verify session belongs to user's active project
#         cur.execute('''
#             SELECT s.id 
#             FROM chat_sessions s
#             JOIN projects p ON p.id = s.project_id
#             WHERE s.id = %s AND p.user_id = %s
#             LIMIT 1
#         ''', (session_id, current_user.id))
        
#         if not cur.fetchone():
#             cur.close()
#             conn.close()
#             return jsonify({"error": "Session not found"}), 404
        
#         # Get all messages for this session
#         cur.execute('''
#             SELECT id, user_message, ai_response, query_result, created_at
#             FROM chat_history
#             WHERE session_id = %s
#             ORDER BY created_at ASC
#         ''', (session_id,))
        
#         messages = cur.fetchall()
#         cur.close()
#         conn.close()
        
#         return jsonify({
#             "success": True,
#             "messages": [dict(m) for m in messages]
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# @app.route('/api/chat-sessions/<int:session_id>/title', methods=['PUT'])
# @login_required
# def update_session_title(session_id):
#     """Update chat session title"""
#     try:
#         data = request.json
#         new_title = data.get('title', '').strip()
        
#         if not new_title:
#             return jsonify({"error": "Title is required"}), 400
        
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
        
#         # Verify session belongs to user's active project
#         cur.execute('''
#             SELECT s.id 
#             FROM chat_sessions s
#             JOIN projects p ON p.id = s.project_id
#             WHERE s.id = %s AND p.user_id = %s
#             LIMIT 1
#         ''', (session_id, current_user.id))
        
#         if not cur.fetchone():
#             cur.close()
#             conn.close()
#             return jsonify({"error": "Session not found"}), 404
        
#         # Update title
#         cur.execute('''
#             UPDATE chat_sessions
#             SET title = %s, updated_at = CURRENT_TIMESTAMP
#             WHERE id = %s
#             RETURNING id, project_id, title, created_at, updated_at
#         ''', (new_title, session_id))
        
#         session = cur.fetchone()
#         conn.commit()
#         cur.close()
#         conn.close()
        
#         return jsonify({
#             "success": True,
#             "session": dict(session)
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# @app.route('/api/chat-sessions/<int:session_id>', methods=['DELETE'])
# @login_required
# def delete_session(session_id):
#     """Delete a chat session"""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
        
#         # Verify session belongs to user
#         cur.execute('''
#             SELECT s.id 
#             FROM chat_sessions s
#             JOIN projects p ON p.id = s.project_id
#             WHERE s.id = %s AND p.user_id = %s
#             LIMIT 1
#         ''', (session_id, current_user.id))
        
#         if not cur.fetchone():
#             cur.close()
#             conn.close()
#             return jsonify({"error": "Session not found"}), 404
        
#         # Delete session (cascade will delete related chat_history)
#         cur.execute('DELETE FROM chat_sessions WHERE id = %s', (session_id,))
        
#         conn.commit()
#         cur.close()
#         conn.close()
        
#         return jsonify({
#             "success": True,
#             "message": "Session deleted successfully"
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# # ============================================
# # Account Settings API
# # ============================================

# @app.route('/api/account/password', methods=['PUT'])
# @login_required
# def change_password():
#     """Change user password"""
#     try:
#         data = request.json
#         current_password = data.get('current_password', '')
#         new_password = data.get('new_password', '')
        
#         if not current_password or not new_password:
#             return jsonify({"error": "Current password and new password are required"}), 400
        
#         if len(new_password) < 6:
#             return jsonify({"error": "New password must be at least 6 characters"}), 400
        
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
        
#         # Get current password hash
#         cur.execute('SELECT password_hash FROM users WHERE id = %s', (current_user.id,))
#         user = cur.fetchone()
        
#         if not user:
#             cur.close()
#             conn.close()
#             return jsonify({"error": "User not found"}), 404
        
#         # Verify current password
#         if not bcrypt.checkpw(current_password.encode('utf-8'), user['password_hash'].encode('utf-8')):
#             cur.close()
#             conn.close()
#             return jsonify({"error": "Current password is incorrect"}), 401
        
#         # Hash new password
#         new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
#         # Update password
#         cur.execute('''
#             UPDATE users
#             SET password_hash = %s
#             WHERE id = %s
#         ''', (new_password_hash, current_user.id))
        
#         conn.commit()
#         cur.close()
#         conn.close()
        
#         return jsonify({
#             "success": True,
#             "message": "Password changed successfully"
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# @app.route('/api/account/email', methods=['PUT'])
# @login_required
# def update_email():
#     """Update user email"""
#     try:
#         data = request.json
#         email = data.get('email', '').strip()
        
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
        
#         # Update email
#         cur.execute('''
#             UPDATE users
#             SET email = %s
#             WHERE id = %s
#         ''', (email if email else None, current_user.id))
        
#         conn.commit()
#         cur.close()
#         conn.close()
        
#         return jsonify({
#             "success": True,
#             "message": "Email updated successfully"
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# @app.route('/api/account', methods=['DELETE'])
# @login_required
# def delete_account():
#     """Delete user account"""
#     try:
#         data = request.json
#         password = data.get('password', '')
        
#         if not password:
#             return jsonify({"error": "Password is required"}), 400
        
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
        
#         # Get current password hash
#         cur.execute('SELECT password_hash FROM users WHERE id = %s', (current_user.id,))
#         user = cur.fetchone()
        
#         if not user:
#             cur.close()
#             conn.close()
#             return jsonify({"error": "User not found"}), 404
        
#         # Verify password
#         if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
#             cur.close()
#             conn.close()
#             return jsonify({"error": "Password is incorrect"}), 401
        
#         # Delete user (cascades to projects, chat_sessions, chat_history)
#         cur.execute('DELETE FROM users WHERE id = %s', (current_user.id,))
        
#         conn.commit()
#         cur.close()
#         conn.close()
        
#         # Logout user
#         logout_user()
        
#         return jsonify({
#             "success": True,
#             "message": "Account deleted successfully"
#         })
#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }), 500

# Security headers for production
# @app.after_request
# def set_security_headers(response):
#     """Set security headers for all responses"""
#     # Only set security headers in production
#     if os.getenv('FLASK_ENV') == 'production':
#         response.headers['X-Content-Type-Options'] = 'nosniff'
#         response.headers['X-Frame-Options'] = 'DENY'
#         response.headers['X-XSS-Protection'] = '1; mode=block'
#         response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
#         response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://fonts.googleapis.com; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net; img-src 'self' data:; connect-src 'self';"
#     return response


@app.route("/test")
def root():
    return jsonify(
        {
            "message": "Cloud Run + Cloud SQL minimal CRUD API",
            "endpoints": ["/items (GET, POST)", "/items/<id> (GET, PUT, DELETE)"],
        }
    )

# Cloud Run で直接実行されることはあまり無いが、ローカル動作用
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
