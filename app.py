import matplotlib
matplotlib.use("Agg")  # Fixes the Matplotlib GUI error

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import google.generativeai as genai
import re
import pyodbc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import logging
# Load environment variables
load_dotenv()

# Configure Google Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
CORS(app)  # Allow frontend requests

logging.basicConfig(level=logging.DEBUG)
# ============================ #
# SQL QUERY GENERATION & EXECUTION #
# ============================ #

def connect_to_sql_server(server, database):
    try:
        conn = pyodbc.connect(
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            'Trusted_Connection=yes;'
        )
        return conn, None
    except pyodbc.Error as e:
        return None, str(e)

def get_table_schema(server, database):
    conn, error = connect_to_sql_server(server, database)
    if error:
        return {}
    try:
        query = "SELECT TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS"
        with conn.cursor() as cursor:
            cursor.execute(query)
            schema = {}
            for table, column in cursor.fetchall():
                schema.setdefault(table, []).append(column)
        return schema
    except pyodbc.Error:
        return {}
    finally:
        conn.close()

def get_gemini_response(question, dialect, database, schema):
    prompt = f"""
    You are an expert in converting English questions into valid {dialect} SQL queries!
    The database in use is '{database}', and it contains the following tables with their columns:
    {schema}
    Always generate queries using only the tables listed above.
    Ensure the query is syntactically correct for {dialect}.
    """
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    full_prompt = f"{prompt}\n\nUser's question: {question}"
    response = model.generate_content(full_prompt).text.strip()
    return response

def clean_sql_query(query):
    return re.sub(r'sql|', '', query, flags=re.IGNORECASE).strip()

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    server = data.get("server")
    database = data.get("database")
    dialect = data.get("dialect", "SQL Server")
    question = data.get("question")
    
    if not server or not database or not question:
        return jsonify({"error": "Missing required parameters."}), 400
    
    schema = get_table_schema(server, database)
    generated_query = get_gemini_response(question, dialect, database, schema)
    
    if "ERROR" in generated_query.upper():
        return jsonify({"error": "Invalid query generated."})
    
    conn, error = connect_to_sql_server(server, database)
    if error:
        return jsonify({"error": error}), 500
    
    try:
        query = clean_sql_query(generated_query)
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        return jsonify({"query": query, "result": {"columns": columns, "rows": [list(row) for row in rows]}})
    except pyodbc.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ============================ #
# AI-POWERED GRAPH VISUALIZER #
# ============================ #

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logging.error("No file part in request")
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        if file.filename.endswith(".csv"):
            logging.info("Processing CSV file")
            data = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip")
        elif file.filename.endswith(".xlsx"):
            logging.info("Processing Excel file")
            data = pd.read_excel(file, engine="openpyxl")
        else:
            logging.error("Unsupported file type")
            return jsonify({"error": "Unsupported file type"}), 400
        
        return jsonify({
            "columns": data.columns.tolist(),
            "preview": data.head().to_dict(orient="records")
        })
    except Exception as e:
        logging.error(f"File processing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate-graph', methods=['POST'])
def generate_graph():
    data_json = request.json
    if not all(key in data_json for key in ["data", "x_axis", "y_axis", "chart_type"]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    df = pd.DataFrame(data_json["data"])
    x_axis = data_json["x_axis"]
    y_axis = data_json["y_axis"]
    chart_type = data_json["chart_type"]
    
    fig, ax = plt.subplots()
    try:
        if chart_type == "Bar Chart":
            sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif chart_type == "Line Chart":
            sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif chart_type == "Scatter Plot":
            sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif chart_type == "Pie Chart":
            df.groupby(x_axis)[y_axis].sum().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        elif chart_type == "Histogram":
            sns.histplot(df[y_axis], kde=True, ax=ax)
        elif chart_type == "Box Plot":
            sns.boxplot(x=df[x_axis], y=df[y_axis], ax=ax)
        else:
            return jsonify({"error": "Invalid chart type"}), 400
        
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"{chart_type} of {y_axis} vs {x_axis}")
        
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format="png")
        plt.close(fig)
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode()
        
        return jsonify({"image": img_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================ #
# SQL QUERY VALIDATION & CORRECTION #
# ============================ #

def correct_sql_query(query, dialect):
    prompt = f"""
    You are an SQL expert. A user provided the following SQL query with syntax errors:
    
    ```sql
    {query}
    ```
    
    Identify the error, explain what went wrong, and provide a corrected SQL query.
    Ensure the corrected query follows {dialect} syntax.
    
    Respond in this format:
    - **Error Explanation**: <explanation>
    - **Corrected SQL Query**:
    ```sql
    <corrected query>
    ```
    """

    model = genai.GenerativeModel('gemini-1.5-pro')  # Use the correct model
    response = model.generate_content(prompt).text.strip()
    return response

def clean_sql_query(response):
    """Extract only the SQL query from Gemini response"""
    match = re.search(r'```sql(.*?)```', response, re.DOTALL)
    return match.group(1).strip() if match else response  # Remove Markdown formatting

@app.route('/validate-query', methods=['POST'])
def validate_query():
    data = request.json
    query = data.get("query")
    dialect = data.get("dialect", "SQL Server")

    if not query:
        return jsonify({"error": "Missing query input."}), 400

    correction_response = correct_sql_query(query, dialect)
    corrected_query = clean_sql_query(correction_response)

    return jsonify({
        "original_query": query,
        "correction": corrected_query,
        "explanation": correction_response  # Full AI response with explanation
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
