from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

import pandas as pd

from src.project_utils import (
    MODEL_PATH,
    SCHEMA_PATH,
    load_json,
    load_pickle,
    probability_to_decision,
    probability_to_risk_band,
    probability_to_score,
)


HOST = "127.0.0.1"
PORT = 8002


def load_artifacts():
    model = load_pickle(MODEL_PATH)
    schema = load_json(SCHEMA_PATH)
    return model, schema


def build_form(schema, values=None):
    values = values or {}
    fields = []

    for column in schema["numeric_columns"]:
        value = values.get(column, schema["numeric_defaults"][column])
        fields.append(
            f"""
            <label>{column}</label>
            <input name="{column}" type="number" step="any" value="{value}">
            """
        )

    for column in schema["categorical_columns"]:
        selected_value = values.get(column)
        options = []
        for option in schema["categorical_options"].get(column, []):
            selected = "selected" if option == selected_value else ""
            options.append(f'<option value="{option}" {selected}>{option}</option>')

        fields.append(
            f"""
            <label>{column}</label>
            <select name="{column}">
                {''.join(options)}
            </select>
            """
        )

    return "\n".join(fields)


def render_page(schema, result_html="", values=None):
    form_fields = build_form(schema, values)
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Credit Risk Scoring</title>
        <style>
            body {{ margin: 0; font-family: Arial, sans-serif; background: #0f172a; color: #e5edf7; }}
            main {{ max-width: 1050px; margin: 0 auto; padding: 28px 18px 44px; }}
            h1 {{ margin-bottom: 4px; }}
            p {{ color: #aeb9ca; }}
            form {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 14px; }}
            label {{ display: block; margin-bottom: 5px; color: #cbd5e1; font-size: 14px; }}
            input, select {{ width: 100%; padding: 10px; border: 1px solid #334155; background: #111827; color: #f8fafc; }}
            button {{ margin-top: 18px; padding: 12px 18px; border: 0; background: #38bdf8; color: #082f49; font-weight: 700; cursor: pointer; }}
            .button-row {{ grid-column: 1 / -1; }}
            .result {{ margin: 22px 0; padding: 18px; background: #111827; border: 1px solid #334155; }}
            .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
            .metric {{ padding: 14px; background: #172033; border: 1px solid #334155; }}
            .metric strong {{ display: block; font-size: 22px; margin-top: 4px; }}
        </style>
    </head>
    <body>
    <main>
        <h1>Credit Risk Scoring & Explainability</h1>
        <p>Enter applicant details to calculate default probability, risk score, risk band and lending decision.</p>
        {result_html}
        <form method="post">
            {form_fields}
            <div class="button-row">
                <button type="submit">Calculate Credit Risk</button>
            </div>
        </form>
    </main>
    </body>
    </html>
    """


class CreditRiskHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        _, schema = load_artifacts()
        html = render_page(schema)
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def do_POST(self):
        model, schema = load_artifacts()
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        form_data = parse_qs(body)

        user_input = {}
        for column in schema["numeric_columns"]:
            user_input[column] = float(form_data.get(column, [schema["numeric_defaults"][column]])[0])

        for column in schema["categorical_columns"]:
            options = schema["categorical_options"].get(column, [])
            default_value = options[0] if options else "Unknown"
            user_input[column] = form_data.get(column, [default_value])[0]

        input_df = pd.DataFrame([user_input], columns=schema["feature_columns"])
        probability = float(model.predict_proba(input_df)[0][1])
        score = probability_to_score(probability)
        band = probability_to_risk_band(probability)
        decision = probability_to_decision(probability)

        result_html = f"""
        <section class="result">
            <h2>Credit Risk Output</h2>
            <div class="metrics">
                <div class="metric">Default Probability<strong>{probability:.2%}</strong></div>
                <div class="metric">Risk Score<strong>{score}</strong></div>
                <div class="metric">Risk Band<strong>{band}</strong></div>
                <div class="metric">Decision<strong>{decision}</strong></div>
            </div>
        </section>
        """

        html = render_page(schema, result_html, user_input)
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), CreditRiskHandler)
    print(f"Open http://{HOST}:{PORT}")
    server.serve_forever()
