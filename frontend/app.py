# frontend/app.py
from flask import Flask, render_template, request, jsonify
import requests, os

app = Flask(__name__, template_folder='templates', static_folder='static')
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/pred", methods=["POST"])
def pred():
    try:
        data = request.json
        r = requests.post(API_URL, json=data, timeout=10)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
