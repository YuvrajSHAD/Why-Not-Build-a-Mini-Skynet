from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import subprocess
import json
from langchain_ollama import OllamaLLM  # Import the LLM here

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
GRAPH_FOLDER = "graphs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# Initialize the LLM with the desired model
llm = OllamaLLM(model="llama3.1")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Run the model script
    result = subprocess.run(["python", "model.py", file_path], capture_output=True, text=True)

    if result.returncode != 0:
        return jsonify({"error": "Model execution failed", "details": result.stderr}), 500

    # Load model results from output.json
    try:
        with open("output.json", "r") as f:
            model_results = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "Output file not found"}), 500
    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding JSON from output file"}), 500

    # Prepare the context for chat
    context = {
        "dataset": file.filename,
        "output": model_results
    }

    # Store context in a global variable or similar approach
    app.config['CHAT_CONTEXT'] = context

    # Extract relevant results for response
    response_results = {
        "ANN Accuracy": model_results.get("ANN Accuracy", "N/A"),
        "Hybrid Model Accuracy": model_results.get("Hybrid Model Accuracy", "N/A"),
        "XGBoost Accuracy": model_results.get("XGBoost Accuracy", "N/A"),
        "Classification Report": model_results.get("Classification Report", {}),
        "Confusion Matrix": model_results.get("Confusion Matrix", [])
    }

    return jsonify({
        "success": True,
        "results": response_results,
        "graphs": [
            "churn_distribution.png",
            "feature_correlation_heatmap.png",
            "xgboost_feature_importance.png",
            "confusion_matrix.png"
        ]
    })

@app.route("/get_graph/<filename>")
def get_graph(filename):
    return send_from_directory(GRAPH_FOLDER, filename)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    context = app.config.get('CHAT_CONTEXT', {})

    # Prepare the message for the model
    full_message = f"{user_input}\n\nContext: {json.dumps(context)}"  # Include context

    try:
        # Send the message to the model
        response = llm.invoke(full_message)
        return jsonify({"reply": response})
    except Exception as e:
        return jsonify({"reply": f"Error communicating with the model: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=False)
