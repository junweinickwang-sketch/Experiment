import os
import time
import csv
import numpy as np
from flask import Flask, request, render_template, make_response
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client (new SDK)
client = OpenAI()

app = Flask(__name__)

# Load HTML pages and their content
WEBPAGE_DIR = "webpages"
webpages = []
page_contents = []

for filename in os.listdir(WEBPAGE_DIR):
    if filename.endswith(".html"):
        path = os.path.join(WEBPAGE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            webpages.append((filename, content))
            page_contents.append(content)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
page_embeddings = model.encode(page_contents, convert_to_tensor=True)

def get_overview_with_gpt(pages_text):
    messages = [
        {"role": "system", "content": "You are an AI assistant that summarizes search results."},
        {"role": "user", "content": f"Please provide a short, objective summary of the following content:\n{pages_text}"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

@app.route("/")
def home():
    uid = request.args.get("uid", "anonymous")
    resp = make_response(render_template("search.html", uid=uid))
    resp.set_cookie("uid", uid)
    return resp

@app.route("/search", methods=["POST"])
def search():
    query = request.form["query"]
    uid = request.form["uid"]
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, page_embeddings)[0].cpu().numpy()
    top_indices = np.argsort(scores)[::-1][:10]
    results = [(webpages[i][0], float(scores[i])) for i in top_indices]
    selected_texts = "\n\n".join([page_contents[i] for i in top_indices])
    overview = get_overview_with_gpt(selected_texts)

    log_event("search_log.csv", [uid, query, time.time()])

    return render_template("results.html", results=results, overview=overview, uid=uid, query=query)

@app.route("/log_click", methods=["POST"])
def log_click():
    data = request.get_json()
    log_event("click_log.csv", [data["uid"], data["target"], data["timestamp"]])
    return "", 204

@app.route("/log_stay", methods=["POST"])
def log_stay():
    data = request.get_json()
    log_event("stay_log.csv", [data["uid"], data["page"], data["duration"]])
    return "", 204

def log_event(filename, row):
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", filename)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
