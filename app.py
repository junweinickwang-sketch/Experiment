import os
import time
import csv
import openai
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, make_response
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Load all HTML pages and their content
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
    prompt = f"""You are an AI assistant that summarizes search results.
Given the following articles, summarize them into a concise and objective paragraph.

{pages_text}

Summary:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
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
    results = [(webpages[i][0], scores[i]) for i in top_indices]
    selected_texts = "\n\n".join([page_contents[i] for i in top_indices])
    overview = get_overview_with_gpt(selected_texts)
    return render_template("results.html", results=results, overview=overview, uid=uid, query=query)

@app.route("/log_click", methods=["POST"])
def log_click():
    data = request.get_json()
    os.makedirs("logs", exist_ok=True)
    with open("logs/click_log.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([data["uid"], data["target"], data["timestamp"]])
    return "", 204

@app.route("/log_stay", methods=["POST"])
def log_stay():
    data = request.get_json()
    os.makedirs("logs", exist_ok=True)
    with open("logs/stay_log.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([data["uid"], data["page"], data["duration"]])
    return "", 204

# Entry point for Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
