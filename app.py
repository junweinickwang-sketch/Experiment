import os
import csv
import time
import numpy as np
from flask import Flask, request, render_template, make_response, jsonify
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# ---- Gemini API KEY ----
genai.configure(api_key="AIzaSyBSdiz1d5uUU_I-dHJK9LsiODGySnSE6Kk")  # 替换成你自己的 key
model = genai.GenerativeModel("gemini-pro")

# ---- 初始化 ----
app = Flask(__name__)
WEBPAGE_DIR = "webpages"
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# ---- 加载网页内容 ----
webpages, page_contents = [], []
for filename in os.listdir(WEBPAGE_DIR):
    if filename.endswith(".html"):
        path = os.path.join(WEBPAGE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            webpages.append((filename, content))
            page_contents.append(content)
page_embeddings = model_embed.encode(page_contents, convert_to_tensor=True)

# ---- Gemini 摘要函数 ----
def get_overview_with_gpt(pages_text):
    prompt = f"""You are an AI assistant that summarizes search results.
Given the following articles, summarize them into a concise and objective paragraph.

{pages_text}

Summary:"""
    response = model.generate_content(prompt)
    return response.text.strip()

# ---- 路由 ----
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
    query_embedding = model_embed.encode(query, convert_to_tensor=True)
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

# ---- 启动 ----
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
