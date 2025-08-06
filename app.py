import os
import csv
import numpy as np
from flask import Flask, request, render_template, make_response, jsonify
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from bs4 import BeautifulSoup

# ---- Gemini API KEY ----
genai.configure(api_key="AIzaSyBSdiz1d5uUU_I-dHJK9LsiODGySnSE6Kk")
model = genai.GenerativeModel("gemini-1.5-flash")

# ---- Initialization ----
app = Flask(__name__, static_folder='webpages')
WEBPAGE_DIR = "webpages"
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Function to extract a preview from HTML content ----
def get_preview_from_html(html_content, length=200):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.body.get_text(separator=' ', strip=True)
    return text[:length] + '...' if len(text) > length else text

# ---- Load webpage content ----
webpages, page_contents, page_previews, page_titles = [], [], [], []
page_embeddings = None
if os.path.exists(WEBPAGE_DIR) and os.listdir(WEBPAGE_DIR):
    for filename in os.listdir(WEBPAGE_DIR):
        if filename.endswith(".html"):
            path = os.path.join(WEBPAGE_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            soup = BeautifulSoup(content, 'html.parser')
            title = soup.title.string.strip() if soup.title else filename
            preview = get_preview_from_html(content)
            webpages.append(filename)
            page_contents.append(content)
            page_previews.append(preview)
            page_titles.append(title)
    if page_contents:
        page_embeddings = model_embed.encode(page_contents, convert_to_tensor=True)

# ---- Gemini Summary Function ----
def get_overview_with_gemini(pages_text):
    prompt = f"""You are an AI assistant that summarizes search results.
Given the following articles, summarize them into a concise and objective paragraph.

{pages_text}

Summary:"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating Gemini overview: {e}")
        return "Sorry, I couldn't generate a summary for these results."

# ---- Routes ----
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
    
    if not page_embeddings is None:
        query_embedding = model_embed.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, page_embeddings)[0].cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:10]
        
        results = [
            {
                "filename": webpages[i],
                "preview": page_previews[i],
                "title": page_titles[i]
            }
            for i in top_indices
        ]
        
        selected_texts = "\n\n".join([page_contents[i] for i in top_indices])
        overview = get_overview_with_gemini(selected_texts)
    else:
        results = []
        overview = "No webpages found to search."
        
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

# ---- Start ----
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)))
