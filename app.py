import os
import csv
import time
import json
import uuid
import threading
from datetime import datetime
import numpy as np

# 解决 OpenMP 冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flask import Flask, request, render_template_string, make_response, jsonify
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# ---- Gemini API KEY ----
genai.configure(api_key="AIzaSyBSdiz1d5uUU_I-dHJK9LsiODGySnSE6Kk")  # 替换成你自己的 key
model = genai.GenerativeModel("gemini-1.5-flash")

# ---- 初始化 ----
app = Flask(__name__)
WEBPAGE_DIR = "webpages"
USER_DATA_FILE = "user_data.json"
LOGS_DIR = "logs"

# 创建必要的目录
os.makedirs(LOGS_DIR, exist_ok=True)

# 线程锁，确保文件操作的安全性
file_locks = {
    'users': threading.Lock(),
    'clicks': threading.Lock(),
    'stays': threading.Lock()
}

model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# ---- 用户管理 ----
def load_users():
    with file_locks['users']:
        if os.path.exists(USER_DATA_FILE):
            try:
                with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}

def save_users(users):
    with file_locks['users']:
        try:
            with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(users, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存用户数据失败: {e}")

def get_or_create_user(request):
    uid = request.cookies.get("uid")
    users = load_users()
    
    if not uid or uid not in users:
        uid = str(uuid.uuid4())
        username = f"用户{len(users) + 1}"
        users[uid] = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "search_count": 0
        }
        save_users(users)
    else:
        users[uid]["last_active"] = datetime.now().isoformat()
        save_users(users)
    
    return uid, users[uid]

# ---- 加载网页内容 ----
webpages, page_contents = [], []
webpage_titles = {}

for filename in os.listdir(WEBPAGE_DIR):
    if filename.endswith(".html"):
        path = os.path.join(WEBPAGE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            webpages.append((filename, content))
            page_contents.append(content)
            
            # 提取标题
            title_start = content.find("<title>")
            title_end = content.find("</title>")
            if title_start != -1 and title_end != -1:
                title = content[title_start + 7:title_end]
                webpage_titles[filename] = title
            else:
                webpage_titles[filename] = filename.replace(".html", "").replace("_", " ").title()

page_embeddings = model_embed.encode(page_contents, convert_to_tensor=True)

# ---- Gemini 摘要函数（带引用）----
def get_overview_with_citations(pages_info, query):
    """生成带引用的摘要"""
    pages_text = ""
    for i, (filename, content) in enumerate(pages_info):
        title = webpage_titles.get(filename, filename)
        # 截取内容的前1000个字符作为摘要用
        preview = content[:1000] + "..." if len(content) > 1000 else content
        pages_text += f"\n[{i+1}] 来源：{title} (文件：{filename})\n内容：{preview}\n"
    
    prompt = f"""你是一个AI助手，需要根据以下搜索结果为用户查询"{query}"生成一个简洁客观的摘要。

重要要求：
1. 生成一个3-5句话的摘要，客观描述搜索结果的主要内容
2. 在摘要中适当位置插入引用标记，格式为[数字]，对应下面的来源编号
3. 确保每个重要事实都有引用支撑
4. 引用应该自然地融入文本中

搜索结果：
{pages_text}

请生成带引用的摘要："""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"摘要生成失败: {str(e)}"

# ---- 日志记录 ----
def log_click(uid, username, target, query, score):
    with file_locks['clicks']:
        log_entry = {
            "uid": uid,
            "username": username,
            "target": target,
            "query": query,
            "score": score,
            "timestamp": datetime.now().isoformat()
        }
        
        log_file = os.path.join(LOGS_DIR, "clicks.json")
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logs = []
        
        logs.append(log_entry)
        
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存点击日志失败: {e}")

def log_stay(uid, username, page, duration, query=""):
    with file_locks['stays']:
        log_entry = {
            "uid": uid,
            "username": username,
            "page": page,
            "duration": duration,
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        
        log_file = os.path.join(LOGS_DIR, "stays.json")
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logs = []
        
        logs.append(log_entry)
        
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存停留日志失败: {e}")

def log_scroll(uid, username, page, event_type, scroll_percentage, total_scroll_events, time_to_reach, webpage_filename=""):
    """记录滚动事件"""
    with file_locks['stays']:  # 复用stays的锁
        log_entry = {
            "uid": uid,
            "username": username,
            "page": page,
            "event_type": event_type,
            "scroll_percentage": scroll_percentage,
            "total_scroll_events": total_scroll_events,
            "time_to_reach": time_to_reach,
            "webpage_filename": webpage_filename,
            "timestamp": datetime.now().isoformat()
        }
        
        log_file = os.path.join(LOGS_DIR, "scrolls.json")
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logs = []
        
        logs.append(log_entry)
        
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存滚动日志失败: {e}")

def load_logs():
    logs = {"clicks": [], "stays": [], "scrolls": []}
    
    # 加载点击日志
    with file_locks['clicks']:
        click_file = os.path.join(LOGS_DIR, "clicks.json")
        if os.path.exists(click_file):
            try:
                with open(click_file, "r", encoding="utf-8") as f:
                    logs["clicks"] = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logs["clicks"] = []
    
    # 加载停留日志
    with file_locks['stays']:
        stay_file = os.path.join(LOGS_DIR, "stays.json")
        if os.path.exists(stay_file):
            try:
                with open(stay_file, "r", encoding="utf-8") as f:
                    logs["stays"] = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logs["stays"] = []
    
    # 加载滚动日志
    with file_locks['stays']:  # 复用stays的锁
        scroll_file = os.path.join(LOGS_DIR, "scrolls.json")
        if os.path.exists(scroll_file):
            try:
                with open(scroll_file, "r", encoding="utf-8") as f:
                    logs["scrolls"] = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logs["scrolls"] = []
    
    return logs

# ---- 模板加载 ----
def load_template(template_name):
    template_path = os.path.join("templates", template_name)
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# ---- 路由 ----
@app.route("/")
def home():
    uid, user = get_or_create_user(request)
    template = load_template("search.html")
    resp = make_response(render_template_string(template, uid=uid, user=user))
    resp.set_cookie("uid", uid, max_age=30*24*60*60)  # 30天过期
    return resp

@app.route("/search", methods=["POST"])
def search():
    query = request.form["query"]
    uid, user = get_or_create_user(request)
    
    print(f"搜索请求: 用户={user['username']} (UID={uid[:8]}...), 查询={query}")
    
    # 更新搜索次数
    users = load_users()
    users[uid]["search_count"] += 1
    save_users(users)
    user = users[uid]
    
    # 执行搜索
    query_embedding = model_embed.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, page_embeddings)[0].cpu().numpy()
    top_indices = np.argsort(scores)[::-1][:10]
    
    # 准备结果
    results = []
    pages_for_overview = []
    
    for i in top_indices:
        filename, content = webpages[i]
        score = float(scores[i])
        title = webpage_titles.get(filename, filename)
        
        # 生成预览
        preview = content[:200] + "..." if len(content) > 200 else content
        preview = preview.replace("<", "&lt;").replace(">", "&gt;")
        
        results.append({
            "filename": filename,
            "title": title,
            "preview": preview,
            "score": score
        })
        
        pages_for_overview.append((filename, content))
    
    # 生成摘要（只使用前5个结果）
    overview = ""
    if pages_for_overview:
        overview_raw = get_overview_with_citations(pages_for_overview[:5], query)
        # 处理引用链接
        overview = process_citations(overview_raw, pages_for_overview[:5])
    
    template = load_template("results.html")
    resp = make_response(render_template_string(template, 
                                              results=results, 
                                              overview=overview, 
                                              uid=uid, 
                                              user=user,
                                              query=query))
    
    # 确保cookie正确设置
    resp.set_cookie("uid", uid, max_age=30*24*60*60)
    return resp

def process_citations(text, pages_info):
    """处理引用，将[数字]转换为链接"""
    import re
    
    def replace_citation(match):
        citations = match.group(1)  # 获取括号内的内容，如 "2, 4, 5"
        
        # 分割多个引用
        citation_numbers = [num.strip() for num in citations.split(',')]
        citation_links = []
        
        for num in citation_numbers:
            try:
                index = int(num) - 1
                if 0 <= index < len(pages_info):
                    filename = pages_info[index][0]
                    citation_links.append(f'<a href="/webpages/{filename}" target="_blank" class="citation-link">[{num}]</a>')
                else:
                    citation_links.append(f'[{num}]')
            except ValueError:
                citation_links.append(f'[{num}]')
        
        return f'<span class="citation">{"".join(citation_links)}</span>'
    
    # 匹配 [数字] 或 [数字, 数字, 数字] 格式
    return re.sub(r'\[([0-9, ]+)\]', replace_citation, text)

@app.route("/webpages/<filename>")
def serve_webpage(filename):
    # 记录页面访问
    uid, user = get_or_create_user(request)
    
    path = os.path.join(WEBPAGE_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 确保文件名安全
        safe_filename = filename.replace('"', '\\"').replace("'", "\\'")
        
        # 创建更强大的追踪脚本，包含滚动跟踪
        tracking_script = f'''
<!-- 停留时间和滚动跟踪脚本 -->
<script>
(function() {{
    const uid = "{uid}";
    const filename = "{safe_filename}";
    let startTime = Date.now();
    let totalVisibleTime = 0;
    let lastVisibleStart = Date.now();
    let isPageVisible = !document.hidden;
    let hasLoggedStay = false;
    
    // 滚动跟踪变量
    let maxScrollPercentage = 0;
    let scrollMilestones = {{5: false, 10: false, 25: false, 50: false, 75: false, 90: false, 100: false}};
    let lastScrollTime = Date.now();
    let totalScrollEvents = 0;
    let scrollStartTime = null;
    let totalScrollTime = 0;

    function calculateScrollPercentage() {{
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        if (documentHeight <= windowHeight) {{
            return 100; // 如果页面内容不足一屏，算作100%
        }}
        
        const maxScroll = documentHeight - windowHeight;
        const percentage = Math.round((scrollTop / maxScroll) * 100);
        return Math.min(100, Math.max(0, percentage));
    }}

    function logScrollMilestone(percentage) {{
        const data = {{
            uid: uid,
            page: "/webpages/" + filename,
            event_type: "scroll_milestone",
            scroll_percentage: percentage,
            timestamp: new Date().toISOString(),
            webpage_filename: filename,
            total_scroll_events: totalScrollEvents,
            time_to_reach: Date.now() - startTime
        }};

        console.log("记录滚动里程碑:", percentage + "%");

        fetch("/log_scroll", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify(data),
            keepalive: true
        }}).catch(error => {{
            console.error("记录滚动里程碑失败:", error);
        }});
    }}

    function handleScroll() {{
        if (!isPageVisible) return; // 只在页面可见时记录滚动
        
        const currentPercentage = calculateScrollPercentage();
        totalScrollEvents++;
        lastScrollTime = Date.now();
        
        // 开始滚动计时
        if (scrollStartTime === null) {{
            scrollStartTime = Date.now();
        }}
        
        // 更新最大滚动百分比
        if (currentPercentage > maxScrollPercentage) {{
            maxScrollPercentage = currentPercentage;
        }}
        
        // 检查是否达到新的里程碑
        for (let milestone in scrollMilestones) {{
            const milestoneNum = parseInt(milestone);
            if (currentPercentage >= milestoneNum && !scrollMilestones[milestone]) {{
                scrollMilestones[milestone] = true;
                logScrollMilestone(milestoneNum);
            }}
        }}
    }}

    // 防抖滚动事件
    let scrollTimeout;
    function debouncedScrollHandler() {{
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(() => {{
            handleScroll();
            // 更新总滚动时间
            if (scrollStartTime) {{
                totalScrollTime = Date.now() - scrollStartTime;
            }}
        }}, 100);
    }}

    function updateVisibleTime() {{
        if (isPageVisible) {{
            const now = Date.now();
            totalVisibleTime += (now - lastVisibleStart);
            lastVisibleStart = now;
        }}
    }}

    function logWebpageStay() {{
        if (hasLoggedStay) return;
        
        updateVisibleTime();
        
        if (totalVisibleTime < 1000) return; // 只记录可见时间超过1秒的
        
        hasLoggedStay = true;
        
        const data = {{
            uid: uid,
            page: "/webpages/" + filename,
            duration: totalVisibleTime,
            query: "webpage_view",
            timestamp: new Date().toISOString(),
            webpage_filename: filename,
            // 滚动数据
            max_scroll_percentage: maxScrollPercentage,
            reached_milestones: Object.keys(scrollMilestones).filter(m => scrollMilestones[m]).map(Number),
            total_scroll_events: totalScrollEvents,
            total_scroll_time: totalScrollTime
        }};

        console.log("记录网页停留（含滚动数据）:", data);

        if (navigator.sendBeacon) {{
            const blob = new Blob([JSON.stringify(data)], {{type: 'application/json'}});
            const result = navigator.sendBeacon("/log_stay", blob);
            console.log("sendBeacon result:", result);
        }} else {{
            fetch("/log_stay", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify(data),
                keepalive: true
            }}).then(response => {{
                console.log("fetch response:", response.status);
            }}).catch(error => {{
                console.error("fetch error:", error);
            }});
        }}
    }}

    // 页面可见性变化
    document.addEventListener("visibilitychange", function() {{
        const now = Date.now();
        
        if (document.hidden) {{
            if (isPageVisible) {{
                totalVisibleTime += (now - lastVisibleStart);
                isPageVisible = false;
                console.log("页面隐藏，当前总可见时间:", totalVisibleTime/1000, "秒");
            }}
        }} else {{
            if (!isPageVisible) {{
                lastVisibleStart = now;
                isPageVisible = true;
                console.log("页面显示，重新开始计时");
                // 重新开始滚动计时
                scrollStartTime = Date.now();
            }}
        }}
    }});

    // 窗口焦点变化
    window.addEventListener("focus", function() {{
        if (!isPageVisible && !document.hidden) {{
            lastVisibleStart = Date.now();
            isPageVisible = true;
            console.log("窗口获得焦点");
            scrollStartTime = Date.now();
        }}
    }});

    window.addEventListener("blur", function() {{
        if (isPageVisible) {{
            totalVisibleTime += (Date.now() - lastVisibleStart);
            isPageVisible = false;
            console.log("窗口失去焦点，当前总可见时间:", totalVisibleTime/1000, "秒");
        }}
    }});

    // 滚动事件监听
    window.addEventListener("scroll", debouncedScrollHandler, {{ passive: true }});

    // 页面卸载时记录停留时间
    let isUnloading = false;
    
    function handleUnload() {{
        if (!isUnloading) {{
            isUnloading = true;
            updateVisibleTime();
            console.log("页面卸载，最终可见时间:", totalVisibleTime/1000, "秒, 最大滚动:", maxScrollPercentage + "%");
            logWebpageStay();
        }}
    }}
    
    window.addEventListener("beforeunload", handleUnload);
    window.addEventListener("pagehide", handleUnload);

    // 页面加载完成后初始化
    function initTracking() {{
        startTime = Date.now();
        lastVisibleStart = Date.now();
        isPageVisible = !document.hidden;
        scrollStartTime = Date.now();
        
        console.log("网页追踪初始化完成:", filename);
        
        // 初始滚动位置检查
        setTimeout(() => {{
            const initialScroll = calculateScrollPercentage();
            if (initialScroll > 0) {{
                handleScroll();
            }}
        }}, 500);
        
        // 测试连接
        fetch("/log_stay", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{
                uid: uid,
                page: "/webpages/" + filename,
                duration: 100,
                query: "test_connection",
                timestamp: new Date().toISOString(),
                webpage_filename: filename
            }})
        }}).then(response => {{
            console.log("连接测试成功:", response.status);
        }}).catch(error => {{
            console.error("连接测试失败:", error);
        }});
    }}

    // 确保DOM加载完成后初始化
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', initTracking);
    }} else {{
        initTracking();
    }}

    // 添加调试信息显示（如果URL包含debug参数）
    if (window.location.search.includes('debug=1')) {{
        const debugDiv = document.createElement('div');
        debugDiv.style.cssText = `
            position: fixed; top: 10px; right: 10px; 
            background: rgba(0,0,0,0.9); color: white; 
            padding: 15px; border-radius: 8px; font-size: 12px;
            z-index: 9999; font-family: monospace; min-width: 250px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        `;
        document.body.appendChild(debugDiv);
        
        setInterval(() => {{
            updateVisibleTime();
            const currentScroll = calculateScrollPercentage();
            const reachedMilestones = Object.keys(scrollMilestones).filter(m => scrollMilestones[m]).join(', ');
            
            debugDiv.innerHTML = `
                📄 文件: ${{filename}}<br>
                👁️ 状态: ${{isPageVisible ? '可见' : '隐藏'}}<br>
                ⏱️ 可见时间: ${{(totalVisibleTime/1000).toFixed(1)}}秒<br>
                📊 当前滚动: ${{currentScroll}}%<br>
                🏆 最大滚动: ${{maxScrollPercentage}}%<br>
                🎯 达成里程碑: ${{reachedMilestones || '无'}}<br>
                🖱️ 滚动次数: ${{totalScrollEvents}}<br>
                🔄 已记录: ${{hasLoggedStay ? '是' : '否'}}
            `;
        }}, 100);
    }}
}})();
</script>
'''
        
        # 尝试多种方式插入脚本
        script_inserted = False
        
        # 方法1: 在</head>前插入
        if "</head>" in content:
            content = content.replace("</head>", tracking_script + "\n</head>")
            script_inserted = True
        # 方法2: 在</body>前插入
        elif "</body>" in content:
            content = content.replace("</body>", tracking_script + "\n</body>")
            script_inserted = True
        # 方法3: 在</html>前插入
        elif "</html>" in content:
            content = content.replace("</html>", tracking_script + "\n</html>")
            script_inserted = True
        # 方法4: 直接追加到末尾
        else:
            content += tracking_script
            script_inserted = True
        
        print(f"🔧 为 {filename} 注入追踪脚本: {'成功' if script_inserted else '失败'}")
        
        # 设置cookie并返回
        resp = make_response(content)
        resp.set_cookie("uid", uid, max_age=30*24*60*60)
        resp.headers['Content-Type'] = 'text/html; charset=utf-8'
        return resp
    
    return "页面不存在", 404

@app.route("/log_click", methods=["POST"])
def log_click_endpoint():
    data = request.get_json()
    uid = data.get("uid")
    
    # 确保用户存在
    if not uid:
        return "", 400
    
    users = load_users()
    if uid not in users:
        # 如果用户不存在，创建一个新用户
        username = f"用户{len(users) + 1}"
        users[uid] = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "search_count": 0
        }
        save_users(users)
    
    username = users[uid]["username"]
    
    log_click(
        uid=uid,
        username=username,
        target=data.get("target"),
        query=data.get("query", ""),
        score=data.get("score", 0)
    )
    
    print(f"记录点击日志: 用户={username}, 目标={data.get('target')}, 查询={data.get('query')}")
    return "", 204

@app.route("/log_stay", methods=["POST"])
def log_stay_endpoint():
    # 支持两种数据格式：JSON和form data
    if request.is_json:
        data = request.get_json()
    else:
        # sendBeacon 可能发送为 blob，需要解析
        try:
            data = json.loads(request.data.decode('utf-8'))
        except:
            # 如果解析失败，从form data获取
            data = {
                "uid": request.form.get("uid"),
                "page": request.form.get("page"),
                "duration": request.form.get("duration"),
                "query": request.form.get("query", "")
            }
    
    uid = data.get("uid")
    
    # 确保用户存在
    if not uid:
        return "", 400
    
    users = load_users()
    if uid not in users:
        # 如果用户不存在，创建一个新用户
        username = f"用户{len(users) + 1}"
        users[uid] = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "search_count": 0
        }
        save_users(users)
    
    username = users[uid]["username"]
    duration = data.get("duration", 0)
    page = data.get("page", "")
    query = data.get("query", "")
    
    print(f"🔍 收到停留日志请求: 用户={username}, 页面={page}, 时长={duration}ms, 查询={query}")
    
    # 区分测试连接和真实记录
    if query == "test_connection":
        print(f"🔗 连接测试成功: {filename if 'webpage_filename' in data else '未知页面'}")
        return "", 204
    
    # 只记录停留时间大于1秒的记录，避免噪音和重复
    try:
        duration_int = int(duration)
        if duration_int > 1000:  # 1秒以上
            log_stay(
                uid=uid,
                username=username,
                page=page,
                duration=duration,
                query=query
            )
            
            if page.startswith("/webpages/"):
                webpage_filename = data.get("webpage_filename", "未知")
                print(f"✅ 记录网页停留日志: 用户={username}, 网页={webpage_filename}, 可见时长={duration_int/1000:.1f}秒")
            else:
                print(f"✅ 记录页面停留日志: 用户={username}, 页面={page}, 可见时长={duration_int/1000:.1f}秒")
        else:
            print(f"⏭️  忽略短暂停留: 用户={username}, 页面={page}, 时长={duration_int/1000:.1f}秒")
    except (ValueError, TypeError):
        print(f"❌ 无效的停留时间数据: {duration}")
    
    return "", 204

@app.route("/log_scroll", methods=["POST"])
def log_scroll_endpoint():
    """处理滚动里程碑记录"""
    if request.is_json:
        data = request.get_json()
    else:
        try:
            data = json.loads(request.data.decode('utf-8'))
        except:
            return "", 400
    
    uid = data.get("uid")
    
    # 确保用户存在
    if not uid:
        return "", 400
    
    users = load_users()
    if uid not in users:
        username = f"用户{len(users) + 1}"
        users[uid] = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "search_count": 0
        }
        save_users(users)
    
    username = users[uid]["username"]
    
    log_scroll(
        uid=uid,
        username=username,
        page=data.get("page", ""),
        event_type=data.get("event_type", "scroll_milestone"),
        scroll_percentage=data.get("scroll_percentage", 0),
        total_scroll_events=data.get("total_scroll_events", 0),
        time_to_reach=data.get("time_to_reach", 0),
        webpage_filename=data.get("webpage_filename", "")
    )
    
    webpage_filename = data.get("webpage_filename", "未知")
    scroll_percentage = data.get("scroll_percentage", 0)
    print(f"📊 记录滚动里程碑: 用户={username}, 网页={webpage_filename}, 滚动到={scroll_percentage}%")
    
    return "", 204
    uid, user = get_or_create_user(request)
    users_data = load_users()
    
    template = load_template("user.html")
    return render_template_string(template, users=users_data)

@app.route("/logs")
def logs():
    uid, user = get_or_create_user(request)
    logs_data = load_logs()
    
    # 分类日志数据
    categorized_logs = {
        "clicks": logs_data["clicks"],
        "scrolls": logs_data["scrolls"],
        "stays": {
            "search_pages": [],      # 搜索相关页面 (/, /search)
            "system_pages": [],      # 系统页面 (/logs, /users, /test, /debug)
            "content_pages": []      # 内容页面 (/webpages/*)
        }
    }
    
    # 按页面类型分类停留日志
    for stay in logs_data["stays"]:
        page = stay.get("page", "")
        if page.startswith("/webpages/"):
            categorized_logs["stays"]["content_pages"].append(stay)
        elif page in ["/", "/search"]:
            categorized_logs["stays"]["search_pages"].append(stay)
        else:
            categorized_logs["stays"]["system_pages"].append(stay)
    
    # 计算统计数据
    stats = {
        "total_clicks": len(logs_data["clicks"]),
        "total_stays": len(logs_data["stays"]),
        "total_scrolls": len(logs_data["scrolls"]),
        "content_page_views": len(categorized_logs["stays"]["content_pages"]),
        "unique_users_clicks": len(set(click.get("username", "") for click in logs_data["clicks"])),
        "avg_stay_time": 0,
        "avg_content_stay_time": 0
    }
    
    if logs_data["stays"]:
        total_duration = sum(int(stay.get("duration", 0)) for stay in logs_data["stays"])
        stats["avg_stay_time"] = round(total_duration / len(logs_data["stays"]) / 1000, 1)
    
    if categorized_logs["stays"]["content_pages"]:
        content_duration = sum(int(stay.get("duration", 0)) for stay in categorized_logs["stays"]["content_pages"])
        stats["avg_content_stay_time"] = round(content_duration / len(categorized_logs["stays"]["content_pages"]) / 1000, 1)
    
    # 只取最近的记录
    recent_logs = {
        "clicks": logs_data["clicks"][-50:],
        "scrolls": logs_data["scrolls"][-30:],  # 滚动日志
        "stays": {
            "search_pages": categorized_logs["stays"]["search_pages"][-20:],
            "system_pages": categorized_logs["stays"]["system_pages"][-20:],
            "content_pages": categorized_logs["stays"]["content_pages"][-30:]
        }
    }
    
    template = load_template("logs.html")
    return render_template_string(template, logs=recent_logs, stats=stats)

@app.route("/webpage_stats")
def webpage_stats():
    """网页访问统计页面"""
    uid, user = get_or_create_user(request)
    logs_data = load_logs()
    
    # 统计每个网页的访问情况
    webpage_stats = {}
    
    # 处理停留日志
    for stay in logs_data["stays"]:
        page = stay.get("page", "")
        if page.startswith("/webpages/"):
            filename = page.split("/")[-1]
            if filename not in webpage_stats:
                webpage_stats[filename] = {
                    "filename": filename,
                    "title": webpage_titles.get(filename, filename),
                    "total_visits": 0,
                    "total_time": 0,
                    "unique_users": set(),
                    "avg_time": 0,
                    "max_scroll_avg": 0,
                    "scroll_data": [],
                    "milestone_stats": {25: 0, 50: 0, 75: 0, 90: 0, 100: 0}
                }
            
            webpage_stats[filename]["total_visits"] += 1
            webpage_stats[filename]["total_time"] += int(stay.get("duration", 0))
            webpage_stats[filename]["unique_users"].add(stay.get("username", ""))
            
            # 滚动数据
            max_scroll = stay.get("max_scroll_percentage", 0)
            if max_scroll > 0:
                webpage_stats[filename]["scroll_data"].append(max_scroll)
            
            # 里程碑统计
            milestones = stay.get("reached_milestones", [])
            for milestone in milestones:
                if milestone in webpage_stats[filename]["milestone_stats"]:
                    webpage_stats[filename]["milestone_stats"][milestone] += 1
    
    # 处理滚动日志
    for scroll in logs_data["scrolls"]:
        webpage_filename = scroll.get("webpage_filename", "")
        if webpage_filename and webpage_filename in webpage_stats:
            # 滚动里程碑已经在停留日志中处理了，这里可以添加额外的滚动分析
            pass
    
    # 计算平均时间、滚动数据等
    for filename, stats in webpage_stats.items():
        if stats["total_visits"] > 0:
            stats["avg_time"] = round(stats["total_time"] / stats["total_visits"] / 1000, 1)
        
        if stats["scroll_data"]:
            stats["max_scroll_avg"] = round(sum(stats["scroll_data"]) / len(stats["scroll_data"]), 1)
        
        stats["unique_users"] = len(stats["unique_users"])
        
        # 计算阅读完成率（滚动到90%以上的比例）
        stats["completion_rate"] = 0
        if stats["total_visits"] > 0:
            completed = stats["milestone_stats"].get(90, 0)
            stats["completion_rate"] = round((completed / stats["total_visits"]) * 100, 1)
    
    # 按总访问时间排序
    sorted_stats = sorted(webpage_stats.values(), key=lambda x: x["total_time"], reverse=True)
    
    stats_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>网页访问统计</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }}
            .header {{ background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }}
            .back-link {{ color: #666; text-decoration: none; font-size: 14px; float: left; }}
            .back-link:hover {{ color: #4CAF50; }}
            .stats-container {{ background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .stats-header {{ background: #4CAF50; color: white; padding: 15px 20px; font-size: 18px; font-weight: bold; }}
            .stats-table {{ width: 100%; border-collapse: collapse; }}
            .stats-table th {{ background: #f0f0f0; padding: 12px; text-align: left; font-weight: bold; border-bottom: 2px solid #ddd; font-size: 13px; }}
            .stats-table td {{ padding: 10px 12px; border-bottom: 1px solid #eee; font-size: 14px; }}
            .stats-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .stats-table tr:hover {{ background-color: #f0f8ff; }}
            .filename {{ font-family: monospace; font-size: 12px; color: #666; }}
            .title {{ font-weight: bold; color: #333; }}
            .time-bar {{ background: #e0e0e0; height: 8px; border-radius: 4px; overflow: hidden; }}
            .time-fill {{ background: #4CAF50; height: 100%; }}
            .scroll-bar {{ background: #e0e0e0; height: 6px; border-radius: 3px; overflow: hidden; margin-top: 2px; }}
            .scroll-fill {{ background: #2196F3; height: 100%; }}
            .milestone-badges {{ display: flex; gap: 2px; flex-wrap: wrap; }}
            .milestone-badge {{ 
                background: #f0f0f0; color: #666; padding: 1px 4px; border-radius: 3px; 
                font-size: 10px; font-weight: bold; 
            }}
            .milestone-badge.reached {{ background: #4CAF50; color: white; }}
            .completion-rate {{ 
                padding: 2px 6px; border-radius: 8px; font-size: 11px; font-weight: bold;
                color: white;
            }}
            .no-data {{ text-align: center; padding: 40px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <a href="/logs" class="back-link">← 返回日志</a>
            <h1>网页访问统计</h1>
        </div>

        <div class="stats-container">
            <div class="stats-header">
                📊 内容页面访问统计 (共 {len(sorted_stats)} 个页面)
            </div>
            
            {generate_enhanced_stats_table(sorted_stats)}
        </div>

        <div style="text-align: center; margin-top: 20px;">
            <a href="/logs" style="color: #666; text-decoration: none;">← 返回日志</a>
            <span style="margin: 0 20px; color: #ccc;">|</span>
            <a href="/" style="color: #666; text-decoration: none;">首页</a>
        </div>
    </body>
    </html>
    """
    
    return stats_html

def generate_enhanced_stats_table(sorted_stats):
    if not sorted_stats:
        return '<div class="no-data"><h3>暂无网页访问数据</h3><p>用户访问内容页面时会显示统计信息</p></div>'
    
    max_time = max(stats["total_time"] for stats in sorted_stats) if sorted_stats else 1
    
    table_html = '''
    <table class="stats-table">
        <thead>
            <tr>
                <th>排名</th>
                <th>页面标题</th>
                <th>访问次数</th>
                <th>独立用户</th>
                <th>总停留时间</th>
                <th>平均停留</th>
                <th>平均滚动深度</th>
                <th>阅读完成率</th>
                <th>滚动里程碑</th>
                <th>热度</th>
            </tr>
        </thead>
        <tbody>
    '''
    
    for i, stats in enumerate(sorted_stats, 1):
        fill_width = (stats["total_time"] / max_time) * 100 if max_time > 0 else 0
        scroll_fill_width = stats["max_scroll_avg"] if stats["max_scroll_avg"] > 0 else 0
        
        # 完成率颜色
        completion_rate = stats["completion_rate"]
        if completion_rate >= 70:
            completion_color = "#4CAF50"
        elif completion_rate >= 40:
            completion_color = "#FF9800"
        else:
            completion_color = "#F44336"
        
        # 里程碑徽章
        milestone_badges = ""
        for milestone, count in stats["milestone_stats"].items():
            if milestone in [25, 50, 75, 90, 100]:
                badge_class = "reached" if count > 0 else ""
                milestone_badges += f'<span class="milestone-badge {badge_class}">{milestone}%({count})</span>'
        
        table_html += f'''
            <tr>
                <td>{i}</td>
                <td>
                    <div class="title">{stats["title"]}</div>
                    <div class="filename">{stats["filename"]}</div>
                    <a href="/webpages/{stats["filename"]}" target="_blank" style="color: #4CAF50; text-decoration: none; font-size: 12px;">查看页面</a>
                </td>
                <td>{stats["total_visits"]}</td>
                <td>{stats["unique_users"]}</td>
                <td>{round(stats["total_time"]/1000, 1)}秒</td>
                <td>{stats["avg_time"]}秒</td>
                <td>
                    {stats["max_scroll_avg"]}%
                    <div class="scroll-bar">
                        <div class="scroll-fill" style="width: {scroll_fill_width}%;"></div>
                    </div>
                </td>
                <td>
                    <span class="completion-rate" style="background: {completion_color};">
                        {completion_rate}%
                    </span>
                </td>
                <td>
                    <div class="milestone-badges">
                        {milestone_badges}
                    </div>
                </td>
                <td>
                    <div class="time-bar">
                        <div class="time-fill" style="width: {fill_width}%;"></div>
                    </div>
                </td>
            </tr>
        '''
    
    table_html += '''
        </tbody>
    </table>
    '''
    
    return table_html
    """测试页面，用于验证引用和日志功能"""
    uid, user = get_or_create_user(request)
    
    test_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>功能测试页面</title>
        <style>
            body {{ font-family: Arial; padding: 20px; line-height: 1.6; }}
            .citation {{ display: inline; margin-left: 2px; }}
            .citation-link {{ 
                color: #1a73e8; text-decoration: none; font-size: 13px; 
                padding: 2px 4px; background-color: rgba(26, 115, 232, 0.1); 
                border-radius: 3px; margin: 0 1px; 
            }}
            .citation-link:hover {{ background-color: rgba(26, 115, 232, 0.2); text-decoration: underline; }}
            .test-section {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>功能测试页面</h1>
        <p><strong>当前用户:</strong> {user['username']} (UID: {uid[:8]}...)</p>
        
        <div class="test-section">
            <h2>1. Citation链接测试</h2>
            <p>这是一个测试引用的段落 <span class="citation"><a href="/webpages/ai_in_games.html" target="_blank" class="citation-link">[1]</a><a href="/webpages/game_engines.html" target="_blank" class="citation-link">[2]</a><a href="/webpages/indie_games.html" target="_blank" class="citation-link">[3]</a></span>。</p>
            <p>另一个引用例子 <span class="citation"><a href="/webpages/mobile_vs_console.html" target="_blank" class="citation-link">[4]</a><a href="/webpages/vr_future.html" target="_blank" class="citation-link">[5]</a></span>。</p>
        </div>
        
        <div class="test-section">
            <h2>2. 点击测试</h2>
            <button onclick="testClick()">测试点击记录</button>
            <p id="click-result">点击上面按钮测试点击记录功能</p>
        </div>
        
        <div class="test-section">
            <h2>3. 真实停留时间测试</h2>
            <p><strong>页面状态:</strong> <span id="visibility-status">👁️ 可见</span></p>
            <p><strong>实际可见时间:</strong> <span id="visible-time">0</span> 秒</p>
            <p><strong>测试说明:</strong></p>
            <ul>
                <li>只有页面在前台可见时才计算停留时间</li>
                <li>切换标签页、最小化窗口时会暂停计时</li>
                <li>重新回到页面时继续计时</li>
                <li>只记录可见时间超过1秒的访问</li>
            </ul>
            <button onclick="manualLogStay()">手动记录可见停留时间</button>
            <p id="stay-result">可见停留时间会在页面关闭时自动记录</p>
        </div>
        
        <div class="test-section">
            <h2>4. 快速链接</h2>
            <a href="/">返回首页</a> | 
            <a href="/logs">查看日志</a> | 
            <a href="/debug">调试信息</a>
        </div>

        <script>
            const uid = "{uid}";
            let startTime = Date.now();
            let testClickCount = 0;
            let hasLoggedStay = false;

            function testClick() {{
                testClickCount++;
                fetch("/log_click", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{
                        uid: uid,
                        target: "test_button_" + testClickCount,
                        query: "test_query",
                        score: 0.999,
                        timestamp: new Date().toISOString()
                    }})
                }}).then(() => {{
                    document.getElementById('click-result').innerHTML = 
                        `✅ 点击记录已发送 (第 ${{testClickCount}} 次)`;
                }});
            }}

            function manualLogStay() {{
                if (hasLoggedStay) {{
                    document.getElementById('stay-result').innerHTML = 
                        `⚠️ 已经记录过停留时间，防止重复`;
                    return;
                }}
                
                updateVisibleTime(); // 更新到最新的可见时间
                
                if (totalVisibleTime < 1000) {{
                    document.getElementById('stay-result').innerHTML = 
                        `⏱️ 可见时间不足1秒，无法记录 (当前: ${{(totalVisibleTime/1000).toFixed(1)}}秒)`;
                    return;
                }}
                
                hasLoggedStay = true;
                
                fetch("/log_stay", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{
                        uid: uid,
                        page: "/test",
                        duration: totalVisibleTime,
                        query: "test_page",
                        timestamp: new Date().toISOString()
                    }})
                }}).then(() => {{
                    document.getElementById('stay-result').innerHTML = 
                        `✅ 可见停留时间已记录: ${{(totalVisibleTime/1000).toFixed(1)}} 秒`;
                }});
            }}

            // 页面离开时记录停留时间（只记录一次）
            let isUnloading = false;
            window.addEventListener("beforeunload", function() {{
                if (!isUnloading && !hasLoggedStay) {{
                    isUnloading = true;
                    updateVisibleTime(); // 确保获取最新的可见时间
                    if (totalVisibleTime > 1000) {{ // 只记录超过1秒的可见停留
                        navigator.sendBeacon("/log_stay", new Blob([JSON.stringify({{
                            uid: uid,
                            page: "/test",
                            duration: totalVisibleTime,
                            query: "test_page",
                            timestamp: new Date().toISOString()
                        }})], {{type: 'application/json'}}));
                    }}
                }}
            }});

            // 页面加载完成后初始化
            window.addEventListener("load", function() {{
                startTime = Date.now();
                lastVisibleStart = Date.now();
                isPageVisible = !document.hidden;
            }});
        </script>
    </body>
    </html>
    """
    
    resp = make_response(test_html)
    resp.set_cookie("uid", uid, max_age=30*24*60*60)
    return resp
    """调试页面，显示当前用户信息和系统状态"""
    uid, user = get_or_create_user(request)
    users = load_users()
    logs = load_logs()
    
    debug_info = {
        "current_uid": uid,
        "current_user": user,
        "all_users": users,
        "total_users": len(users),
        "total_clicks": len(logs["clicks"]),
        "total_stays": len(logs["stays"]),
        "recent_clicks": logs["clicks"][-5:] if logs["clicks"] else [],
        "recent_stays": logs["stays"][-5:] if logs["stays"] else []
    }
    
    return f"""
    <html>
    <head><title>调试信息</title></head>
    <body style="font-family: Arial; padding: 20px;">
        <h1>系统调试信息</h1>
        
        <h2>当前用户</h2>
        <p><strong>UID:</strong> {uid}</p>
        <p><strong>用户名:</strong> {user['username']}</p>
        <p><strong>搜索次数:</strong> {user['search_count']}</p>
        
        <h2>所有用户 ({len(users)})</h2>
        <ul>
        {''.join([f"<li>{u['username']} (UID: {uid_key[:8]}..., 搜索: {u['search_count']}次)</li>" for uid_key, u in users.items()])}
        </ul>
        
        <h2>日志统计</h2>
        <p><strong>点击日志:</strong> {len(logs["clicks"])} 条</p>
        <p><strong>停留日志:</strong> {len(logs["stays"])} 条</p>
        
        <h2>最近5条点击日志</h2>
        <ul>
        {''.join([f"<li>{c.get('username', '未知')} - {c.get('target', '未知')} - {c.get('query', '无')}</li>" for c in logs["clicks"][-5:]])}
        </ul>
        
        <h2>最近5条停留日志</h2>
        <ul>
        {''.join([f"<li>{s.get('username', '未知')} - {s.get('page', '未知')} - {s.get('duration', 0)/1000:.1f}秒</li>" for s in logs["stays"][-5:]])}
        </ul>
        
        <p><a href="/">返回首页</a></p>
    </body>
    </html>
    """

# ---- 启动 ----
if __name__ == "__main__":
    print("启动搜索引擎...")
    print("访问 http://localhost:5000/debug 查看调试信息")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=True)
