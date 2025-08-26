from flask import Flask, request, jsonify, render_template, session, url_for
import pandas as pd
import pickle, re
from sentence_transformers import SentenceTransformer, util
from flask_session import Session
import uuid
 
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
 
# Load SOP data with embeddings
with open("sop_embeddings.pkl", "rb") as f:
    df = pickle.load(f)
 
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
 
 
def make_sql_links(text):
    import re, uuid
    from flask import url_for
 
    def replacer(match):
        sql_code = match.group(0).strip()
        query_id = str(uuid.uuid4())  # unique id for query
        if 'sql_queries' not in session:
            session['sql_queries'] = {}
        session['sql_queries'][query_id] = sql_code
        return f'<a href="{url_for("view_sql", query_id=query_id)}" target="_blank">View SQL Query</a>'
 
    # Regex: SELECT ... ; plus optional trailing -- comments until newline breaks
    pattern = r"(SELECT[\s\S]*?;(?:\s*--.*(?:\n|$))*)"
    return re.sub(pattern, replacer, text, flags=re.IGNORECASE)
 
def make_login_link():
    login_details = """
1. Login to Citrix: https://ukcc.caas-int.vodafone.com/Citrix/UKCCWeb/
2. Login to Putty:
   Ops server1: -uk00029do / 104.242.252.68
3. Login to DB:
   Hostname: uk1exe3-7-4klea-scan.sne3001.ie1cn013309o1.oraclevcn.com
   Port: 1521
   Service Name: DRCMIRRPRCRM.sne3001.ie1cn013307o1.oraclevcn.com
"""
    query_id = str(uuid.uuid4())
    if 'sql_queries' not in session:
        session['sql_queries'] = {}
    session['sql_queries'][query_id] = login_details
    return f'<a href="{url_for("view_sql", query_id=query_id)}" target="_blank">View Login Details</a>'
 
 
@app.route("/view_login/<login_id>")
def view_login(login_id):
    login_info = session.get('login_details', {}).get(login_id, "Login details not found")
    # Wrap in <pre> to preserve line breaks
    return f"<pre>{login_info}</pre>"
@app.route("/")
def home():
    session.clear()
    session['greeted'] = False  # Wait for user greeting
    return render_template("index_sop.html")
 
 
def detect_intent(query):
    q = query.lower()
 
    # Special rule: help me/us with assetisation → How
    if re.search(r"\b(help|assist|support|need|give|want).*(assetisation|assetize|assetise|hbb cancellation|HBB Cancellation)\b", q):
        return "How"
 
    # Prioritize mutually exclusive intent detection
    if any(word in q for word in ["when", "time", "timing"]):
        return "When"
    elif any(word in q for word in ["why", "what", "reason", "purpose", "need"]):
        return "Why"
    elif any(word in q for word in ["how", "perform", "do", "action", "steps"]):
        return "How"
    else:
        return "General"
 
 
@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query").strip()
 
    greetings = ["hi", "hello", "hey"]
    if not session.get('greeted') and any(g.lower() in user_query.lower() for g in greetings):
        session['greeted'] = True
        session['awaiting_more'] = False
        return jsonify({"response": "Hello! How may I assist you?"})
    # List of common irrelevant phrases
    irrelevant_phrases = [
    "how are you", "who are you", "what's the weather", "what is the weather",
    "how is the weather", "what do you do", "who made you", "where are you"
    ]
    # Inside ask() route, right after greetings handling
    if any(phrase in user_query.lower() for phrase in irrelevant_phrases):
        return jsonify({"response": "Please ask relevant questions"})
 
    # Check if bot is waiting for "Do you need more assistance?"
    if session.get('awaiting_more', False):
        if user_query.lower() in ["no", "nope", "nah", "not really"]:
            session['awaiting_more'] = False
            return jsonify({"response": "Thank you for contacting us! Have a great day 😊"})
        else:
            session['awaiting_more'] = False
            # continue normal processing for "Yes" or other queries
 
    # Check for SOP count request
    if re.search(r"\b(how many|total|count).*sops?\b", user_query, re.IGNORECASE):
        total_sops = df['SOP Name'].nunique()
        sop_list = df['SOP Name'].unique()
        return jsonify({
            "response": f"Total SOPs: {total_sops}\nList: {', '.join(sop_list)}"
        })
 
    # Section selection by button or partial match typing
    section_match = None
    if user_query.startswith("SECTION_"):
        section_match = user_query.replace("SECTION_", "", 1)
    else:
        # partial match search
        for sec in df['Section'].unique():
            if user_query.lower() in sec.lower():
                section_match = sec
                break
 
    if section_match:
        sop_name = session.get("sop_name")
        intent_type = session.get("intent_type")
        filtered = df[(df['SOP Name'] == sop_name) & (df['Intent Type'] == intent_type)]
        content_list = filtered[filtered['Section'] == section_match]['Content'].tolist()
 
        # Clean content: remove old bullets, add Pre Checks on new line
        content_text = "\n".join([c for c in content_list])
        content_text = re.sub(r"• Source:", "", content_text)
        content_text = re.sub(r"(Pre Checks:)", r"\n\1", content_text)
        #content_text = "\n".join([f"• {line}" for line in content_text.split("\n") if line.strip()])
        # Preserve formatting in HTML
        #content_text = "<pre>" + content_text + "</pre>"
        content_text = content_text.replace("\n", "<br>")  # convert all newlines to <br>
 
 
        # Replace SQL blocks with links
        content_text = make_sql_links(content_text)
        # Prechecks ke baad View Login Details add karna
        if "Pre Checks:" in content_text:
            content_text = content_text.replace("Pre Checks:", "Pre Checks:<br>" + make_login_link())
        response_text = f"{content_text}<br><br>Do you need more assistance?"
        session['awaiting_more'] = True
        return jsonify({"response": response_text})
 
    # Normal SOP intent handling with embeddings
    query_emb = embed_model.encode(user_query, convert_to_tensor=True)
    df['similarity'] = df['embedding'].apply(lambda x: util.cos_sim(query_emb, x).item())
    sop_name = df.sort_values(by='similarity', ascending=False).iloc[0]['SOP Name']
    session['sop_name'] = sop_name
 
    intent_type = detect_intent(user_query)
    session['intent_type'] = intent_type
 
    filtered = df[(df['SOP Name'] == sop_name) & (df['Intent Type'] == intent_type)]
    sections = filtered['Section'].unique()
    buttons = [{"text": sec, "value": f"SECTION_{sec}"} for sec in sections]
 
    session['awaiting_more'] = False  # reset flag
 
    return jsonify({
        "response": "Please select a section to view details:",
        "buttons": buttons
    })
 
 
@app.route("/query/<query_id>")
def view_sql(query_id):
    sql_code = session.get('sql_queries', {}).get(query_id, "Query not found")
    return f"<pre>{sql_code}</pre>"
 
 
if __name__ == "__main__":
    app.run(debug=True)
