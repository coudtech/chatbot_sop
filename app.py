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
    elif any(word in q for word in ["why", "reason", "purpose", "need"]):
        return "Why"
    elif any(word in q for word in ["how", "perform", "do", "action", "steps"]):
        return "How"
    else:
        return "General"

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query").strip()

    greetings = ["hi","hello","hey"]
    if not session.get('greeted') and any(g.lower() in user_query.lower() for g in greetings):
        session['greeted'] = True
        session['awaiting_more'] = False
        return jsonify({"response": "Hello! How may I assist you?"})

    # Check if bot is waiting for "Do you need more assistance?"
    if session.get('awaiting_more', False):
        if user_query.lower() in ["no", "nope", "nah", "not really"]:
            session['awaiting_more'] = False
            return jsonify({"response": "Thank you for contacting us! Have a great day 😊"})
        else:
            session['awaiting_more'] = False
            # continue normal processing for "Yes" or other queries

    # Section button click
    # Section button click
    if user_query.startswith("SECTION_"):
        section_text = user_query.replace("SECTION_", "", 1)  # Remove SECTION_ prefix
        sop_name = session.get("sop_name")
        intent_type = session.get("intent_type")

        filtered = df[(df['SOP Name'] == sop_name) & (df['Intent Type'] == intent_type)]
        content_list = filtered[filtered['Section'] == section_text]['Content'].tolist()
        
        # First join content into text
        content_text = "\n".join([f"• {c}" for c in content_list])
        
        '''
        def make_sql_links(text):
            # Replace ```sql ... ``` blocks with link
            def replacer(match):
                sql_code = match.group(1).strip()
                query_id = str(uuid.uuid4())  # unique id for query
                # save sql to session or dict
                if 'sql_queries' not in session:
                    session['sql_queries'] = {}
                session['sql_queries'][query_id] = sql_code
                return f'<a href="{url_for("view_sql", query_id=query_id)}" target="_blank">View SQL Query</a>'
            import re
            return re.sub(r"```sql([\s\S]*?)```", replacer, text, flags=re.IGNORECASE)
        '''

        # Then replace sql blocks with links
        content_text = make_sql_links(content_text)

        # Only content + Do you need more assistance? with one line gap
        response_text = f"{content_text}\n\nDo you need more assistance?"
        session['awaiting_more'] = True  # set flag
        return jsonify({"response": response_text})


    # Normal SOP intent handling...
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