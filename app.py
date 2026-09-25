import streamlit as st
import sqlite3
import pandas as pd
import re
import io
import os
import hashlib
import math
from collections import Counter
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Download NLTK data silently ──────────────────────────────────────────────
for corpus in ["stopwords", "punkt", "wordnet", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f"corpora/{corpus}")
    except LookupError:
        nltk.download(corpus, quiet=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="SmartHire AI", page_icon="🤖", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

*, body { font-family: 'DM Sans', sans-serif; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    padding: 48px 40px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 32px;
}
.hero h1 { font-family: 'DM Serif Display'; font-size: 3rem; margin: 0; letter-spacing: -1px; }
.hero p  { font-size: 1.1rem; opacity: 0.75; margin: 10px 0 0; }

/* ── Portal cards ── */
.pcard {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 32px 28px;
    box-shadow: 0 4px 24px rgba(0,0,0,.07);
    text-align: center;
    transition: box-shadow .2s;
}
.pcard:hover { box-shadow: 0 8px 32px rgba(0,0,0,.14); }
.pcard h2   { font-family: 'DM Serif Display'; font-size: 1.7rem; margin: 14px 0 8px; }

/* ── Section header ── */
.section-title {
    font-family: 'DM Serif Display';
    font-size: 1.5rem;
    color: #1e3a5f;
    border-left: 4px solid #2c5364;
    padding-left: 12px;
    margin-bottom: 18px;
}

/* ── Metric card ── */
.mcard {
    background: linear-gradient(135deg,#1e3a5f,#2c5364);
    color: white;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
}
.mcard .val { font-size: 2rem; font-weight: 700; }
.mcard .lbl { font-size: 0.8rem; opacity: .75; }

/* ── Badges ── */
.badge-g { background:#d1fae5; color:#065f46; padding:3px 12px; border-radius:20px; font-size:.82rem; font-weight:600; }
.badge-y { background:#fef3c7; color:#92400e; padding:3px 12px; border-radius:20px; font-size:.82rem; font-weight:600; }
.badge-r { background:#fee2e2; color:#991b1b; padding:3px 12px; border-radius:20px; font-size:.82rem; font-weight:600; }

/* ── Job post card ── */
.jobcard {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-left: 5px solid #2c5364;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 14px;
}
.jobcard h4 { margin: 0 0 4px; color: #1e3a5f; }
.jobcard p  { margin: 2px 0; font-size: .9rem; color: #555; }

/* ── Roadmap step ── */
.roadstep {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
}

/* ── AI Detection result ── */
.ai-human  { background:#d1fae5; color:#065f46; border-radius:12px; padding:14px 20px; font-size:1.1rem; font-weight:600; }
.ai-risk   { background:#fee2e2; color:#991b1b; border-radius:12px; padding:14px 20px; font-size:1.1rem; font-weight:600; }
.ai-warn   { background:#fef3c7; color:#92400e; border-radius:12px; padding:14px 20px; font-size:1.1rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
DB = "smarthire_v3.db"

def get_conn():
    return sqlite3.connect(DB, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS recruiters (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        username  TEXT UNIQUE,
        password  TEXT,
        company   TEXT,
        email     TEXT
    );
    CREATE TABLE IF NOT EXISTS jobseekers (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        username   TEXT UNIQUE,
        password   TEXT,
        name       TEXT,
        email      TEXT,
        seeker_type TEXT,
        -- experienced
        prev_company TEXT, years_exp INTEGER, prev_role TEXT,
        -- fresher
        qualification TEXT, skills TEXT, institution TEXT, certifications TEXT
    );
    CREATE TABLE IF NOT EXISTS job_posts (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        recruiter     TEXT,
        company       TEXT,
        title         TEXT,
        description   TEXT,
        required_skills TEXT,
        experience_needed TEXT,
        education_needed  TEXT,
        posted_on     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS screening_reports (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        recruiter   TEXT,
        job_id      INTEGER,
        filename    TEXT,
        overall     REAL,
        tfidf       REAL,
        skill_pct   REAL,
        exp_pct     REAL,
        qual_pct    REAL,
        ai_score    REAL,
        ai_verdict  TEXT,
        matched_skills TEXT,
        missing_skills TEXT,
        status      TEXT,
        screened_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()

init_db()

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

ss("page", "landing")
ss("role", None)
ss("user", None)

# ══════════════════════════════════════════════════════════════════════════════
# NLP ENGINE
# ══════════════════════════════════════════════════════════════════════════════
STOP = set(stopwords.words("english"))
LEM  = WordNetLemmatizer()

SKILL_VOCAB = [
    "python","java","c","c++","javascript","typescript","html","css","sql","nosql",
    "mysql","postgresql","mongodb","sqlite","redis","elasticsearch","graphql","rest","api",
    "machine learning","deep learning","nlp","natural language processing","computer vision",
    "artificial intelligence","data science","data analysis","statistics","mathematics",
    "tensorflow","pytorch","keras","scikit-learn","pandas","numpy","matplotlib","seaborn",
    "spark","hadoop","airflow","kafka","dbt","aws","azure","gcp","google cloud","cloud",
    "docker","kubernetes","terraform","ansible","jenkins","ci/cd","devops","mlops",
    "git","github","gitlab","linux","bash","shell","excel","power bi","tableau","looker",
    "django","flask","fastapi","spring","react","angular","vue","nodejs","nextjs",
    "communication","leadership","teamwork","problem solving","agile","scrum","jira",
    "project management","presentation","critical thinking","time management"
]

COURSERA_LINKS = {
    "Python":               "https://www.coursera.org/learn/python",
    "Machine Learning":     "https://www.coursera.org/learn/machine-learning",
    "Deep Learning":        "https://www.coursera.org/specializations/deep-learning",
    "Nlp":                  "https://www.coursera.org/specializations/natural-language-processing",
    "Data Science":         "https://www.coursera.org/professional-certificates/ibm-data-science",
    "Sql":                  "https://www.coursera.org/learn/sql-for-data-science",
    "Aws":                  "https://www.coursera.org/learn/aws-fundamentals",
    "Azure":                "https://www.coursera.org/learn/microsoft-azure-cloud-services",
    "Tableau":              "https://www.coursera.org/learn/analytics-tableau",
    "Power Bi":             "https://www.coursera.org/learn/power-bi-in-90-minutes",
    "Docker":               "https://www.coursera.org/learn/docker-for-the-absolute-beginner",
    "Kubernetes":           "https://www.coursera.org/learn/google-kubernetes-engine",
    "React":                "https://www.coursera.org/learn/react-basics",
    "Django":               "https://www.coursera.org/learn/django-features-libraries",
    "Tensorflow":           "https://www.coursera.org/professional-certificates/tensorflow-in-practice",
    "Git":                  "https://www.coursera.org/learn/introduction-git-github",
    "Agile":                "https://www.coursera.org/learn/agile-development-and-scrum",
    "Project Management":   "https://www.coursera.org/professional-certificates/google-project-management",
}

def extract_text(uploaded_file):
    raw = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            text = "\n".join(
                p.extract_text() for p in reader.pages if p.extract_text()
            )
            if not text.strip():
                st.warning("⚠️ Could not extract text — may be a scanned PDF.")
            return text
        except Exception as e:
            st.error(f"PDF read error: {e}")
            return ""
    return raw.decode("utf-8", errors="ignore")

def nlp_clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s\+\#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [LEM.lemmatize(w) for w in text.split() if w not in STOP and len(w) > 1]
    return " ".join(tokens)

def extract_skills_nlp(text):
    tl = text.lower()
    found = []
    for skill in SKILL_VOCAB:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, tl):
            found.append(skill.title())
    return list(set(found))

def tfidf_similarity(text_a, text_b):
    ca, cb = nlp_clean(text_a), nlp_clean(text_b)
    if not ca or not cb:
        return 0.0
    try:
        vec = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
        mat = vec.fit_transform([ca, cb])
        return round(float(cosine_similarity(mat[0:1], mat[1:2])[0][0]) * 100, 2)
    except:
        return 0.0

def skill_gap(job_skills, resume_skills):
    js = {s.lower() for s in job_skills}
    rs = {s.lower() for s in resume_skills}
    matched = [s.title() for s in js & rs]
    missing = [s.title() for s in js - rs]
    score   = round(len(matched) / max(len(js), 1) * 100, 2)
    return score, matched, missing

def experience_match(resume_text, required_yrs):
    if required_yrs == 0:
        return 100.0
    nums = re.findall(r"(\d+)\s*(?:\+\s*)?(?:year|years|yrs)", resume_text.lower())
    if not nums:
        return 40.0
    max_exp = max(int(n) for n in nums)
    return 100.0 if max_exp >= required_yrs else round(max_exp / required_yrs * 100, 2)

def qualification_match(resume_text, edu_required):
    if not edu_required.strip():
        return 100.0
    return 100.0 if edu_required.lower() in resume_text.lower() else 55.0

def keyword_density(text, keywords):
    words = text.lower().split()
    total = max(len(words), 1)
    hits  = sum(words.count(kw.lower()) for kw in keywords)
    return hits / total

# ── AI / Human Detection (advanced heuristics + NLP signals) ────────────────
AI_PHRASES = [
    "as an ai","as a language model","i am an ai","i cannot","i do not have",
    "highly motivated","results-driven","results driven","dynamic professional",
    "team player","fast learner","hardworking individual","passionate about",
    "detail-oriented","strong communication skills","proven track record",
    "synergy","leverage","utilize","spearhead","catalyze","orchestrate",
    "in today's fast-paced","in the ever-evolving","in conclusion","in summary",
    "to whom it may concern","i am writing to express"
]

def detect_ai_content(text):
    text_lower = text.lower()
    suspicion  = 0
    signals    = []

    # 1. Generic AI phrases
    for phrase in AI_PHRASES:
        if phrase in text_lower:
            suspicion += 8
            signals.append(f"Generic/AI phrase: '{phrase}'")

    # 2. Keyword stuffing
    words = text_lower.split()
    tech_kws = ["python","sql","java","ai","machine","learning","cloud","docker","aws"]
    for kw in tech_kws:
        cnt = words.count(kw)
        if cnt > 5:
            suspicion += min(cnt * 3, 18)
            signals.append(f"Keyword stuffed ({cnt}×): '{kw}'")

    # 3. Skill density without evidence
    skills_found = extract_skills_nlp(text)
    has_evidence = any(w in text_lower for w in
                       ["project","built","developed","implemented","designed",
                        "achieved","delivered","automated","reduced","increased"])
    if len(skills_found) > 12 and not has_evidence:
        suspicion += 20
        signals.append("Many skills but zero project evidence found.")

    # 4. Sentence length uniformity (AI tends to be very uniform)
    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if sentences:
        lens  = [len(s.split()) for s in sentences]
        mean  = sum(lens) / len(lens)
        variance = sum((l - mean)**2 for l in lens) / len(lens)
        if variance < 15 and len(sentences) > 5:
            suspicion += 15
            signals.append("Unusually uniform sentence lengths (AI pattern).")

    # 5. Lexical diversity (TTR)
    words_clean = [w for w in re.findall(r"[a-z]+", text_lower) if len(w) > 2]
    if words_clean:
        ttr = len(set(words_clean)) / len(words_clean)
        if ttr < 0.35:
            suspicion += 12
            signals.append(f"Low lexical diversity (TTR={ttr:.2f}), typical of AI text.")

    # 6. Repetitive bullet structure
    bullets = [l for l in text.split("\n") if l.strip().startswith(("•","-","*","–"))]
    if len(bullets) > 15:
        suspicion += 10
        signals.append(f"Excessively structured ({len(bullets)} bullets) — AI formatting.")

    # 7. Word count extremes
    wc = len(text.split())
    if wc < 80:
        suspicion += 12
        signals.append("Resume too short — insufficient content.")
    elif wc > 1200:
        suspicion += 8
        signals.append("Extremely long resume — possible padding.")

    ai_probability = min(100, suspicion)
    human_score    = max(0, 100 - suspicion)

    if ai_probability >= 55:
        verdict = "🚨 HIGH AI RISK — Likely AI-Generated"
        cls     = "ai-risk"
    elif ai_probability >= 30:
        verdict = "⚠️ MODERATE RISK — Needs Manual Review"
        cls     = "ai-warn"
    else:
        verdict = "✅ LIKELY HUMAN-WRITTEN — Appears Genuine"
        cls     = "ai-human"

    return human_score, ai_probability, verdict, cls, signals

def composite_score(resume_text, job_full_text, job_skills,
                    required_exp, edu_required):
    tfidf  = tfidf_similarity(resume_text, job_full_text)
    r_skills = extract_skills_nlp(resume_text)
    sk_pct, matched, missing = skill_gap(job_skills, r_skills)
    exp_pct  = experience_match(resume_text, required_exp)
    qual_pct = qualification_match(resume_text, edu_required)
    overall  = round(tfidf*0.30 + sk_pct*0.40 + exp_pct*0.20 + qual_pct*0.10, 2)
    return overall, tfidf, sk_pct, exp_pct, qual_pct, matched, missing

def final_status(score, human_score):
    if human_score < 45:
        return "🔎 Manual Review (Authenticity Risk)"
    if score >= 78:
        return "🌟 Highly Suitable — Shortlisted"
    elif score >= 55:
        return "👍 Moderately Suitable"
    else:
        return "❌ Does Not Meet Requirements"

def learning_roadmap(missing_skills):
    steps = []
    for i, skill in enumerate(missing_skills, 1):
        link = COURSERA_LINKS.get(skill, f"https://www.coursera.org/search?query={skill.replace(' ','%20')}")
        steps.append((i, skill, link))
    return steps

# ══════════════════════════════════════════════════════════════════════════════
# ── HELPERS ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def metric_html(value, label):
    return f"""
    <div class="mcard">
        <div class="val">{value}</div>
        <div class="lbl">{label}</div>
    </div>"""

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def job_card_html(title, company, skills, exp, edu, posted):
    return f"""
    <div class="jobcard">
        <h4>🏢 {title} — {company}</h4>
        <p>🛠 <b>Skills:</b> {skills}</p>
        <p>📅 <b>Experience:</b> {exp} &nbsp;|&nbsp; 🎓 <b>Education:</b> {edu}</p>
        <p style="font-size:.78rem;color:#888;">Posted: {posted}</p>
    </div>"""

# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_landing():
    st.markdown("""
    <div class="hero">
        <h1>🤖 SmartHire AI</h1>
        <p>Advanced NLP-Powered Two-Sided Recruitment Ecosystem</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="pcard">
            <div style="font-size:3.5rem">🏢</div>
            <h2>Recruiters</h2>
            <p style="color:#555;font-size:.95rem;">
                Post jobs · Screen resumes with NLP · Detect AI-written CVs ·
                Rank candidates · Download shortlist reports
            </p>
        </div>""", unsafe_allow_html=True)
        st.write("")
        if st.button("Enter Recruiter Portal →", use_container_width=True, type="primary"):
            st.session_state.role = "recruiter"
            st.session_state.page = "auth"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="pcard">
            <div style="font-size:3.5rem">🎓</div>
            <h2>Job Seekers</h2>
            <p style="color:#555;font-size:.95rem;">
                Browse live jobs · Upload your resume · Get gap analysis ·
                Follow personalised learning roadmap with course links
            </p>
        </div>""", unsafe_allow_html=True)
        st.write("")
        if st.button("Enter Job Seeker Portal →", use_container_width=True, type="primary"):
            st.session_state.role = "jobseeker"
            st.session_state.page = "auth"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE  (shared, role-aware)
# ══════════════════════════════════════════════════════════════════════════════
def render_auth():
    role_label = "Recruiter" if st.session_state.role == "recruiter" else "Job Seeker"
    st.markdown(f"""
    <div class="hero" style="padding:28px;">
        <h1 style="font-size:2rem;">🔐 {role_label} Gateway</h1>
    </div>""", unsafe_allow_html=True)

    mode = st.radio("", ["🔑 Login", "📝 Create Account"], horizontal=True)
    st.write("")

    col_gap, col_form, col_gap2 = st.columns([1, 2, 1])
    with col_form:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        # ── CREATE ACCOUNT ──────────────────────────────────────────────────
        if "Create Account" in mode:
            if st.session_state.role == "recruiter":
                company = st.text_input("Company Name")
                email   = st.text_input("Work Email")
                if st.button("Create Recruiter Account", type="primary", use_container_width=True):
                    if username and password and company:
                        conn = get_conn(); c = conn.cursor()
                        try:
                            c.execute(
                                "INSERT INTO recruiters(username,password,company,email) VALUES(?,?,?,?)",
                                (username, hash_pw(password), company, email)
                            )
                            conn.commit()
                            st.success("✅ Account created! Please login.")
                        except sqlite3.IntegrityError:
                            st.error("Username already taken.")
                        finally:
                            conn.close()
                    else:
                        st.error("Please fill all required fields.")

            else:  # job seeker
                name   = st.text_input("Full Name", key="reg_name")
                email  = st.text_input("Email", key="reg_email")
                s_type = st.radio("I am a:", ["Experienced Professional", "Fresh Graduate"],
                                  horizontal=True, key="reg_stype")

                if "Experienced" in s_type:
                    prev_co   = st.text_input("Previous Company", key="reg_prevco")
                    yrs       = st.number_input("Years of Experience", 0, 50, 1, key="reg_yrs")
                    prev_role = st.text_input("Previous Role / Job Title", key="reg_role")
                    skills_i  = st.text_area("Key Skills (comma-separated)", "Python, SQL", key="reg_skills")
                    qual_i    = st.text_input("Highest Qualification", "B.Tech / B.E.", key="reg_qual")
                    inst      = st.text_input("Institution", key="reg_inst")
                    certs     = st.text_area("Certifications (if any)", key="reg_certs")
                else:
                    prev_co   = ""
                    prev_role = ""
                    yrs       = 0
                    qual_i    = st.text_input("Highest Qualification / Pursuing Degree", key="reg_qual")
                    inst      = st.text_input("Institution / College Name", key="reg_inst")
                    skills_i  = st.text_area("Skills (comma-separated)", "Python, HTML", key="reg_skills")
                    certs     = st.text_area("Certifications / Online Courses", key="reg_certs")

                if st.button("Create Job Seeker Account", type="primary", use_container_width=True):
                    # Collect values directly from session_state keys to avoid scope issues
                    _username  = username.strip()
                    _password  = password.strip()
                    _name      = st.session_state.get("reg_name", "").strip()
                    _email     = st.session_state.get("reg_email", "").strip()
                    _stype     = st.session_state.get("reg_stype", "Experienced Professional")
                    _prevco    = st.session_state.get("reg_prevco", "").strip()
                    _yrs       = st.session_state.get("reg_yrs", 0)
                    _role      = st.session_state.get("reg_role", "").strip()
                    _qual      = st.session_state.get("reg_qual", "").strip()
                    _skills    = st.session_state.get("reg_skills", "").strip()
                    _inst      = st.session_state.get("reg_inst", "").strip()
                    _certs     = st.session_state.get("reg_certs", "").strip()

                    missing_fields = []
                    if not _username: missing_fields.append("Username")
                    if not _password: missing_fields.append("Password")
                    if not _name:     missing_fields.append("Full Name")

                    if missing_fields:
                        st.error(f"Please fill required fields: {', '.join(missing_fields)}")
                    else:
                        conn = get_conn(); c = conn.cursor()
                        try:
                            c.execute("""INSERT INTO jobseekers
                                (username,password,name,email,seeker_type,
                                 prev_company,years_exp,prev_role,
                                 qualification,skills,institution,certifications)
                                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                                (_username, hash_pw(_password), _name, _email,
                                 "experienced" if "Experienced" in _stype else "fresher",
                                 _prevco, _yrs, _role,
                                 _qual, _skills, _inst, _certs))
                            conn.commit()
                            st.success("✅ Account created! Switch to Login to continue.")
                        except sqlite3.IntegrityError:
                            st.error("Username already taken. Please choose a different one.")
                        finally:
                            conn.close()

        # ── LOGIN ────────────────────────────────────────────────────────────
        else:
            if st.button("Login", type="primary", use_container_width=True):
                table = "recruiters" if st.session_state.role == "recruiter" else "jobseekers"
                conn = get_conn(); c = conn.cursor()
                c.execute(f"SELECT * FROM {table} WHERE username=? AND password=?",
                          (username, hash_pw(password)))
                row = c.fetchone()
                conn.close()
                if row:
                    st.session_state.user = username
                    st.session_state.page = ("recruiter_dash"
                                             if st.session_state.role == "recruiter"
                                             else "seeker_dash")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

        st.write("")
        if st.button("← Back to Home"):
            st.session_state.page = "landing"
            st.session_state.role = None
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# RECRUITER DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def render_recruiter_dash():
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT company FROM recruiters WHERE username=?", (st.session_state.user,))
    row = c.fetchone()
    company = row[0] if row else "Your Company"

    st.markdown(f"""
    <div class="hero" style="padding:24px;">
        <h1 style="font-size:1.9rem;">🏢 {company} — Recruiter Dashboard</h1>
        <p>Logged in as <b>{st.session_state.user}</b></p>
    </div>""", unsafe_allow_html=True)

    if st.sidebar.button("🚪 Logout"):
        st.session_state.user = None
        st.session_state.page = "landing"
        st.session_state.role = None
        st.rerun()

    tab1, tab2, tab3 = st.tabs([
        "📌 Post & Manage Jobs",
        "📄 Screen Resumes",
        "📊 Screening Reports"
    ])

    # ══ TAB 1: POST JOBS ════════════════════════════════════════════════════
    with tab1:
        section("Create a New Job Posting")
        with st.form("job_form"):
            c1, c2 = st.columns(2)
            title       = c1.text_input("Job Title *", placeholder="e.g. Data Scientist")
            company_f   = c2.text_input("Company Name *", value=company)
            description = st.text_area("Job Description *", height=130,
                                       placeholder="Describe the role, responsibilities, team...")
            req_skills  = st.text_area("Required Skills (comma-separated) *",
                                       placeholder="Python, SQL, Machine Learning, NLP")
            c3, c4 = st.columns(2)
            exp_need    = c3.text_input("Experience Needed", placeholder="e.g. 3+ years, Freshers OK")
            edu_need    = c4.text_input("Educational Requirement", placeholder="e.g. B.Tech, MCA")
            submitted   = st.form_submit_button("🚀 Publish Job Post", type="primary")
            if submitted:
                if title and req_skills and description:
                    c.execute("""INSERT INTO job_posts
                        (recruiter,company,title,description,required_skills,
                         experience_needed,education_needed)
                        VALUES(?,?,?,?,?,?,?)""",
                        (st.session_state.user, company_f, title, description,
                         req_skills, exp_need, edu_need))
                    conn.commit()
                    st.success(f"✅ **{title}** published successfully!")
                else:
                    st.error("Please fill Title, Description, and Required Skills.")

        st.write("")
        section("Your Active Job Posts")
        c.execute("""SELECT id,title,company,required_skills,experience_needed,
                            education_needed,posted_on
                     FROM job_posts WHERE recruiter=? ORDER BY posted_on DESC""",
                  (st.session_state.user,))
        posts = c.fetchall()
        if posts:
            for p in posts:
                st.markdown(job_card_html(p[1], p[2], p[3], p[4] or "—",
                                          p[5] or "—", p[6][:10]),
                            unsafe_allow_html=True)
        else:
            st.info("No job posts yet. Create one above!")

    # ══ TAB 2: SCREEN RESUMES ═══════════════════════════════════════════════
    with tab2:
        section("AI-Powered Resume Screening")

        c.execute("SELECT id,title FROM job_posts WHERE recruiter=? ORDER BY posted_on DESC",
                  (st.session_state.user,))
        jobs = c.fetchall()
        if not jobs:
            st.warning("Post at least one job to start screening.")
        else:
            job_map = {f"{j[1]} (#{j[0]})": j[0] for j in jobs}
            sel_job = st.selectbox("Select Job Profile to Screen Against", list(job_map.keys()))
            job_id  = job_map[sel_job]

            c.execute("""SELECT title,description,required_skills,
                                experience_needed,education_needed
                         FROM job_posts WHERE id=?""", (job_id,))
            jrow = c.fetchone()
            j_title, j_desc, j_skills_raw, j_exp_raw, j_edu = jrow
            full_job_text = f"{j_title} {j_desc} {j_skills_raw}"
            job_skills    = extract_skills_nlp(full_job_text)

            # Parse experience years from text
            exp_nums = re.findall(r"(\d+)", j_exp_raw or "0")
            req_exp  = int(exp_nums[0]) if exp_nums else 0

            st.info(f"**Detected Required Skills ({len(job_skills)}):** {', '.join(job_skills) or 'None'}")

            uploaded = st.file_uploader("Upload Resumes (.pdf or .txt)",
                                        type=["pdf","txt"],
                                        accept_multiple_files=True)

            if st.button("🔍 Run NLP Screening", type="primary") and uploaded:
                results = []
                prog = st.progress(0)
                for i, f in enumerate(uploaded):
                    prog.progress((i+1)/len(uploaded))
                    text = extract_text(f)
                    if not text.strip():
                        results.append({
                            "File": f.name, "Overall": 0, "TF-IDF": 0,
                            "Skill %": 0, "Exp %": 0, "Qual %": 0,
                            "Human Score": 0, "AI Risk %": 100,
                            "AI Verdict": "🚨 Unreadable",
                            "Matched": "", "Missing": "", "Status": "❌ Unreadable"
                        })
                        continue

                    overall, tfidf, sk, exp_pct, qual_pct, matched, missing = \
                        composite_score(text, full_job_text, job_skills, req_exp, j_edu or "")

                    human_s, ai_risk, verdict, cls, signals = detect_ai_content(text)
                    status = final_status(overall, human_s)

                    results.append({
                        "File":        f.name,
                        "Overall":     overall,
                        "TF-IDF":      tfidf,
                        "Skill %":     sk,
                        "Exp %":       exp_pct,
                        "Qual %":      qual_pct,
                        "Human Score": human_s,
                        "AI Risk %":   ai_risk,
                        "AI Verdict":  verdict,
                        "Matched":     ", ".join(matched),
                        "Missing":     ", ".join(missing),
                        "Status":      status,
                        "_text":       text,
                        "_signals":    signals
                    })

                    # Save to DB
                    c.execute("""INSERT INTO screening_reports
                        (recruiter,job_id,filename,overall,tfidf,skill_pct,exp_pct,
                         qual_pct,ai_score,ai_verdict,matched_skills,missing_skills,status)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (st.session_state.user, job_id, f.name, overall, tfidf, sk,
                         exp_pct, qual_pct, ai_risk, verdict,
                         ", ".join(matched), ", ".join(missing), status))
                conn.commit()
                prog.empty()

                df = pd.DataFrame(results).sort_values("Overall", ascending=False).reset_index(drop=True)
                df.insert(0, "Rank", range(1, len(df)+1))

                st.success(f"✅ Screened **{len(df)}** resumes successfully!")
                st.write("")
                section("🏆 Candidate Ranking Leaderboard")
                st.dataframe(
                    df[["Rank","File","Overall","Skill %","Exp %","Human Score","AI Risk %","Status"]],
                    use_container_width=True
                )

                section("📋 Individual Candidate Reports")
                for _, row in df.iterrows():
                    shortlist_tag = "🌟 SHORTLISTED" if "Shortlisted" in row["Status"] else ""
                    with st.expander(f"Rank #{row['Rank']}: {row['File']}  {shortlist_tag}"):

                        # Score metrics
                        m1,m2,m3,m4,m5 = st.columns(5)
                        m1.metric("Overall", f"{row['Overall']}%")
                        m2.metric("TF-IDF NLP", f"{row['TF-IDF']}%")
                        m3.metric("Skill Match", f"{row['Skill %']}%")
                        m4.metric("Experience", f"{row['Exp %']}%")
                        m5.metric("Qualification", f"{row['Qual %']}%")

                        # AI detection box
                        st.markdown(
                            f'<div class="{row.get("_cls","ai-warn") if "_cls" in row else "ai-warn"}">'
                            f'AI Detection: {row["AI Verdict"]} &nbsp;|&nbsp; '
                            f'Human Score: {row["Human Score"]}% &nbsp;|&nbsp; '
                            f'AI Risk: {row["AI Risk %"]}%</div>',
                            unsafe_allow_html=True
                        )

                        if "_signals" in row and row["_signals"]:
                            st.markdown("**🕵️ Authenticity Signals Detected:**")
                            for sig in row["_signals"]:
                                st.write(f"  • {sig}")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"✅ **Matched Skills:** {row['Matched'] or 'None'}")
                        with col_b:
                            st.markdown(f"❌ **Missing Skills:** {row['Missing'] or 'None'}")

                        st.markdown(f"**Final Status:** {row['Status']}")
                        st.progress(row["Overall"] / 100)

                # Download
                export_cols = ["Rank","File","Overall","TF-IDF","Skill %",
                               "Exp %","Qual %","Human Score","AI Risk %",
                               "AI Verdict","Matched","Missing","Status"]
                csv = df[export_cols].to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Full Report (CSV)", csv,
                                   "SmartHire_Screening_Report.csv", "text/csv")

    # ══ TAB 3: PAST REPORTS ══════════════════════════════════════════════════
    with tab3:
        section("Screening History")
        c.execute("""SELECT filename,overall,skill_pct,ai_score,ai_verdict,
                            status,screened_on
                     FROM screening_reports WHERE recruiter=?
                     ORDER BY screened_on DESC LIMIT 50""",
                  (st.session_state.user,))
        hist = c.fetchall()
        if hist:
            hdf = pd.DataFrame(hist, columns=["File","Overall%","Skill%",
                                               "AI Risk%","AI Verdict",
                                               "Status","Screened On"])
            st.dataframe(hdf, use_container_width=True)
        else:
            st.info("No screening history yet.")

    conn.close()

# ══════════════════════════════════════════════════════════════════════════════
# JOB SEEKER DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def render_seeker_dash():
    conn = get_conn(); c = conn.cursor()
    c.execute("""SELECT name,seeker_type,prev_company,years_exp,prev_role,
                        qualification,skills,institution,certifications
                 FROM jobseekers WHERE username=?""", (st.session_state.user,))
    row = c.fetchone()
    if not row:
        st.error("Profile not found."); conn.close(); return

    (name, s_type, prev_co, yrs_exp, prev_role,
     qual, skills_raw, institution, certs) = row

    profile_text = f"{skills_raw or ''} {qual or ''} {prev_role or ''} {certs or ''}"
    is_exp = s_type == "experienced"

    st.markdown(f"""
    <div class="hero" style="padding:24px;">
        <h1 style="font-size:1.9rem;">🎓 Welcome, {name}!</h1>
        <p>{'Experienced Professional' if is_exp else 'Fresh Graduate'} Portal</p>
    </div>""", unsafe_allow_html=True)

    if st.sidebar.button("🚪 Logout"):
        st.session_state.user = None
        st.session_state.page = "landing"
        st.session_state.role = None
        st.rerun()

    # Profile summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Your Profile")
    if is_exp:
        st.sidebar.markdown(f"**Company:** {prev_co or '—'}")
        st.sidebar.markdown(f"**Role:** {prev_role or '—'}")
        st.sidebar.markdown(f"**Experience:** {yrs_exp} yrs")
    else:
        st.sidebar.markdown(f"**Qualification:** {qual or '—'}")
        st.sidebar.markdown(f"**Institution:** {institution or '—'}")
    st.sidebar.markdown(f"**Skills:** {skills_raw or '—'}")

    tab1, tab2, tab3 = st.tabs([
        "🏢 Browse Job Opportunities",
        "📊 Resume Analyzer & Gap Report",
        "📚 Learning Roadmap & Courses"
    ])

    # ══ TAB 1: BROWSE JOBS ══════════════════════════════════════════════════
    with tab1:
        section("Live Job Opportunities")

        c.execute("""SELECT jp.id,jp.title,jp.company,jp.description,
                            jp.required_skills,jp.experience_needed,
                            jp.education_needed,jp.posted_on,r.company
                     FROM job_posts jp
                     JOIN recruiters r ON jp.recruiter=r.username
                     ORDER BY jp.posted_on DESC""")
        jobs = c.fetchall()

        if not jobs:
            st.info("No job postings available yet. Check back soon!")
        else:
            # Quick AI match filter
            my_skills = extract_skills_nlp(profile_text)
            show_matching = st.checkbox("🔍 Show only roles matching my skills", value=False)

            count = 0
            for j in jobs:
                (jid, title, company, desc, req_sk, exp_need,
                 edu_need, posted, rec_co) = j

                full_jt   = f"{title} {desc} {req_sk}"
                job_skills= extract_skills_nlp(full_jt)
                match_pct = tfidf_similarity(profile_text, full_jt)

                if show_matching and match_pct < 20:
                    continue

                count += 1
                _, matched_sk, missing_sk = skill_gap(job_skills, my_skills)

                with st.container():
                    st.markdown(job_card_html(
                        title, company or rec_co,
                        req_sk, exp_need or "Not specified",
                        edu_need or "Not specified", posted[:10]
                    ), unsafe_allow_html=True)

                    col1, col2, col3 = st.columns([2,2,1])
                    col1.markdown(f"**NLP Match Score:** `{match_pct:.1f}%`")
                    col2.markdown(f"**Your Skills Match:** {len(matched_sk)}/{len(job_skills)}")

                    if match_pct >= 70:
                        col3.markdown('<span class="badge-g">Strong Match</span>', unsafe_allow_html=True)
                    elif match_pct >= 40:
                        col3.markdown('<span class="badge-y">Partial Match</span>', unsafe_allow_html=True)
                    else:
                        col3.markdown('<span class="badge-r">Low Match</span>', unsafe_allow_html=True)

                    with st.expander("View Full Details"):
                        st.markdown(f"**Description:** {desc}")
                        st.markdown(f"✅ **Your matching skills:** {', '.join(matched_sk) or 'None'}")
                        st.markdown(f"❌ **Skills you are missing:** {', '.join(missing_sk) or 'None'}")
                    st.markdown("---")

            if count == 0:
                st.info("No matching jobs found. Uncheck the filter to see all postings.")

    # ══ TAB 2: RESUME ANALYZER ══════════════════════════════════════════════
    with tab2:
        section("Resume Analyzer & Skill Gap Report")

        c.execute("SELECT id,title,company,required_skills,experience_needed,education_needed FROM job_posts ORDER BY posted_on DESC")
        all_jobs = c.fetchall()

        if not all_jobs:
            st.info("No jobs posted yet — come back when companies add openings.")
        else:
            job_opts = {f"{j[1]} @ {j[2]}": j for j in all_jobs}
            sel = st.selectbox("Select a Job to Analyze Against", list(job_opts.keys()))
            jrow = job_opts[sel]
            _, j_title, j_comp, j_sk_raw, j_exp_raw, j_edu = jrow
            full_jt    = f"{j_title} {j_sk_raw}"
            job_skills = extract_skills_nlp(full_jt)
            exp_nums   = re.findall(r"(\d+)", j_exp_raw or "0")
            req_exp    = int(exp_nums[0]) if exp_nums else 0

            uploaded = st.file_uploader("Upload Your Resume (.pdf or .txt)",
                                        type=["pdf","txt"], key="seeker_resume")

            if uploaded:
                text = extract_text(uploaded)
                if text.strip():
                    overall, tfidf, sk_pct, exp_pct, qual_pct, matched, missing = \
                        composite_score(text, full_jt, job_skills, req_exp, j_edu or "")
                    human_s, ai_risk, verdict, cls, signals = detect_ai_content(text)
                    status = final_status(overall, human_s)

                    st.write("")
                    section("📊 Your Resume Analysis Report")

                    # Top metrics
                    m1,m2,m3,m4,m5 = st.columns(5)
                    m1.metric("Overall Match", f"{overall}%")
                    m2.metric("NLP Similarity", f"{tfidf}%")
                    m3.metric("Skill Match", f"{sk_pct}%")
                    m4.metric("Experience", f"{exp_pct}%")
                    m5.metric("Qualification", f"{qual_pct}%")

                    st.progress(overall / 100)

                    # Resume authenticity
                    st.write("")
                    st.markdown(f'<div class="{cls}">Resume Authenticity: {verdict}</div>',
                                unsafe_allow_html=True)
                    if signals:
                        with st.expander("View Authenticity Details"):
                            for sig in signals:
                                st.write(f"• {sig}")

                    # Shortlist verdict
                    st.write("")
                    if "Shortlisted" in status:
                        st.success(f"🎉 **{status}** — Great match for this role!")
                    elif "Moderately" in status:
                        st.warning(f"⚡ **{status}** — Close! Work on missing skills.")
                    else:
                        st.error(f"❌ **{status}** — Focus on the learning roadmap below.")

                    # Skill breakdown
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("### ✅ Matching Skills")
                        if matched:
                            for sk in matched:
                                st.write(f"• {sk}")
                        else:
                            st.write("None matched.")
                    with col_b:
                        st.markdown("### ❌ Missing Skills")
                        if missing:
                            for sk in missing:
                                st.write(f"• {sk}")
                        else:
                            st.success("You have all required skills!")

                    # Store missing in session for tab 3
                    st.session_state["missing_skills"] = missing
                    st.session_state["overall_match"]  = overall

    # ══ TAB 3: LEARNING ROADMAP ═════════════════════════════════════════════
    with tab3:
        section("📚 Personalised Learning Roadmap")

        missing = st.session_state.get("missing_skills", None)
        overall_m = st.session_state.get("overall_match", None)

        if missing is None:
            # Compute from profile skills vs all jobs
            c.execute("SELECT required_skills FROM job_posts")
            all_req = " ".join(r[0] for r in c.fetchall())
            job_skills_all = extract_skills_nlp(all_req)
            my_skills      = extract_skills_nlp(profile_text)
            _, _, missing  = skill_gap(job_skills_all, my_skills)

        if not missing:
            st.success("🎉 Excellent! Your profile covers all skills in current job postings.")
        else:
            if overall_m is not None:
                st.info(f"Your current match score is **{overall_m}%**. Completing the roadmap below could boost it significantly.")

            st.markdown("### 🗺️ Step-by-Step Learning Plan")
            roadmap = learning_roadmap(missing)

            for week, skill, link in roadmap:
                st.markdown(f"""
                <div class="roadstep">
                    <b>Week {week} — {skill}</b><br>
                    Master <b>{skill}</b> concepts and build at least one hands-on project.<br>
                    <a href="{link}" target="_blank">
                        📖 Start Course on Coursera →
                    </a>
                </div>""", unsafe_allow_html=True)

            st.write("")
            section("🎓 Additional Free Resources")
            free_resources = {
                "Kaggle (Hands-on ML)":        "https://www.kaggle.com/learn",
                "Google ML Crash Course":       "https://developers.google.com/machine-learning/crash-course",
                "MIT OpenCourseWare":           "https://ocw.mit.edu/",
                "fast.ai (Deep Learning)":      "https://www.fast.ai/",
                "CS50 (Harvard Free)":          "https://cs50.harvard.edu/",
                "LinkedIn Learning":            "https://www.linkedin.com/learning/",
                "YouTube – freeCodeCamp":       "https://www.youtube.com/@freecodecamp",
                "edX Free Courses":             "https://www.edx.org/",
            }
            cols = st.columns(2)
            for i, (name_r, url) in enumerate(free_resources.items()):
                cols[i % 2].markdown(f"🔗 [{name_r}]({url})")

            st.write("")
            section("🏆 Top Certifications to Boost Your Profile")
            certs_list = [
                ("AWS Certified Solutions Architect",
                 "https://aws.amazon.com/certification/certified-solutions-architect-associate/"),
                ("Google Professional Data Engineer",
                 "https://cloud.google.com/learn/certification/data-engineer"),
                ("TensorFlow Developer Certificate",
                 "https://www.tensorflow.org/certificate"),
                ("Microsoft Azure Data Scientist",
                 "https://learn.microsoft.com/en-us/certifications/azure-data-scientist/"),
                ("Databricks Certified Associate",
                 "https://www.databricks.com/learn/certification"),
                ("Certified Kubernetes Administrator",
                 "https://www.cncf.io/certification/cka/"),
                ("IBM Data Science Professional Certificate",
                 "https://www.coursera.org/professional-certificates/ibm-data-science"),
                ("Meta Back-End Developer Certificate",
                 "https://www.coursera.org/professional-certificates/meta-back-end-developer"),
            ]
            for cert_name, cert_url in certs_list:
                st.markdown(f"🎖️ [{cert_name}]({cert_url})")

    conn.close()

# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    page = st.session_state.page

    if page == "landing":
        render_landing()
    elif page == "auth":
        render_auth()
    elif page == "recruiter_dash":
        if st.session_state.user:
            render_recruiter_dash()
        else:
            st.session_state.page = "auth"
            st.rerun()
    elif page == "seeker_dash":
        if st.session_state.user:
            render_seeker_dash()
        else:
            st.session_state.page = "auth"
            st.rerun()
    else:
        st.session_state.page = "landing"
        st.rerun()

    st.markdown("---")
    st.caption("SmartHire AI · Advanced NLP-Powered Recruitment Ecosystem")

if __name__ == "__main__":
    main()
