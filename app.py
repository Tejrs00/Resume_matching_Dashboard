# app_streamlit_final.py
# Resume Relevance — Final (Accuracy + Pro Analytics + PDF Export)
# Paste this entire file into app_streamlit_final.py and run with Streamlit.

import streamlit as st

# =====================
# Dark Mode + Teal Theme CSS
# =====================
dark_teal_style = '''
<style>
/* Global dark mode background */
body, .main, .block-container {
    background-color: #2B2B2B;
    color: #EAEAEA;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #00BFA5;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #383838;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #00BFA5;
    box-shadow: 0 0 8px rgba(0,191,165,0.4);
}

/* Buttons */
div.stButton > button {
    background-color: #00BFA5;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #009e8c;
}

/* Tables */
table {
    color: #EAEAEA;
}
thead tr th {
    background-color: #00BFA5;
    color: white;
}
tbody tr:nth-child(even) {
    background-color: #333333;
}
</style>
'''
st.markdown(dark_teal_style, unsafe_allow_html=True)

import docx2txt
import PyPDF2
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import io
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# -----------------------
# Page + Theme
# -----------------------
st.set_page_config(page_title="Resume Relevance — Final", layout="wide")
# Use checkbox (more widely supported) rather than st.toggle
dark_mode = st.checkbox("🌙 Dark Mode", value=False)

def inject_css(is_dark: bool):
    if is_dark:
        st.markdown(
            """
            <style>
            body, .stApp { background-color: #111214; color: #e6e6e6 !important; }
            .dataframe, .stDataFrame, table { color: #e6e6e6 !important; background-color: #1e1e1e !important; }
            .resume-snippet { background-color: #1b1b1b; color: #eaeaea; padding: 14px; border-radius: 8px; }
            mark { background: #ffd54f; color: #000000; padding: 0 2px; border-radius: 2px; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            body, .stApp { background-color: #f6f7fb; color: #0f1720 !important; }
            .dataframe, .stDataFrame, table { color: #0f1720 !important; background-color: #ffffff !important; }
            .resume-snippet { background-color: #ffffff; color: #000000; padding: 14px; border-radius: 8px; }
            mark { background: #ffd54f; color: #000000; padding: 0 2px; border-radius: 2px; }
            </style>
            """,
            unsafe_allow_html=True,
        )

inject_css(dark_mode)
st.title("📄 Resume Relevance — Final (Accuracy + Pro Analytics + PDF Export)")

# -----------------------
# File helpers
# -----------------------
def extract_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        pages = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_from_docx(uploaded_file) -> str:
    try:
        # docx2txt accepts a path or a file-like object; uploaded_file is file-like but docx2txt
        # sometimes needs a temporary file — but many environments accept file-like. We'll try file-like.
        return docx2txt.process(uploaded_file)
    except Exception:
        # fallback: read bytes and try to write a temp file (avoiding external libs here)
        try:
            data = uploaded_file.read()
            with open("._tmp_resume.docx", "wb") as f:
                f.write(data)
            txt = docx2txt.process("._tmp_resume.docx")
            return txt
        except Exception:
            return ""

def extract_text_from_txt(uploaded_file) -> str:
    try:
        raw = uploaded_file.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return str(raw)
    except Exception:
        return ""

def read_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    if name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    if name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    return ""

# -----------------------
# Accuracy model helpers
# -----------------------
def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower()).strip()

PHRASES = [
    "object oriented programming", "object-oriented programming", "unit testing", "data structures",
    "rest api", "microservices", "version control", "cloud computing", "problem solving",
    "continuous integration", "continuous delivery", "test driven development", "agile methodology",
    "scrum ceremonies", "front end", "back end", "full stack", "n tier", "n-tier"
]

SYNONYMS = {
    "js": ["javascript", "reactjs"],
    "javascript": ["js", "reactjs"],
    "react": ["reactjs"],
    "node": ["nodejs", "node.js"],
    "node.js": ["node", "nodejs"],
    "oop": ["object oriented programming", "object-oriented programming"],
    "ci": ["continuous integration"],
    "cd": ["continuous delivery"],
    "ml": ["machine learning"],
    "sql": ["mysql", "postgresql", "rdbms"],
    "html": ["hypertext markup language"],
    "css": ["cascading style sheets"],
    "rest": ["rest api"],
    "agile": ["scrum"],
    "git": ["version control"],
}

TECH_WEIGHTS = {
    3: set("""java python javascript typescript react reactjs node node.js nodejs django spring dotnet .net
              sql mysql postgresql mongodb azure aws gcp kubernetes docker terraform kafka salesforce apex visualforce""".split()),
    2: set("""html css rest api microservices oop object oriented programming object-oriented programming unit testing
              agile scrum cicd ci cd jira pytest junit n-tier n tier data structures algorithms""".split()),
}

GENERIC_LOW_VALUE = set("team leadership communication collaboration business stakeholders documentation ability willing quick learner".split())
SECTION_WEIGHTS = {"skills": 0.5, "experience": 0.4, "education": 0.1}

def tokenize_words(text: str):
    return re.findall(r"[a-zA-Z0-9\.\+#\-]+", (text or "").lower())

def extract_jd_terms(jd_text: str):
    jd_norm = normalize_text(jd_text)
    terms = set(tokenize_words(jd_norm))
    present_phrases = [p for p in PHRASES if p in jd_norm]
    for p in present_phrases:
        terms.add(p)
    expanded = set(terms)
    for t in list(terms):
        for canon, syns in SYNONYMS.items():
            if t == canon or t in syns:
                expanded.add(canon)
                expanded.update(syns)
    expanded = {t for t in expanded if t not in GENERIC_LOW_VALUE and len(t) >= 2}
    return sorted(expanded)

def keyword_weight(term: str) -> int:
    t = (term or "").lower()
    if t in TECH_WEIGHTS[3]:
        return 3
    if t in TECH_WEIGHTS[2]:
        return 2
    if t in PHRASES:
        return 2
    return 1

def split_resume_sections(text: str):
    sections = {"skills":"", "experience":"", "education":""}
    blocks = re.split(r"(?im)\n\s*(skills|technical skills|experience|work experience|professional experience|projects|education|academics)\s*[:\-]*\n", text or "")
    if len(blocks) > 1:
        for i in range(1, len(blocks), 2):
            header = (blocks[i] or "").lower()
            content = blocks[i+1] or ""
            if "skill" in header:
                sections["skills"] += " " + content
            elif "experience" in header or "project" in header or "work" in header or "professional" in header:
                sections["experience"] += " " + content
            elif "education" in header or "academic" in header:
                sections["education"] += " " + content
    else:
        sections["experience"] = text or ""
    return sections

def section_contains(section_text: str, term: str) -> bool:
    s = normalize_text(section_text)
    t = (term or "").lower()
    if " " in t:
        return t in s
    if t in s.split():
        return True
    return t in s

def compute_weighted_coverage(jd_terms, resume_text):
    sections = split_resume_sections(resume_text)
    section_scores = {k: 0.0 for k in SECTION_WEIGHTS}
    section_max = {k: 0.0 for k in SECTION_WEIGHTS}
    matched_terms = set()

    for term in jd_terms:
        w = keyword_weight(term)
        for sec, sec_w in SECTION_WEIGHTS.items():
            section_max[sec] += w * sec_w
            found = section_contains(sections.get(sec, ""), term)
            if not found:
                for syn in SYNONYMS.get(term, []):
                    if section_contains(sections.get(sec, ""), syn):
                        found = True
                        break
            if not found:
                for canon, syns in SYNONYMS.items():
                    if term in syns and section_contains(sections.get(sec, ""), canon):
                        found = True
                        break
            if found:
                section_scores[sec] += w * sec_w
                matched_terms.add(term)

    earned = sum(section_scores.values())
    max_possible = sum(section_max.values())
    coverage = 100.0 * earned / max_possible if max_possible > 0 else 0.0
    per_section_pct = {sec: (section_scores[sec] / section_max[sec] * 100.0) if section_max[sec] > 0 else 0.0 for sec in SECTION_WEIGHTS}
    return round(coverage,2), {k: round(v,2) for k,v in per_section_pct.items()}, sections, sorted(matched_terms)

def tfidf_similarity(jd_text, resume_text):
    try:
        vec = TfidfVectorizer()
        X = vec.fit_transform([jd_text or "", resume_text or ""])
        score = cosine_similarity(X[0:1], X[1:2])[0][0]
        return round(score * 100, 2)
    except Exception:
        return 0.0

MUST_HAVE_CANONICAL = {"java","python","javascript","react","node","rest","sql","git","agile","oop","unit testing"}

def must_have_penalty(jd_terms, resume_text):
    present_musts = [m for m in MUST_HAVE_CANONICAL if any(m == t or m in t for t in jd_terms)]
    missing = []
    text_norm = normalize_text(resume_text)
    for m in present_musts:
        present = (m in text_norm) or any(s in text_norm for s in SYNONYMS.get(m, []))
        if not present:
            missing.append(m)
    penalty = min(5 * len(missing), 15)
    return penalty, missing

def composite_score(tfidf_pct, coverage_pct, penalty):
    raw = 0.6 * tfidf_pct + 0.4 * coverage_pct
    final = max(0.0, min(100.0, raw - penalty))
    return round(final,2)

def grade_from_score(score):
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    return "C"

def highlight_with_terms(text, terms):
    if not terms:
        return text
    escaped = [re.escape(t) for t in sorted(terms, key=lambda x: len(x), reverse=True)]
    pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)

# -----------------------
# UI Controls & Inputs
# -----------------------
st.header("1️⃣ Upload Job Description (JD)")
jd_file = st.file_uploader("Upload JD (PDF / DOCX / TXT)", type=["pdf","docx","txt"], key="jd")

st.header("2️⃣ Upload Resumes")
resume_files = st.file_uploader("Upload Resumes (PDF / DOCX / TXT) — multiple allowed", type=["pdf","docx","txt"], accept_multiple_files=True, key="resumes")

show_tables = st.checkbox("📋 Show Tables", value=True)
show_highlight = st.checkbox("🖍 Show Highlighted Snippets", value=False)
shortlist_threshold = st.slider("Minimum Final Score to Shortlist", 0, 100, 75)

# -----------------------
# Main processing
# -----------------------
if jd_file is not None and resume_files:
    jd_text_raw = read_text(jd_file)
    jd_terms = extract_jd_terms(jd_text_raw)

    st.success("✅ Parsed JD and expanded keywords (synonyms & phrases).")
    if show_tables:
        st.write("**JD terms used for matching:**")
        st.write(", ".join(jd_terms))

    rows = []
    per_resume_terms = {}
    must_missing_counter = Counter()
    resume_text_cache = {}

    for f in resume_files:
        rtext = read_text(f)
        resume_text_cache[f.name] = rtext

        coverage_pct, per_section_pct, sections, matched_terms = compute_weighted_coverage(jd_terms, rtext)
        tfidf_pct = tfidf_similarity(jd_text_raw, rtext)
        penalty, missing = must_have_penalty(jd_terms, rtext)
        final = composite_score(tfidf_pct, coverage_pct, penalty)
        grade = grade_from_score(final)

        per_resume_terms[f.name] = matched_terms
        for m in missing:
            must_missing_counter[m] += 1

        rows.append({
            "Resume File": f.name,
            "TF-IDF Similarity (%)": tfidf_pct,
            "Weighted Coverage (%)": coverage_pct,
            "Penalty (Must-Have Missing)": penalty,
            "Final Score (%)": final,
            "Grade": grade,
            "Skills Match (%)": per_section_pct.get("skills",0.0),
            "Experience Match (%)": per_section_pct.get("experience",0.0),
            "Education Match (%)": per_section_pct.get("education",0.0),
            "Matched Terms": ", ".join(matched_terms),
            "Resume Text": rtext
        })

    df = pd.DataFrame(rows).sort_values("Final Score (%)", ascending=False).reset_index(drop=True)

    # Recruiter summary
    st.subheader("📢 Recruiter Summary")
    st.write(f"Total resumes: {len(df)}")
    st.write(f"Top resume: {df.iloc[0]['Resume File']} — {df.iloc[0]['Final Score (%)']}% ({df.iloc[0]['Grade']})")
    st.write(f"Lowest resume: {df.iloc[-1]['Resume File']} — {df.iloc[-1]['Final Score (%)']}% ({df.iloc[-1]['Grade']})")

    # Shortlisting & download
    shortlisted_df = df[df["Final Score (%)"] >= shortlist_threshold][["Resume File","Final Score (%)","Grade"]]
    st.subheader("📥 Shortlisting")
    st.write(f"Shortlisted candidates ({len(shortlisted_df)}):")
    st.dataframe(shortlisted_df)
    st.download_button("Download Shortlist CSV", shortlisted_df.to_csv(index=False), "shortlist_final.csv", "text/csv")
    st.download_button("Download Full Results CSV", df.drop(columns=["Resume Text"]).to_csv(index=False), "results_full.csv", "text/csv")

    # Analytics charts
    
    # =====================================
    # 📊 User-Friendly Resume Analysis Dashboard (Improved)
    # =====================================
    import plotly.express as px
    import plotly.graph_objects as go

    st.subheader("📊 Easy-to-Understand Summary")

    # --- KPI Summary Cards ---
    col1, col2, col3, col4 = st.columns(4)
    total_resumes = len(df)
    shortlisted_count = len(shortlisted_df)
    avg_score = float(df["Final Score (%)"].mean()) if total_resumes > 0 else 0.0
    top_missing = None
    if must_missing_counter:
        top_missing = sorted(must_missing_counter.items(), key=lambda x: -x[1])[0][0]
    with col1:
        st.metric("Total Resumes", total_resumes, help="Number of resumes processed")
    with col2:
        st.metric("Shortlisted", shortlisted_count, help="Number of resumes meeting threshold")
    with col3:
        st.metric("Avg Score", f"{avg_score:.1f}%", help="Average final match score")
    with col4:
        st.metric("Top Missing Skill", top_missing or "—", help="Most common missing must-have skill")

    # --- Real Gauge Chart for Avg Score ---
    st.markdown("#### 🎯 Overall Match Level")
    gauge_color = "red" if avg_score < 50 else "orange" if avg_score < 70 else "green"
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = avg_score,
        title = {"text": "Average Match %"},
        gauge = {
            "axis": {"range": [0, 100]},
            "bar": {"color": gauge_color},
            "steps": [
                {"range": [0, 50], "color": "mistyrose"},
                {"range": [50, 70], "color": "lightyellow"},
                {"range": [70, 100], "color": "lightgreen"}
            ]
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # --- Top 5 Skills Present vs Missing ---
    st.markdown("#### 🧠 Top Skills: Present vs Missing")
    if must_missing_counter:
        missing_df = pd.DataFrame(sorted(must_missing_counter.items(), key=lambda x: -x[1]), columns=["Skill","Missing Count"]).head(5)
        present_counts = []
        for skill in missing_df["Skill"]:
            present_count = sum(df["Extracted Keywords"].apply(lambda kws: skill in kws))
            present_counts.append(present_count)
        skill_df = pd.DataFrame({
            "Skill": missing_df["Skill"],
            "Missing": missing_df["Missing Count"],
            "Present": present_counts
        })
        skill_df = skill_df.melt(id_vars="Skill", value_vars=["Present","Missing"], var_name="Status", value_name="Count")
        fig_skills = px.bar(skill_df, x="Skill", y="Count", color="Status", barmode="group", text="Count")
        st.plotly_chart(fig_skills, use_container_width=True)
    else:
        st.info("No missing must-have skills found.")

    # --- Top 5 Candidates Table with Missing Skills ---
    st.markdown("#### 🏆 Top 5 Candidates")
    top_rows = []
    for _, row in df.sort_values("Final Score (%)", ascending=False).head(5).iterrows():
        name = row["Resume File"]
        matched = set((per_resume_terms.get(name) or []))
        missing_terms = [t for t in jd_terms if t not in matched][:3]
        top_rows.append({
            "Resume File": name,
            "Final Score (%)": row["Final Score (%)"],
            "Grade": row["Grade"],
            "Top Missing Skills": ", ".join(missing_terms) if missing_terms else "—"
        })
    st.table(pd.DataFrame(top_rows))

    # --- Plain-English Insights with Color Codes ---
    st.markdown("#### 💡 Key Insights")
    insights = []
    if shortlisted_count == 0:
        insights.append(("No resumes meet the current threshold — consider lowering it or adjusting must-have skills.", "red"))
    elif shortlisted_count < max(2, total_resumes*0.2):
        insights.append(("Very selective shortlist; consider widening criteria or focusing on priority skills.", "orange"))
    if top_missing:
        insights.append((f"'{top_missing}' is missing from {must_missing_counter[top_missing]} resumes.", "red" if must_missing_counter[top_missing] > total_resumes/2 else "orange"))
    if avg_score < 50:
        insights.append(("Average score is low (<50%), suggesting a major skills gap.", "red"))
    elif avg_score < 70:
        insights.append(("Average score is moderate (50–70%). There’s room for improvement.", "orange"))
    else:
        insights.append(("Average score is strong (>70%). Matches are generally good.", "green"))

    for text, color in insights:
        st.markdown(f"<span style='color:{color};font-weight:bold'>• {text}</span>", unsafe_allow_html=True)

# --------------------
    # PDF Report Export
    # --------------------
    def fig_to_image_reader(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        return ImageReader(buf), buf

    # Prepare images (keep buffers in a list to prevent GC)
    img_bufs = []
    try:
        img1, b1 = fig_to_image_reader(fig1); img_bufs.append(b1)
        img2, b2 = fig_to_image_reader(fig2); img_bufs.append(b2)
        img3, b3 = fig_to_image_reader(fig3); img_bufs.append(b3)
        img4, b4 = fig_to_image_reader(fig4); img_bufs.append(b4)
        img5 = None
        if 'fig5' in locals():
            img5, b5 = fig_to_image_reader(fig5); img_bufs.append(b5)
    except Exception:
        img1 = img2 = img3 = img4 = img5 = None

    def build_pdf_report():
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=landscape(A4))
        width, height = landscape(A4)

        # Header
        c.setFont("Helvetica-Bold", 18)
        c.drawString(2*cm, height - 2*cm, "Resume Relevance Report")
        c.setFont("Helvetica", 10)
        c.drawString(2*cm, height - 2.8*cm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(2*cm, height - 3.6*cm, f"JD terms considered: {len(jd_terms)} | Resumes analyzed: {len(df)} | Shortlisted: {len(shortlisted_df)}")
        # Insert figures if available
        y = height - 5*cm
        try:
            if img1:
                c.drawImage(img1, 2*cm, y - 9*cm, 18*cm, 9*cm, preserveAspectRatio=True, mask='auto')
            if img3:
                c.drawImage(img3, 22*cm, y - 9*cm, 12*cm, 9*cm, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass
        c.showPage()

        # Coverage vs Similarity page
        try:
            if img2:
                c.drawImage(img2, 2*cm, 2*cm, width - 4*cm, height - 4*cm, preserveAspectRatio=True, mask='auto')
            c.showPage()
        except Exception:
            pass

        # Section averages + top skills
        try:
            if img4:
                c.drawImage(img4, 2*cm, 2*cm, width - 4*cm, height - 4*cm, preserveAspectRatio=True, mask='auto')
            c.showPage()
        except Exception:
            pass

        # Tables page (compact)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, height - 2*cm, "Top Results (first rows)")
        c.setFont("Helvetica", 9)
        text_y = height - 3*cm
        # render top 30 rows
        table_csv = df[["Resume File","Final Score (%)","Grade","Weighted Coverage (%)","TF-IDF Similarity (%)"]].head(30).to_csv(index=False)
        for line in table_csv.splitlines():
            c.drawString(2*cm, text_y, line)
            text_y -= 0.5*cm
            if text_y < 2*cm:
                c.showPage()
                text_y = height - 2*cm
        c.save()
        buf.seek(0)
        return buf

    if st.button("📥 Download PDF Report"):
        pdf_buf = build_pdf_report()
        st.download_button("Download the PDF report", pdf_buf.getvalue(), file_name=f"resume_relevance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf")

else:
    st.info("Upload a Job Description and at least one Resume to start analysis.")


    # =============================
    # 📄 Visually Styled PDF Export
    # =============================
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.units import inch
    from datetime import datetime
    import tempfile

    def _save_plotly_png(fig, filename):
        try:
            fig.write_image(filename, engine="kaleido", scale=2)
            return True
        except Exception as e:
            return False

    def build_pdf_report(df, shortlisted_df, must_missing_counter, jd_terms, per_resume_terms, output_path):
        # KPI values
        total_resumes = len(df)
        shortlisted_count = len(shortlisted_df)
        avg_score = float(df["Final Score (%)"].mean()) if total_resumes > 0 else 0.0
        top_missing = None
        if must_missing_counter:
            top_missing = sorted(must_missing_counter.items(), key=lambda x: -x[1])[0][0]

        # Recreate charts from the dashboard for export
        import plotly.express as px
        import plotly.graph_objects as go

        # Gauge
        gauge_color = "red" if avg_score < 50 else "orange" if avg_score < 70 else "green"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_score,
            title={"text": "Average Match %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 50], "color": "mistyrose"},
                    {"range": [50, 70], "color": "lightyellow"},
                    {"range": [70, 100], "color": "lightgreen"}
                ]
            }
        ))

        gauge_png = os.path.join(tempfile.gettempdir(), "gauge.png")
        _save_plotly_png(fig_gauge, gauge_png)

        # Skills Present vs Missing
        skills_png = None
        if must_missing_counter:
            missing_df = pd.DataFrame(sorted(must_missing_counter.items(), key=lambda x: -x[1]), columns=["Skill","Missing Count"]).head(5)
            present_counts = []
            for skill in missing_df["Skill"]:
                present_count = sum(df["Extracted Keywords"].apply(lambda kws: skill in kws))
                present_counts.append(present_count)
            skill_df = pd.DataFrame({"Skill": missing_df["Skill"], "Missing": missing_df["Missing Count"], "Present": present_counts})
            skill_df = skill_df.melt(id_vars="Skill", value_vars=["Present","Missing"], var_name="Status", value_name="Count")
            fig_skills = px.bar(skill_df, x="Skill", y="Count", color="Status", barmode="group", text="Count")
            skills_png = os.path.join(tempfile.gettempdir(), "skills.png")
            _save_plotly_png(fig_skills, skills_png)

        # Top 5 Candidates table data with missing skills
        top_rows = []
        for _, row in df.sort_values("Final Score (%)", ascending=False).head(5).iterrows():
            name = row["Resume File"]
            matched = set((per_resume_terms.get(name) or []))
            missing_terms = [t for t in jd_terms if t not in matched][:3]
            top_rows.append([
                str(name),
                f"{float(row['Final Score (%)']):.1f}%",
                str(row.get("Grade","")),
                ", ".join(missing_terms) if missing_terms else "—"
            ])

        # Build PDF
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("title", parent=styles["Title"], textColor=colors.HexColor("#1f77b4"))
        h_style = ParagraphStyle("h", parent=styles["Heading2"], textColor=colors.HexColor("#333333"))
        normal = styles["BodyText"]

        doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []

        # Cover
        story.append(Paragraph("Resume–Job Match Report", title_style))
        story.append(Spacer(1, 8))
        story.append(Paragraph(datetime.now().strftime("%B %d, %Y"), normal))
        story.append(Spacer(1, 16))

        # KPIs
        story.append(Paragraph("Summary", h_style))
        story.append(Spacer(1, 6))
        kpi_data = [
            ["Total Resumes", str(total_resumes), "Shortlisted", str(shortlisted_count)],
            ["Average Score", f"{avg_score:.1f}%", "Top Missing Skill", top_missing or "—"],
        ]
        kpi_table = Table(kpi_data, colWidths=[120,120,120,120])
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f6ff")),
            ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#1f77b4")),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("FONTNAME", (0,0), (-1,-1), "Helvetica")
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 16))

        # Charts
        story.append(Paragraph("Overall Match Level", h_style))
        if os.path.exists(gauge_png):
            story.append(RLImage(gauge_png, width=5.5*inch, height=3*inch))
        story.append(Spacer(1, 12))

        if skills_png and os.path.exists(skills_png):
            story.append(Paragraph("Top Skills: Present vs Missing", h_style))
            story.append(RLImage(skills_png, width=5.5*inch, height=3*inch))
            story.append(Spacer(1, 12))

        # Top candidates
        story.append(Paragraph("Top 5 Candidates", h_style))
        if top_rows:
            tbl = Table([["Resume File","Final Score","Grade","Top Missing Skills"]] + top_rows, colWidths=[160,80,60,160])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8f5e9")),
                ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#2e7d32")),
                ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
            ]))
            story.append(tbl)
        story.append(Spacer(1, 12))

        # Insights
        story.append(Paragraph("Insights", h_style))
        insights = []
        if shortlisted_count == 0:
            insights.append("No resumes meet the threshold — lower it or adjust must-have skills.")
        elif shortlisted_count < max(2, total_resumes*0.2):
            insights.append("Shortlist is very selective; consider widening criteria.")
        if top_missing:
            miss_count = must_missing_counter[top_missing]
            tone = "widespread" if miss_count > total_resumes/2 else "moderate"
            insights.append(f"'{top_missing}' is {tone}ly missing — consider emphasizing it in screening.")
        if avg_score < 50:
            insights.append("Average score < 50% — major skills gap present.")
        elif avg_score < 70:
            insights.append("Average score 50–70% — moderate match; refine requirements or upskill.")
        else:
            insights.append("Average score > 70% — overall match is strong.")

        for i in insights:
            story.append(Paragraph(f"• {i}", normal))

        doc.build(story)

    # Streamlit button to export and download
    st.markdown("### 📄 Export")
    if st.button("Export PDF Report (All Resumes)"):
        import tempfile
        tmp_pdf = os.path.join(tempfile.gettempdir(), "resume_match_report.pdf")
        build_pdf_report(df, shortlisted_df, must_missing_counter, jd_terms, per_resume_terms, tmp_pdf)
        with open(tmp_pdf, "rb") as f:
            pdf_bytes = f.read()
        st.download_button("⬇️ Download Report", data=pdf_bytes, file_name="resume_match_report.pdf", mime="application/pdf")
