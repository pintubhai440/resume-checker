# advanced_resume_checker_v2.py
# Updated Streamlit app: Improved UI, deterministic scoring, caching, safer LLM usage.
import streamlit as st
import hashlib
import requests
import re
import io
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import fitz  # PyMuPDF
import json
import time

# LangChain / Google Gemini imports are optional — only used if API key present
try:
    from langchain import LLMChain, PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# ----------------------------
# Helper dataclass for result
# ----------------------------
@dataclass
class AnalysisResult:
    relevance_score: int
    skills_match: str
    years_experience: str
    education_level: str
    matched_skills: List[str]
    missing_skills: List[str]
    uses_action_verbs: bool
    has_quantifiable_results: bool
    recommendation_summary: str
    recommendation_score: int

# ----------------------------
# Utility functions
# ----------------------------

ACTION_VERBS = {"led","managed","developed","implemented","built","designed","improved","optimized","created","deployed","orchestrated","owned","spearheaded"}
EXPERIENCE_PATTERNS = [
    r'(\d+)\s*\+\s*years',
    r'(\d+)\s+years',
    r'(\d+)\s*yr',
    r'(\d+)-year',
    r'(\d+)\s+yrs'
]

def extract_years_of_experience(text: str) -> Optional[int]:
    text_low = text.lower()
    for pat in EXPERIENCE_PATTERNS:
        m = re.search(pat, text_low)
        if m:
            try:
                return int(m.group(1))
            except:
                continue
    # fallback: estimate from job titles + dates (basic)
    # look for YYYY - YYYY or YYYY–YYYY
    years = re.findall(r'(\b19\d{2}|\b20\d{2})', text)
    if len(years) >= 2:
        try:
            earliest = int(min(years))
            latest = int(max(years))
            est = max(0, latest - earliest)
            return est
        except:
            return None
    return None

def extract_skills_from_text(text: str, skill_vocab: List[str]) -> List[str]:
    found = []
    text_low = text.lower()
    for s in skill_vocab:
        if re.search(r'\b' + re.escape(s.lower()) + r'\b', text_low):
            found.append(s)
    return found

def detect_quantifiable_results(text: str) -> bool:
    # If text contains numbers followed by % or words like "increase", "reduced", "saved"
    if re.search(r'\d+%|\d+\s+percent|\b(increase|decrease|reduc|saved|boost|improv)\w*\b', text.lower()):
        return True
    # raw numbers near achievements
    if re.search(r'\b\d{2,}\b', text):
        return True
    return False

def uses_action_verbs(text: str) -> bool:
    text_low = text.lower()
    for v in ACTION_VERBS:
        if re.search(r'\b' + re.escape(v) + r'\b', text_low):
            return True
    return False

# ----------------------------
# Deterministic scoring function
# ----------------------------
def deterministic_resume_score(job_desc: str, resume_text: str, top_k_skills: int = 8) -> AnalysisResult:
    """
    Rule-based deterministic analyzer:
      - Counts keyword matches
      - Extracts explicit skills
      - Produces stable scoring based on weights
    """
    # 1. Build skill vocabulary from job description (simple heuristic: pick words that look like tech/skill tokens)
    # Simpler: look for commas/slash-separated tokens or common separators
    job_low = job_desc.lower()
    # naive skill candidates from job description: words longer than 2 letters appearing in jobdesc that are not stopwords
    # But better: pick words separated by commas or '•' or listed bullets
    skill_candidates = []
    # find comma-separated phrases of length <=4 words
    for part in re.split(r'[\n•\-•]', job_desc):
        part = part.strip()
        if 2 <= len(part.split()) <= 4 and len(part) < 80:
            skill_candidates.append(part)
    # fallback tokens (single words) from jobdesc
    tokens = re.findall(r'\b[a-zA-Z+#\.\-]{2,30}\b', job_low)
    # common stopwords to ignore
    stop = {"and","or","with","experience","years","knowledge","work","ability","team","strong","excellent","proven","able","will"}
    token_candidates = []
    for t in tokens:
        if t in stop: continue
        # drop common English small words
        if len(t) <= 2: continue
        token_candidates.append(t)
    # combine and dedupe, prioritize comma-phrases first
    skill_vocab = []
    for s in skill_candidates + token_candidates:
        normalized = s.strip()
        if normalized and normalized.lower() not in set(x.lower() for x in skill_vocab):
            skill_vocab.append(normalized)
    # Keep limit
    skill_vocab = skill_vocab[:60]

    matched_skills = extract_skills_from_text(resume_text, skill_vocab)
    # determine missing skills (top 8 from job vocab that are not in matched)
    missing_skills = []
    for s in skill_vocab[:12]:
        if s not in matched_skills:
            missing_skills.append(s)
        if len(missing_skills) >= 3:
            break

    # Scores
    skill_match_pct = int((len(matched_skills) / max(1, min(len(skill_vocab), 1))) * 100) if len(skill_vocab) == 0 else int((len(matched_skills)/len(skill_vocab))*100)
    # but if skill_vocab is huge, clamp sensibly: use top-10 denominator
    denom = max(1, min(len(skill_vocab), 10))
    skill_match_pct = int((len(matched_skills)/denom) * 100)
    if skill_match_pct > 100: skill_match_pct = 100

    years = extract_years_of_experience(resume_text)
    years_str = f"{years}+ Years" if years is not None else "Not Specified"

    # relevance = weighted sum: skills 55%, experience 25%, quantifiable & action verbs 10% each
    w_skills = 0.55
    w_exp = 0.25
    w_quant = 0.10
    w_action = 0.10

    # normalize experience: cap at 10 years
    exp_score = 0
    if years is not None:
        exp_score = min(years, 10) / 10 * 100
    else:
        exp_score = 50  # unknown -> neutral

    quant_flag = detect_quantifiable_results(resume_text)
    action_flag = uses_action_verbs(resume_text)

    relevance = int(w_skills * skill_match_pct + w_exp * exp_score + w_quant * (100 if quant_flag else 0) + w_action * (100 if action_flag else 0))
    relevance = max(0, min(100, relevance))

    # recommendation_score = slightly more conservative than relevance
    recommendation_score = max(0, min(100, relevance - (len(missing_skills)*5)))

    recommendation_summary = (
        f"Rule-based analysis: Resume matches {skill_match_pct}% of top job skills. "
        f"Detected experience: {years_str}. "
        f"{'Shows' if quant_flag else 'Lacks'} quantifiable results and "
        f"{'uses' if action_flag else 'does not use'} strong action verbs. "
        f"Recommendation score {recommendation_score}/100."
    )

    # education_level simple heuristic
    education_level = "Not Specified"
    if re.search(r'\b(bachelor|b\.sc|bsc|b\.e|btech|b\.tech|master|m\.sc|msc|mtech|m\.tech|phd)\b', resume_text.lower()):
        education_level = "High"

    # keep matched_skills trimmed to top_k
    matched_skills = matched_skills[:top_k_skills]

    return AnalysisResult(
        relevance_score=relevance,
        skills_match=f"{skill_match_pct}%",
        years_experience=years_str,
        education_level=education_level,
        matched_skills=matched_skills or ["N/A"],
        missing_skills=missing_skills or ["N/A"],
        uses_action_verbs=action_flag,
        has_quantifiable_results=quant_flag,
        recommendation_summary=recommendation_summary,
        recommendation_score=recommendation_score
    )

# ----------------------------
# GitHub resume fetch (cached)
# ----------------------------
@st.cache_data(ttl=3600)
def fetch_resume_from_github(github_url: str) -> Optional[str]:
    try:
        github_url = github_url.rstrip("/")
        parts = github_url.split("/")
        if len(parts) < 5:
            return None
        username, repo = parts[3], parts[4]
        api_url = f"https://api.github.com/repos/{username}/{repo}/contents"
        r = requests.get(api_url, timeout=10)
        r.raise_for_status()
        files = r.json()
        target_names = {"resume.pdf","resume.md","cv.pdf","cv.md","readme.md","resume.txt"}
        file_info = None
        for f in files:
            if f.get("name","").lower() in target_names:
                file_info = f
                break
        if not file_info:
            # search recursively (first folder)
            for f in files:
                if f.get("type")== "dir":
                    r2 = requests.get(f["url"], timeout=10)
                    r2.raise_for_status()
                    for ff in r2.json():
                        if ff.get("name","").lower() in target_names:
                            file_info = ff
                            break
                    if file_info: break
        if not file_info:
            return None
        download_url = file_info.get("download_url")
        if not download_url:
            return None
        file_r = requests.get(download_url, timeout=10)
        file_r.raise_for_status()
        if file_info['name'].lower().endswith('.pdf'):
            with fitz.open(stream=file_r.content, filetype="pdf") as doc:
                txt = "".join(page.get_text() for page in doc)
                return txt
        else:
            return file_r.text
    except Exception as e:
        st.error(f"Error fetching GitHub resume: {e}")
        return None

# ----------------------------
# Optional: polish summary with model (deterministic if temp=0.0)
# ----------------------------
def polish_summary_with_model(job_desc: str, resume_text: str, base_summary: str) -> str:
    """
    Use Gemini via langchain to produce a concise recommendation summary.
    This function runs only if API key configured and LangChain available.
    Uses temperature=0.0 for determinism.
    """
    # only run if available
    if not LANGCHAIN_AVAILABLE:
        return base_summary + " (Model not available; using deterministic summary.)"
    try:
        # You must set GOOGLE_API_KEY in Streamlit secrets for this to work
        model = ChatGoogleGenerativeAI(model="gemini-2o", temperature=0.0, max_retries=2, timeout=60)
        prompt = PromptTemplate(
            input_variables=["job_description","resume_text","base_summary"],
            template=(
                "You are a concise technical recruiter assistant. Given the job description and a deterministic base summary, "
                "produce a 2-3 sentence professional recommendation summary that is factual and does not hallucinate. "
                "\n\nJOB DESCRIPTION:\n{job_description}\n\nRESUME:\n{resume_text}\n\nBASE SUMMARY:\n{base_summary}\n\nOutput only the final polished summary."
            )
        )
        chain = LLMChain(llm=model, prompt=prompt)
        # Provide truncated resume/job to avoid huge prompts
        out = chain.run({
            "job_description": job_desc[:4000],
            "resume_text": resume_text[:4000],
            "base_summary": base_summary
        })
        return out.strip()
    except Exception as e:
        # don't fail; return base
        return base_summary + f" (Polish failed: {str(e)})"

# ----------------------------
# Report generator
# ----------------------------
def generate_report_text_struct(res: AnalysisResult, job_desc: str, resume_text: str) -> str:
    obj = asdict(res)
    report = [
        "=== AI Resume Analysis Report ===",
        time.strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "Key Metrics:",
        f" - Relevance Score: {obj['relevance_score']}",
        f" - Skills Match: {obj['skills_match']}",
        f" - Years Experience: {obj['years_experience']}",
        f" - Education Level: {obj['education_level']}",
        "",
        "Matched Skills: " + ", ".join(obj['matched_skills']),
        "Missing Skills: " + ", ".join(obj['missing_skills']),
        "",
        "Uses Action Verbs: " + ("Yes" if obj['uses_action_verbs'] else "No"),
        "Has Quantifiable Results: " + ("Yes" if obj['has_quantifiable_results'] else "No"),
        "",
        "Recommendation Summary:",
        obj['recommendation_summary'],
        "",
        "----------",
        "JOB DESCRIPTION (truncated to 4000 chars):",
        job_desc[:4000],
        "",
        "RESUME (truncated to 4000 chars):",
        resume_text[:4000],
        "",
    ]
    return "\n".join(report)

# ----------------------------
# Streamlit UI
# ----------------------------
def setup_styles():
    st.set_page_config(layout="wide", page_title="Advanced AI Resume Checker", page_icon="🚀")
    st.markdown("""
    <style>
    .header {display:flex; align-items:center; gap:16px}
    .card {background-color: #f7f9fb; border-radius:12px; padding:16px; box-shadow: 0 6px 18px rgba(16,24,40,0.06);}
    .metric {background:linear-gradient(90deg,#ffffff,#f1f7ff); padding:12px; border-radius:10px; text-align:center}
    .small-muted {color:#6b7280; font-size:0.95rem}
    </style>
    """, unsafe_allow_html=True)

def main():
    setup_styles()
    st.markdown("<div class='header'><h1>🚀 Advanced AI Resume Checker</h1></div>", unsafe_allow_html=True)
    st.caption("Deterministic scoring + optional model polishing. For consistent hackathon results.")

    # secrets check
    api_ready = ("GOOGLE_API_KEY" in st.secrets) and LANGCHAIN_AVAILABLE

    with st.expander("How this works (click)"):
        st.write("""
            - First a **deterministic, rule-based** scorer computes stable metrics (guaranteed reproducible).
            - Optionally (if API key present) we call the model with **temperature=0.0** to *polish* the final summary.
            - Results are cached by input-hash so repeated submissions with same inputs are fast and identical.
        """)

    left, right = st.columns([1,1])
    with left:
        st.subheader("📋 Job Description")
        job_desc = st.text_area("", height=260, placeholder="Paste full job description here...")
    with right:
        st.subheader("📄 Resume Source")
        input_method = st.radio("", options=["Paste Resume Text", "Fetch from GitHub"], index=0, horizontal=True)
        resume_text = ""
        if input_method == "Paste Resume Text":
            resume_text = st.text_area("", height=260, placeholder="Paste full resume text here...")
        else:
            github_url = st.text_input("Public GitHub repo URL (example: https://github.com/username/repo)")
            if github_url:
                with st.spinner("Fetching resume from GitHub..."):
                    fetched = fetch_resume_from_github(github_url.strip())
                    if fetched:
                        st.success("✅ Resume fetched from GitHub (parsed).")
                        resume_text = st.text_area("Preview (editable)", value=fetched, height=260)
                    else:
                        st.warning("No resume file found in repo root or first folder. Try pasting or provide direct file URL.")

    st.markdown("---")
    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        analyze_btn = st.button("✨ Run Analysis", use_container_width=True)
    with col_b:
        st.write("")
        st.write("")
        st.info("Deterministic core scoring — same inputs => same outputs.")
    with col_c:
        if api_ready:
            st.success("Model available for polishing (temperature=0.0).")
        else:
            st.warning("No model key or LangChain not installed — using deterministic-only mode.")

    if analyze_btn:
        if not job_desc or not resume_text:
            st.error("Please provide both job description and resume content.")
            st.stop()

        # compute input hash
        current_hash = hashlib.md5((job_desc + resume_text).encode("utf-8")).hexdigest()
        # cached display
        if "last_inputs_hash" not in st.session_state:
            st.session_state.last_inputs_hash = ""
        if "analysis_cache" not in st.session_state:
            st.session_state.analysis_cache = {}

        if st.session_state.last_inputs_hash == current_hash and current_hash in st.session_state.analysis_cache:
            st.success("🔁 Same inputs detected — returning cached analysis.")
            res_obj = st.session_state.analysis_cache[current_hash]
        else:
            with st.spinner("Analyzing (deterministic engine)..."):
                res_obj = deterministic_resume_score(job_desc, resume_text)
                # optional polish
                if api_ready:
                    try:
                        polished = polish_summary_with_model(job_desc, resume_text, res_obj.recommendation_summary)
                        res_obj.recommendation_summary = polished
                    except Exception as e:
                        # keep base summary
                        res_obj.recommendation_summary = res_obj.recommendation_summary + f" (Polish error: {e})"
                # cache
                st.session_state.analysis_cache[current_hash] = res_obj
                st.session_state.last_inputs_hash = current_hash
                st.success("✅ Analysis complete.")

        # Render results UI
        st.markdown("### 📊 Summary")
        verdict_color = "green" if res_obj.recommendation_score >= 80 else ("orange" if res_obj.recommendation_score >= 60 else "red")
        verdict_text = "Highly Recommended" if res_obj.recommendation_score >= 80 else ("Worth Considering" if res_obj.recommendation_score >= 60 else "Not a Strong Fit")
        st.markdown(f"**Recommendation:** <span style='color:{verdict_color}; font-weight:600'>{verdict_text} ({res_obj.recommendation_score}%)</span>", unsafe_allow_html=True)
        st.progress(res_obj.recommendation_score / 100)

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div class='metric'><strong>AI Relevance</strong><div style='font-size:22px'>{res_obj.relevance_score}%</div></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric'><strong>Skills Match</strong><div style='font-size:22px'>{res_obj.skills_match}</div></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric'><strong>Experience</strong><div style='font-size:22px'>{res_obj.years_experience}</div></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='metric'><strong>Education</strong><div style='font-size:22px'>{res_obj.education_level}</div></div>", unsafe_allow_html=True)

        st.markdown("### 🛠️ Skills & Quality")
        leftc, rightc = st.columns(2)
        leftc.markdown("**Matched Skills**")
        leftc.write(", ".join(res_obj.matched_skills))
        leftc.markdown("**Action Verbs**")
        leftc.write("✅ Effectively Used" if res_obj.uses_action_verbs else "❌ Needs Improvement")
        rightc.markdown("**Missing Skills (critical)**")
        rightc.write(", ".join(res_obj.missing_skills))
        rightc.markdown("**Quantifiable Results**")
        rightc.write("✅ Well Demonstrated" if res_obj.has_quantifiable_results else "❌ Lacking Metrics")

        st.markdown("---")
        st.markdown("### 💡 Recommendation Summary")
        st.info(res_obj.recommendation_summary)

        # Downloadable report
        report_text = generate_report_text_struct(res_obj, job_desc, resume_text)
        st.download_button("📥 Download Full Analysis (TXT)", data=report_text, file_name="resume_analysis_report.txt", mime="text/plain")

if __name__ == "__main__":
    main()
