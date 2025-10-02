# Required libraries are imported for the application
import streamlit as st
import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.prompts import PromptTemplate
from collections import Counter
import time

# --- Helper Functions for Resume Quality Analysis ---

def get_word_count_status(text):
    """
    Analyze resume word count with context for freshers.
    A fresher/intern resume is often shorter.
    """
    word_count = len(text.split())
    # CHANGED: Lowered the threshold. For a fresher, 250+ words is good. 400+ is for experienced roles.
    if word_count < 250:
        return f"⚠️ Too Short ({word_count} words)"
    elif 250 <= word_count <= 600:
        return f"✅ Optimal Length ({word_count} words)"
    else:
        return f"⚠️ Too Long ({word_count} words)"

def get_repetition_status(text):
    """Analyze keyword repetition. The goal is to check for overuse of words."""
    stop_words = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
        'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do',
        'does', 'doing', 'don', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have',
        'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into',
        'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of',
        'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ourselves', 'out', 'over', 'own', 's', 'same',
        'she', 'should', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves',
        'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very',
        'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you',
        'your', 'yours', 'yourself', 'yourselves', 'experience', 'work', 'project', 'company', 'team', 'role', 'worked',
        'responsibilities', 'development', 'used', 'using', 'responsible'
    }
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = [word for word in clean_text.split() if word not in stop_words and not word.isdigit()]

    if len(words) < 20:
        return "✅ Low Repetition"
        
    word_counts = Counter(words)
    if not word_counts:
        return "✅ Low Repetition"

    total_words = len(words)
    most_common_word, count = word_counts.most_common(1)[0]
    repetition_percentage = (count / total_words) * 100
    
    if repetition_percentage > 4.5:
        return f"⚠️ High repetition of '{most_common_word.title()}'"
    return "✅ Low Repetition"

def clean_json_response(response_text):
    """Extracts and cleans a JSON object from a string."""
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not json_match:
        return None
    json_text = json_match.group(0)
    json_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_text)
    json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
    return json_text

# --- UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Resume Checker", page_icon="🚀")
st.title("🚀 Advanced AI Resume Checker")
st.write("Get consistent, accurate, and data-driven resume analysis with Gemini. This tool provides precise relevance score, skill gap analysis, and detailed evaluation.")

# --- API KEY & MODEL SETUP ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except (FileNotFoundError, KeyError):
    st.error("🤫 Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()

# --- LAYOUT ---
col1, col2 = st.columns(2, gap="large")
with col1:
    st.header("📄 Job Requirements")
    job_description = st.text_area("Paste the Job Description here", height=350, label_visibility="collapsed", placeholder="Enter the complete job description with required skills, qualifications, and experience...")
with col2:
    st.header("👤 Candidate's Resume")
    resume_text = st.text_area("Paste the Resume Text here", height=350, label_visibility="collapsed", placeholder="Enter the complete resume text including education, skills, experience, projects...")

# --- ANALYSIS BUTTON & LOGIC ---
if st.button("Analyze with Gemini AI", use_container_width=True, type="primary"):
    if not resume_text.strip() or not job_description.strip():
        st.warning("⚠️ Please provide both the Job Description and the Resume text.")
    else:
        with st.spinner('🔍 Gemini is performing a nuanced analysis... This might take 20-30 seconds.'):
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite", 
                temperature=0.1,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            try:
                # FIXED: The entire prompt is re-engineered for nuance and accuracy.
                analysis_prompt_template = """
                You are a highly intelligent and nuanced Senior Technical Recruiter. Your task is to analyze a candidate's RESUME against a JOB DESCRIPTION that may contain one or more roles. Your analysis must be balanced, recognizing strong fits even if they aren't a 100% match for every listed skill.

                **YOUR STEP-BY-STEP ANALYSIS PROCESS:**

                1.  **Identify Roles:** First, analyze the JOB DESCRIPTION. Identify if it contains distinct roles (e.g., 'Data Scientist Intern', 'Data Engineer Intern').

                2.  **Differentiate Skills for EACH Role:** For EACH role you identify, create two lists of skills:
                    * **Core Requirements:** These are the absolute must-have skills mentioned for that specific role (e.g., Python, SQL, Spark for a Data Engineer).
                    * **Advantageous Skills:** These are 'good-to-have' skills or tools mentioned as a plus (e.g., Databricks, specific visualization tools like Tableau).

                3.  **Evaluate Candidate Against EACH Role:** Analyze the RESUME and compare it against the 'Core' and 'Advantageous' skill lists for EACH identified role.

                4.  **Determine Best Fit and Score:**
                    * Identify the single role for which the candidate is the **best fit**.
                    * The `skills_match` percentage MUST be calculated based on the match with the **Core Requirements** of their BEST FIT role.
                    * The `recommendation_score` should be heavily weighted on this core skills match. A candidate matching >80% of CORE skills for a role is a strong candidate (recommendation > 80), even if they miss some advantageous skills.
                    * An average candidate might match some core skills (e.g., Python, SQL) but miss others, resulting in a 40-60% score.
                    * A poor candidate will miss most core skills, resulting in a score < 30%.

                5.  **Identify Missing Skills:** List only the **Core Requirements** that the candidate is missing for their best-fit role. Do not list advantageous skills here unless they are critically important.

                **JOB DESCRIPTION:**
                {jd}

                **RESUME:**
                {resume}

                **RETURN ONLY a raw JSON object in the following exact format. Be very precise with scores.**
                {{
                    "relevance_score": 92,
                    "skills_match": 90,
                    "years_experience": "Fresher",
                    "education_level": "High",
                    "matched_skills": ["Python", "SQL", "Spark", "Generative AI", "Computer Vision", "NLP", "Tableau", "Power BI", "Pandas"],
                    "missing_skills": ["Hadoop"],
                    "recommendation_summary": "This is an excellent candidate and a strong fit for the Data Science Intern role. They possess nearly all core requirements, including hands-on project experience in Generative AI, CV, and data engineering. Highly recommended for an interview.",
                    "uses_action_verbs": true,
                    "has_quantifiable_results": true,
                    "recommendation_score": 95
                }}
                """
                analysis_prompt = PromptTemplate.from_template(analysis_prompt_template)
                analysis_chain = analysis_prompt | llm

                response = analysis_chain.invoke({"resume": resume_text, "jd": job_description})
                response_text = response.content

                cleaned_json = clean_json_response(response_text)
                if not cleaned_json:
                    st.error("❌ AI response format error. Could not extract JSON data.")
                    st.text_area("Raw AI Response for debugging:", response_text)
                    st.stop()

                analysis_result = json.loads(cleaned_json)

                # --- DISPLAY RESULTS ---
                word_count_status = get_word_count_status(resume_text)
                repetition_status = get_repetition_status(resume_text)

                st.divider()
                st.header("📊 Detailed Analysis Results")

                recommendation_score = analysis_result.get('recommendation_score', 0)
                
                if recommendation_score >= 80:
                    rec_color, rec_text = "green", "Highly Recommended"
                elif recommendation_score >= 60:
                    rec_color, rec_text = "orange", "Worth Considering"
                elif recommendation_score >= 40:
                    rec_color, rec_text = "red", "Not Recommended"
                else:
                    rec_color, rec_text = "red", "Strongly Not Recommended"

                st.subheader(f"Final Verdict: :{rec_color}[{rec_text} ({recommendation_score}%)]")
                st.progress(recommendation_score / 100)

                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                with res_col1:
                    st.metric("AI Relevance Score", f"{analysis_result.get('relevance_score', 0)}%")
                with res_col2:
                    st.metric("Skills Match", f"{analysis_result.get('skills_match', '0')}%")
                with res_col3:
                    st.metric("Years' Experience", analysis_result.get('years_experience', 'N/A'))
                with res_col4:
                    st.metric("Education Level", analysis_result.get('education_level', 'N/A'))

                st.subheader("🔧 Skills Analysis")
                skill_col1, skill_col2 = st.columns(2)
                
                st.markdown("""
                <style>
                .skill-badge { display: inline-block; padding: 6px 12px; margin: 4px; font-size: 0.9em; font-weight: 500; border-radius: 15px; text-align: center; }
                .matched-skill { background-color: #E0F2E9; color: #0D6938; border: 1px solid #A3D4B6; }
                .missing-skill { background-color: #FFF3D4; color: #B47D00; border: 1px solid #FFDDA0; }
                </style>
                """, unsafe_allow_html=True)

                with skill_col1:
                    st.success("✅ Matched Skills")
                    matched_skills = analysis_result.get('matched_skills', [])
                    if matched_skills:
                        skills_html = "".join([f'<span class="skill-badge matched-skill">{skill}</span>' for skill in matched_skills])
                        st.markdown(f"<div style='line-height: 2.0;'>{skills_html}</div>", unsafe_allow_html=True)
                    else:
                        st.write("No matching skills found.")
                
                with skill_col2:
                    st.warning("❗️ Critical Missing Skills")
                    missing_skills = analysis_result.get('missing_skills', [])
                    if missing_skills:
                        skills_html = "".join([f'<span class="skill-badge missing-skill">{skill}</span>' for skill in missing_skills])
                        st.markdown(f"<div style='line-height: 2.0;'>{skills_html}</div>", unsafe_allow_html=True)
                    else:
                        st.write("No major skill gaps identified.")

                st.subheader("💡 Professional Assessment")
                st.info(analysis_result.get('recommendation_summary', 'No analysis available.'))

                st.subheader("📝 Resume Quality Analysis")
                action_verbs = "✅ Yes" if analysis_result.get('uses_action_verbs') else "❌ No"
                quant_results = "✅ Yes" if analysis_result.get('has_quantifiable_results') else "❌ No"
                
                st.markdown("""
                <style>
                .metric-card { background-color: #F8F9FA; border-radius: 10px; padding: 15px; text-align: center; border: 1px solid #E0E0E0; margin: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
                .metric-card p.label { font-size: 14px; color: #555; margin-bottom: 5px; font-weight: 500; }
                .metric-card p.value { font-size: 16px; font-weight: bold; color: #333; margin: 0; }
                </style>
                """, unsafe_allow_html=True)

                quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
                with quality_col1:
                    st.markdown(f'<div class="metric-card"><p class="label">Word Count</p><p class="value">{word_count_status}</p></div>', unsafe_allow_html=True)
                with quality_col2:
                    st.markdown(f'<div class="metric-card"><p class="label">Keyword Repetition</p><p class="value">{repetition_status}</p></div>', unsafe_allow_html=True)
                with quality_col3:
                    st.markdown(f'<div class="metric-card"><p class="label">Action Verbs</p><p class="value">{action_verbs}</p></div>', unsafe_allow_html=True)
                with quality_col4:
                    st.markdown(f'<div class="metric-card"><p class="label">Quantifiable Results</p><p class="value">{quant_results}</p></div>', unsafe_allow_html=True)

                st.divider()
                
                report_text = f"""...""" # Report text logic is unchanged
                
                st.download_button(
                    label="📥 Download Comprehensive Report",
                    data=report_text,
                    file_name="detailed_resume_analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            except json.JSONDecodeError as e:
                st.error(f"❌ JSON parsing error: {str(e)}")
                st.text_area("Raw AI Response for debugging:", response_text, height=200)
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
                st.text_area("Raw AI Response for debugging:", response_text, height=200)

# Add footer with information
st.divider()
st.markdown("""...""") # Footer is unchanged
