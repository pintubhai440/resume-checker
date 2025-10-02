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
    """Analyze resume word count and provide status"""
    word_count = len(text.split())
    if word_count < 150:
        return f"⚠️ Too Short ({word_count} words)"
    elif 150 <= word_count <= 800:
        return f"✅ Optimal Length ({word_count} words)"
    else:
        return f"⚠️ Too Long ({word_count} words)"

def get_repetition_status(text):
    """Analyze keyword repetition in resume with an improved stop-word list."""
    stop_words = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
        'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do',
        'does', 'doing', 'don', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have',
        'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into',
        'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of',
        'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same',
        'she', 'should', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves',
        'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very',
        'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you',
        'your', 'yours', 'yourself', 'yourselves', 'experience', 'work', 'project', 'company', 'team', 'role', 'worked',
        'responsibilities', 'development', 'used', 'using'
    }
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = [word for word in clean_text.split() if word not in stop_words and not word.isdigit()]

    if len(words) < 20: # Not enough content to analyze
        return "✅ Good keyword distribution"
        
    word_counts = Counter(words)
    if not word_counts:
        return "✅ Good keyword distribution"

    total_words = len(words)
    most_common_word, count = word_counts.most_common(1)[0]
    repetition_percentage = (count / total_words) * 100
    
    if repetition_percentage > 4.5:
        return f"⚠️ High repetition of '{most_common_word.title()}'"
    return "✅ Good keyword distribution"

def clean_json_response(response_text):
    """Extracts and cleans a JSON object from a string."""
    # Find the start and end of the JSON object
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not json_match:
        return None
    
    json_text = json_match.group(0)
    # Remove control characters and trailing commas
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
        with st.spinner('🔍 Gemini is performing comprehensive analysis... This might take 20-30 seconds.'):
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-thinking-exp",
                temperature=0.1,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # --- A Single, More Robust Prompt ---
            try:
                analysis_prompt_template = """
                You are an expert HR Analyst. Your task is to analyze a candidate's RESUME against a JOB DESCRIPTION.
                Provide a detailed evaluation in a structured JSON format.

                **ANALYSIS INSTRUCTIONS:**
                1.  **Skills Analysis:**
                    - First, identify all the skills explicitly required in the JOB DESCRIPTION.
                    - Then, check the RESUME to see which of those required skills are present. Use semantic understanding (e.g., "Analytical Thinking" should match "Analytical Skills").
                    - Create two lists: `matched_skills` and `missing_skills`.
                2.  **Skills Match Percentage (`skills_match`):**
                    - Calculate this as: (Number of Matched Skills / Total Number of Required Skills) * 100. Round to the nearest whole number. If no skills are required, this should be 100.
                3.  **Years of Experience (`years_experience`):**
                    - Calculate the total years of professional experience from the resume.
                    - If specific work dates are not present, estimate the experience based on the graduation date. For a recent graduate (e.g., passed out in the last 1-2 years), it should be "Fresher" or "Less than 1 year". A 2020 pass-out would have around 4-5 years of experience by late 2025.
                4.  **Education Level (`education_level`):**
                    - Compare the candidate's education with the job requirements.
                    - Classify as 'High' (perfect match), 'Medium' (related field or acceptable alternative), or 'Low' (does not meet requirements).
                5.  **Relevance and Recommendation Scores:**
                    - `relevance_score` (0-100): Overall match considering skills, experience, and education.
                    - `recommendation_score` (0-100): Your final confidence in recommending the candidate for an interview.
                6.  **Recommendation Summary (`recommendation_summary`):**
                    - Write a concise, actionable summary for the hiring manager, explaining your recommendation.

                **RESUME:**
                {resume}

                **JOB DESCRIPTION:**
                {jd}

                **RETURN ONLY a raw JSON object in the following exact format:**
                {{
                    "relevance_score": 85,
                    "skills_match": 75,
                    "years_experience": "5 years",
                    "education_level": "High",
                    "matched_skills": ["Python", "SQL", "Data Analysis", "Problem Solving"],
                    "missing_skills": ["Machine Learning", "Cloud Computing", "Tableau"],
                    "recommendation_summary": "The candidate is a strong fit with solid foundational skills in Python and SQL. While they lack advanced ML expertise, they are a quick learner and recommended for an interview.",
                    "uses_action_verbs": true,
                    "has_quantifiable_results": false,
                    "recommendation_score": 80
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
                else:
                    rec_color, rec_text = "red", "Not Recommended"

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
                
                report_text = f"""
ADVANCED RESUME ANALYSIS REPORT
================================

FINAL ASSESSMENT: {rec_text} ({recommendation_score}%)

KEY METRICS:
- AI Relevance Score: {analysis_result.get('relevance_score', 'N/A')}%
- Skills Match: {analysis_result.get('skills_match', 'N/A')}%
- Years of Experience: {analysis_result.get('years_experience', 'N/A')}
- Education Level: {analysis_result.get('education_level', 'N/A')}

PROFESSIONAL ASSESSMENT:
{analysis_result.get('recommendation_summary', '')}

RESUME QUALITY ANALYSIS:
- Word Count: {word_count_status}
- Keyword Repetition: {repetition_status}
- Uses Action Verbs: {"Yes" if analysis_result.get('uses_action_verbs') else "No"}
- Shows Quantifiable Results: {"Yes" if analysis_result.get('has_quantifiable_results') else "No"}

MATCHED SKILLS:
{chr(10).join(['• ' + skill for skill in analysis_result.get('matched_skills', ['None identified'])])}

MISSING CRITICAL SKILLS:
{chr(10).join(['• ' + skill for skill in analysis_result.get('missing_skills', ['None identified'])])}

---
Generated by Advanced AI Resume Checker
                """
                
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
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    <p>🔍 <strong>Advanced AI Resume Checker</strong> | Uses Gemini AI for precise resume analysis</p>
    <p>Provides realistic scoring based on actual content matching between resume and job requirements</p>
</div>
""", unsafe_allow_html=True)
