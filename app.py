import streamlit as st
import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from collections import Counter

# --- Helper Functions for Resume Analysis (No Changes Here) ---

def get_word_count_status(text):
    """Checks the word count and returns a detailed status message."""
    word_count = len(text.split())
    if word_count < 50:
        return f"⚠️ Too Short ({word_count} words)"
    elif 50 <= word_count <= 1000:
        return f"✅ Optimal Length ({word_count} words)"
    else: # More than 1000 words
        return f"⚠️ Exceeded Max Limit ({word_count} words)"

def get_repetition_status(text):
    """
    Checks for keyword repetition, ignoring common stop words and punctuation,
    to ensure good keyword distribution.
    """
    stop_words = {
        'the', 'in', 'or', 'and', 'a', 'an', 'to', 'is', 'of', 'for', 'with', 'on', 'it', 'i', 'was',
        'are', 'as', 'at', 'be', 'by', 'that', 'this', 'from', 'my', 'we', 'our', 'you', 'your'
    }
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = [word for word in clean_text.split() if word not in stop_words]
    
    if not words:
        return "✅ Good keyword distribution"
    
    word_counts = Counter(words)
    total_words = len(words)
    most_common_word, count = word_counts.most_common(1)[0]
    repetition_percentage = (count / total_words) * 100
    
    if repetition_percentage > 5:
        return f"⚠️ High repetition of '{most_common_word}'"
    return "✅ Good keyword distribution"

# --- UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Resume Checker")
st.title("🚀 AI Resume Checker")
st.write("Get consistent, accurate, and data-driven resume analysis with Gemini. This tool provides a relevance score, skill gap analysis, and more.")

# --- API KEY & MODEL SETUP ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("🤫 Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()
    
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- LAYOUT ---
col1, col2 = st.columns(2, gap="large")
with col1:
    st.header("📄 Job Requirements")
    job_description = st.text_area("Paste the Job Description here", height=350, label_visibility="collapsed", placeholder="Enter the job description...")
with col2:
    st.header("👤 Candidate's Resume")
    resume_text = st.text_area("Paste the Resume Text here", height=350, label_visibility="collapsed", placeholder="Enter the candidate's resume...")

# --- ANALYSIS BUTTON & LOGIC ---
if st.button("Analyze with Gemini AI", use_container_width=True, type="primary"):
    
    if not resume_text or not job_description:
        st.warning("Please provide both the Job Description and the Resume text.")
    else:
        with st.spinner('Gemini is performing a deep analysis... This might take a moment.'):
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro-preview-03-25", # <-- CHANGE 1: Using gemini-pro as requested
                temperature=0,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                },
            )
            
            prompt_template_str = """
            You are a highly advanced AI hiring assistant and a data-driven analyst. Your task is to provide a strict, objective, and consistent analysis of a resume against a job description.

            IMPORTANT CONTEXT:
            - The current date is October 2, 2025. Calculate experience durations based on this date.
            - Base your entire analysis STRICTLY on the text provided in the resume and job description. Do not infer or assume any skills or experiences not explicitly mentioned.

            RESPONSE FORMAT:
            You must provide ONLY a raw JSON response. Do not include any introductory text, explanations, or markdown formatting like ```json. The response must contain the following keys:
            - "relevance_score": An integer (0-100) representing how well the resume's experience and skills align with the job requirements.
            - "skills_match": A percentage string (e.g., "85%") calculated from the required skills found in the resume.
            - "years_experience": A string representing the candidate's total relevant experience (e.g., "3 years", "5+ years"). Be precise.
            - "education_level": A brief alignment description: "High" (meets or exceeds), "Medium" (partially meets), or "Low" (does not meet).
            - "matched_skills": A list of up to 7 skills the candidate possesses that are also mentioned in the job description.
            - "missing_skills": A list of up to 3 critical skills required by the job but absent from the resume.
            - "recommendation_summary": A concise, 2-sentence summary of the candidate's fit, highlighting strengths and weaknesses objectively.
            - "uses_action_verbs": A boolean (true/false). True if the resume uses action verbs (e.g., "Managed", "Developed", "Led").
            - "has_quantifiable_results": A boolean (true/false). True if the resume includes achievements with numbers or metrics (e.g., "increased efficiency by 20%", "managed a budget of $50k").
            - "recommendation_score": An integer (0-100) representing your overall confidence in the candidate, combining all factors.

            Resume: {resume}
            Job Description: {jd}
            """

            prompt = PromptTemplate(input_variables=["resume", "jd"], template=prompt_template_str)
            chain = RunnableSequence(prompt, llm)
            
            response_text = "" # <-- CHANGE 2: Initializing the variable to prevent NameError
            try:
                response = chain.invoke({"resume": resume_text, "jd": job_description})
                response_text = response.content.strip()
                
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                    analysis_result = json.loads(json_text)
                else:
                    raise ValueError("No valid JSON object found in the AI's response.")
                
                word_count_status = get_word_count_status(resume_text)
                repetition_status = get_repetition_status(resume_text)

                st.divider()
                st.header("📊 Analysis Results")

                recommendation_score = analysis_result.get('recommendation_score', 0)
                if recommendation_score >= 75:
                    rec_color = "green"
                    rec_text = "Highly Recommended"
                elif recommendation_score >= 50:
                    rec_color = "orange"
                    rec_text = "Worth Considering"
                else:
                    rec_color = "red"
                    rec_text = "Not a Strong Fit"

                st.subheader(f"Final Verdict: :{rec_color}[{rec_text} ({recommendation_score}%)]")
                st.progress(recommendation_score)
                
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                res_col1.metric("AI Relevance Score", f"{analysis_result.get('relevance_score', 0)}%")
                res_col2.metric("Skills Match", analysis_result.get('skills_match', 'N/A'))
                res_col3.metric("Years' Experience", analysis_result.get('years_experience', 'N/A'))
                res_col4.metric("Education Level", analysis_result.get('education_level', 'N/A'))

                st.subheader("Skills Analysis")
                skill_col1, skill_col2 = st.columns(2)
                with skill_col1:
                    st.success("✅ Matched Skills")
                    st.write(", ".join(analysis_result.get('matched_skills', ["Not found"])))
                with skill_col2:
                    st.warning("❗️ Missing Skills")
                    st.write(", ".join(analysis_result.get('missing_skills', ["None found"])))

                st.subheader("💡 Recommendation")
                st.info(analysis_result.get('recommendation_summary', 'No summary available.'))
                
                st.subheader("Resume Quality Checks")
                
                action_verbs = "✅ Yes" if analysis_result.get('uses_action_verbs') else "⚠️ No"
                quant_results = "✅ Yes" if analysis_result.get('has_quantifiable_results') else "⚠️ No"

                st.markdown("""
                <style>
                .metric-card {
                    background-color: #F0F2F6;
                    border-radius: 10px;
                    padding: 15px;
                    text-align: center;
                    border: 1px solid #E0E0E0;
                }
                .metric-card p.label {
                    font-size: 14px;
                    color: #555;
                    margin-bottom: 5px;
                }
                .metric-card p.value {
                    font-size: 16px;
                    font-weight: bold;
                    color: #333;
                    margin: 0;
                }
                </style>
                """, unsafe_allow_html=True)

                add_col1, add_col2, add_col3, add_col4 = st.columns(4)
                with add_col1:
                    st.markdown(f'<div class="metric-card"><p class="label">Word Count</p><p class="value">{word_count_status}</p></div>', unsafe_allow_html=True)
                with add_col2:
                    st.markdown(f'<div class="metric-card"><p class="label">Keyword Repetition</p><p class="value">{repetition_status}</p></div>', unsafe_allow_html=True)
                with add_col3:
                    st.markdown(f'<div class="metric-card"><p class="label">Uses Action Verbs?</p><p class="value">{action_verbs}</p></div>', unsafe_allow_html=True)
                with add_col4:
                    st.markdown(f'<div class="metric-card"><p class="label">Quantifiable Results?</p><p class="value">{quant_results}</p></div>', unsafe_allow_html=True)

                st.divider()

                report_text = f"""
AI RESUME ANALYSIS REPORT
=========================
FINAL VERDICT: {rec_text} ({recommendation_score}%)
AI RELEVANCE SCORE: {analysis_result.get('relevance_score', 0)}%
SKILLS MATCH: {analysis_result.get('skills_match', 'N/A')}
YEARS' EXPERIENCE: {analysis_result.get('years_experience', 'N/A')}

RECOMMENDATION:
{analysis_result.get('recommendation_summary', '')}

RESUME QUALITY CHECKS:
- Word Count: {word_count_status}
- Repetition: {repetition_status}
- Uses Action Verbs: {"Yes" if analysis_result.get('uses_action_verbs') else "No"}
- Shows Quantifiable Results: {"Yes" if analysis_result.get('has_quantifiable_results') else "No"}

MATCHED SKILLS:
- {', '.join(analysis_result.get('matched_skills', []))}

MISSING SKILLS:
- {', '.join(analysis_result.get('missing_skills', []))}
"""
                st.download_button(
                    label="⬇️ Download Full Report",
                    data=report_text,
                    file_name="resume_analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.text_area("Raw AI Response for debugging:", response_text, height=150)



