# Zaroori libraries ko import karna
import streamlit as st
import json
import google.generativeai as genai

# --- FUNCTIONS ---

def get_gemini_response(job_desc, resume_txt):
    """
    Calls the Gemini API with the resume and job description to get a JSON analysis.
    Uses gemini-2.5-pro and forces a JSON output for reliability.
    """
    model = genai.GenerativeModel('gemini-2.5-pro')
    generation_config = genai.GenerationConfig(response_mime_type="application/json")
    full_prompt = f"""
    You are an expert AI hiring assistant with deep expertise in tech and HR.
    Analyze the following resume against the provided job description.
    Provide a comprehensive analysis in a structured JSON format.

    JSON Keys:
    - "relevance_score": integer (0-100)
    - "skills_match": string (e.g., "85%")
    - "years_experience": string (e.g., "5+ Years")
    - "education_level": string ("High", "Medium", "Low", "Not Specified")
    - "matched_skills": list of up to 7 matching skills
    - "missing_skills": list of up to 3 critical missing skills
    - "recommendation_summary": 2-sentence summary
    - "uses_action_verbs": boolean
    - "has_quantifiable_results": boolean
    - "recommendation_score": integer (0-100)

    Resume Text: ```{resume_txt}```
    Job Description: ```{job_desc}```
    """
    response = model.generate_content(full_prompt, generation_config=generation_config)
    return json.loads(response.text)

def generate_report_text(analysis_result):
    """
    Generates a downloadable .txt report from the analysis results.
    """
    report_lines = []
    score = analysis_result.get('recommendation_score', 0)
    
    if score >= 75: verdict = "Highly Recommended"
    elif score >= 50: verdict = "Worth Considering"
    else: verdict = "Not a Strong Fit"

    report_lines.append("🚀 AI Resume Analysis Report 🚀")
    report_lines.append("="*30)
    report_lines.append(f"FINAL VERDICT: {verdict} ({score}/100)")
    report_lines.append("\n💡 AI SUMMARY:")
    report_lines.append(analysis_result.get('recommendation_summary', "N/A"))
    report_lines.append("\n📊 KEY METRICS:")
    report_lines.append(f"- AI Relevance Score: {analysis_result.get('relevance_score', 0)}%")
    report_lines.append(f"- Skills Match: {analysis_result.get('skills_match', 'N/A')}")
    report_lines.append(f"- Relevant Experience: {analysis_result.get('years_experience', 'N/A')}")
    
    report_lines.append("\n✅ MATCHED SKILLS:")
    report_lines.append(", ".join(analysis_result.get('matched_skills', ["None"])))
    
    report_lines.append("\n❌ SKILLS TO IMPROVE (Missing):")
    report_lines.append(", ".join(analysis_result.get('missing_skills', ["None"])))
    
    report_lines.append("\n📝 RESUME QUALITY:")
    report_lines.append(f"- Uses Action Verbs: {'Yes' if analysis_result.get('uses_action_verbs') else 'No'}")
    report_lines.append(f"- Has Quantifiable Results: {'Yes' if analysis_result.get('has_quantifiable_results') else 'No'}")
    
    return "\n".join(report_lines)

# --- UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Resume Checker", page_icon="🚀")

# Custom CSS for a clean, simple, and readable design
st.markdown("""
<style>
    /* Basic card structure that works with Streamlit's theme */
    .card {
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #444; /* Uses a neutral border color */
    }
    .card h5 {
        margin: 0;
        padding: 0;
        font-size: 1.1em;
        display: flex;
        align-items: center;
    }
    .card p {
        padding-top: 10px;
        margin-bottom: 0;
    }
    
    /* Colored left border for visual distinction, NO background color */
    .matched {
        border-left: 5px solid #04AA6D;
    }
    .missing {
        border-left: 5px solid #FFC107;
    }
    .recommendation {
        border-left: 5px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)


st.title("🚀 AI Resume Checker")
st.write("Analyze a resume against a job description to get instant, powerful insights powered by **Gemini 2.5 Pro**.")

# --- API KEY SETUP ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (FileNotFoundError, KeyError):
    st.error("🤫 Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()

# --- LAYOUT ---
col1, col2 = st.columns(2, gap="large")
with col1:
    st.header("📄 Job Requirements")
    job_description = st.text_area("Job Description", height=350, label_visibility="collapsed", placeholder="Paste the job description here...", key="job_desc")
with col2:
    st.header("👤 Resume Content")
    resume_text = st.text_area("Paste Resume Text", height=350, label_visibility="collapsed", placeholder="Paste the candidate's resume here...", key="resume_text")

# --- ANALYSIS BUTTON & LOGIC ---
if st.button("Analyze with Gemini AI", use_container_width=True, type="primary"):
    if not resume_text or not job_description:
        st.warning("Please provide both the Job Description and the Resume text.")
    else:
        with st.spinner('Gemini 2.5 Pro is performing a deep analysis... This might take a moment.'):
            try:
                analysis_result = get_gemini_response(job_description, resume_text)
                
                # --- DISPLAY RESULTS ---
                st.divider()
                st.header("📊 Analysis Results")

                score = analysis_result.get('recommendation_score', 0)
                if score >= 75: color, text = "green", "Highly Recommended"
                elif score >= 50: color, text = "orange", "Worth Considering"
                else: color, text = "red", "Not a Strong Fit"

                st.subheader(f"Final Verdict: :{color}[{text}]")
                st.progress(score / 100)
                
                # --- KEY METRICS (FEATURE ADDED BACK) ---
                st.markdown("### Key Metrics")
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                res_col1.metric("AI Relevance Score", f"{analysis_result.get('relevance_score', 0)}%")
                res_col2.metric("Skills Match", analysis_result.get('skills_match', 'N/A'))
                res_col3.metric("Years' Experience", analysis_result.get('years_experience', 'N/A'))
                res_col4.metric("Education Level", analysis_result.get('education_level', 'N/A'))
                # --- END OF ADDED FEATURE ---

                st.markdown("### Skills Analysis")
                
                # Matched Skills Card
                st.markdown(f"""
                <div class="card matched">
                    <h5>✅ Matched Skills</h5>
                    <p>{', '.join(analysis_result.get('matched_skills', ['N/A']))}</p>
                </div>
                """, unsafe_allow_html=True)

                # Missing Skills Card
                st.markdown(f"""
                <div class="card missing">
                    <h5>❌ Missing Skills</h5>
                    <p>{', '.join(analysis_result.get('missing_skills', ['N/A']))}</p>
                </div>
                """, unsafe_allow_html=True)

                # Recommendation Box
                st.markdown("### 💡 Recommendation")
                st.markdown(f"""
                <div class="card recommendation">
                    <p>{analysis_result.get('recommendation_summary', "No summary provided.")}</p>
                </div>
                """, unsafe_allow_html=True)

                # --- DOWNLOAD BUTTON ---
                st.divider()
                report_data = generate_report_text(analysis_result)
                st.download_button(
                    label="📥 Download Full Report",
                    data=report_data,
                    file_name="AI_Resume_Analysis_Report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.info("This might be due to a temporary API issue or invalid content. Please try again.")
