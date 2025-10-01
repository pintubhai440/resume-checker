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

# Custom CSS for a beautiful, dark-themed design
st.markdown("""
<style>
    /* General Card Style */
    .card {
        background-color: #262730;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #444;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .card h5 {
        margin-top: 0;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        font-size: 1.1em;
    }
    
    /* Specific Card Styles */
    .matched {
        background: linear-gradient(to right, #2E4034, #262730);
        border-left: 5px solid #04AA6D;
    }
    .missing {
        background: linear-gradient(to right, #4B3F27, #262730);
        border-left: 5px solid #FFC107;
    }
    .recommendation {
        background: linear-gradient(to right, #1a3a5b, #262730);
        border-left: 5px solid #007bff;
        color: #F0F0F0; /* <<< YEH TEXT COLOR KO THEEK KARTA HAI */
    }

    /* Download Button Container */
    .download-container {
        text-align: center;
        padding: 20px;
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
                
                with st.container():
                    st.markdown('<div class="download-container">', unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download Full Report",
                        data=report_data,
                        file_name="AI_Resume_Analysis_Report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.info("This might be due to a temporary API issue or invalid content. Please try again.")
