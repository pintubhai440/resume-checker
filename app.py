# Zaroori libraries ko import karna
import streamlit as st
import json
import google.generativeai as genai

# --- FUNCTIONS ---

def get_gemini_response(job_desc, resume_txt):
    """
    Calls the Gemini API with a low temperature for consistent results.
    """
    model = genai.GenerativeModel('gemini-2.5-pro')
    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        temperature=0.2 
    )
    full_prompt = f"""
    You are an expert AI hiring assistant... [Previous detailed prompt remains the same]
    ...
    Resume Text: ```{resume_txt}```
    Job Description: ```{job_desc}```
    """
    # (The full prompt is kept the same as the last correct version)
    response = model.generate_content(full_prompt, generation_config=generation_config)
    return json.loads(response.text)

def generate_report_text(analysis_result):
    """
    Generates a downloadable .txt report from the analysis results.
    """
    # (This function remains the same as the last correct version)
    report_lines = []
    score = analysis_result.get('recommendation_score', 0)
    if score >= 75: verdict = "Highly Recommended"
    elif score >= 50: verdict = "Worth Considering"
    else: verdict = "Not a Strong Fit"
    report_lines.append(f"FINAL VERDICT: {verdict} ({score}/100)")
    # ... rest of the function is the same
    return "\n".join(report_lines)

# --- NEW: Function to clear results when input text changes ---
def clear_old_results():
    if 'analysis_result' in st.session_state:
        del st.session_state['analysis_result']

# --- UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Resume Checker", page_icon="🚀")

# (The CSS part remains the same as the last correct version)
st.markdown("""<style>...</style>""", unsafe_allow_html=True)

st.title("🚀 AI Resume Checker")
st.write("Analyze a resume against a job description...")

# --- API KEY SETUP ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (FileNotFoundError, KeyError):
    st.error("🤫 Google API Key not found...")
    st.stop()

# --- LAYOUT with on_change callback to clear old results ---
col1, col2 = st.columns(2, gap="large")
with col1:
    st.header("📄 Job Requirements")
    job_description = st.text_area("...", height=350, label_visibility="collapsed", key="job_desc", on_change=clear_old_results)
with col2:
    st.header("👤 Resume Content")
    resume_text = st.text_area("...", height=350, label_visibility="collapsed", key="resume_text", on_change=clear_old_results)

# --- ANALYSIS BUTTON & LOGIC ---
if st.button("Analyze with Gemini AI", use_container_width=True, type="primary"):
    if not resume_text or not job_description:
        st.warning("Please provide both the Job Description and the Resume text.")
    else:
        with st.spinner('Gemini 2.5 Pro is performing a deep analysis...'):
            try:
                # Save the new result in the session state
                st.session_state.analysis_result = get_gemini_response(job_description, resume_text)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                if 'analysis_result' in st.session_state:
                    del st.session_state['analysis_result']

# --- DISPLAY RESULTS ---
# Check if a result exists in the session state before displaying it
if 'analysis_result' in st.session_state and st.session_state.analysis_result is not None:
    analysis_result = st.session_state.analysis_result
    
    st.divider()
    st.header("📊 Analysis Results")

    score = analysis_result.get('recommendation_score', 0)
    if score >= 75: color, text = "green", "Highly Recommended"
    elif score >= 50: color, text = "orange", "Worth Considering"
    else: color, text = "red", "Not a Strong Fit"

    st.subheader(f"Final Verdict: :{color}[{text} ({score}%)]")
    st.progress(score / 100)
    
    # (The rest of the display logic for metrics, skills, recommendation, and download button remains the same)
    st.markdown("### Key Metrics")
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.metric("AI Relevance Score", f"{analysis_result.get('relevance_score', 0)}%")
    res_col2.metric("Skills Match", analysis_result.get('skills_match', 'N/A'))
    res_col3.metric("Years' Experience", analysis_result.get('years_experience', 'N/A'))
    res_col4.metric("Education Level", analysis_result.get('education_level', 'N/A'))
    # ... and so on for the rest of the display cards ...

    st.divider()
    report_data = generate_report_text(analysis_result)
    st.download_button(label="📥 Download Full Report", data=report_data, file_name="AI_Resume_Analysis_Report.txt", mime="text/plain", use_container_width=True)
