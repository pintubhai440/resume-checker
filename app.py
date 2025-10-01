# Zaroori libraries ko import karna
import streamlit as st
import os
import json
import google.generativeai as genai

# --- FUNCTIONS ---

def get_gemini_response(job_desc, resume_txt):
    """
    Calls the Gemini API with the resume and job description to get a JSON analysis.
    Uses gemini-2.5-pro and forces a JSON output for reliability.
    """
    # Initialize the powerful model
    model = genai.GenerativeModel('gemini-2.5-pro')

    # Define the generation config to force JSON output
    generation_config = genai.GenerationConfig(response_mime_type="application/json")

    # Create the detailed prompt for the AI
    full_prompt = f"""
    You are an expert AI hiring assistant with deep expertise in tech and HR.
    Your task is to analyze the following resume against the provided job description.
    Provide a comprehensive analysis in a structured JSON format.

    Here are the details for the JSON output:
    - "relevance_score": An integer (0-100) representing how relevant the resume is to the job.
    - "skills_match": A percentage string (e.g., "85%") of how well the candidate's skills match the job requirements.
    - "years_experience": A string for the candidate's relevant years of experience (e.g., "5+ Years").
    - "education_level": A brief description of educational alignment ("High", "Medium", "Low", or "Not Specified").
    - "matched_skills": A list of up to 7 key skills that match the job description.
    - "missing_skills": A list of up to 3 critical skills mentioned in the job description but missing from the resume.
    - "recommendation_summary": A concise, 2-sentence summary explaining why the candidate is (or is not) a good fit.
    - "uses_action_verbs": A boolean (true/false) indicating if the resume effectively uses action verbs.
    - "has_quantifiable_results": A boolean (true/false) indicating if the resume shows measurable achievements.
    - "recommendation_score": An integer (0-100) representing your overall confidence in recommending this candidate.

    Resume Text: ```{resume_txt}```
    Job Description: ```{job_desc}```
    """
    
    # Call the API
    response = model.generate_content(full_prompt, generation_config=generation_config)
    return json.loads(response.text)

def display_analysis_results(analysis_result):
    """
    Takes the JSON analysis and displays it in a structured Streamlit format.
    """
    st.divider()
    st.header("📊 Analysis Results")

    recommendation_score = analysis_result.get('recommendation_score', 0)
    
    # Determine the verdict based on the score
    if recommendation_score >= 75:
        rec_color, rec_text = "green", "Highly Recommended"
    elif recommendation_score >= 50:
        rec_color, rec_text = "orange", "Worth Considering"
    else:
        rec_color, rec_text = "red", "Not a Strong Fit"

    st.subheader(f"Final Verdict: :{rec_color}[{rec_text}]")
    st.progress(recommendation_score / 100)

    # Display key metrics in columns
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.metric("AI Relevance Score", f"{analysis_result.get('relevance_score', 0)}%")
    res_col2.metric("Skills Match", analysis_result.get('skills_match', 'N/A'))
    res_col3.metric("Years' Experience", analysis_result.get('years_experience', 'N/A'))
    res_col4.metric("Education Level", analysis_result.get('education_level', 'N/A'))

    st.subheader("💡 AI Summary")
    st.write(analysis_result.get('recommendation_summary', "No summary provided."))

    # Display Matched and Missing skills
    skill_col1, skill_col2 = st.columns(2)
    with skill_col1:
        st.markdown("<h5>✅ Matched Skills</h5>", unsafe_allow_html=True)
        for skill in analysis_result.get('matched_skills', []):
            st.markdown(f"- {skill}")

    with skill_col2:
        st.markdown("<h5>❌ Missing Skills</h5>", unsafe_allow_html=True)
        for skill in analysis_result.get('missing_skills', []):
            st.markdown(f"- {skill}")

    # Display Resume Quality Checks
    st.subheader("📝 Resume Quality Checks")
    check_col1, check_col2 = st.columns(2)
    with check_col1:
        uses_verbs = "✅ Yes" if analysis_result.get('uses_action_verbs', False) else "❌ No"
        st.metric("Uses Action Verbs?", uses_verbs)
    with check_col2:
        has_results = "✅ Yes" if analysis_result.get('has_quantifiable_results', False) else "❌ No"
        st.metric("Has Quantifiable Results?", has_results)

# --- UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Resume Checker", page_icon="🚀")
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
                # Call the function to get analysis
                analysis_result = get_gemini_response(job_description, resume_text)
                # Call the function to display the results
                display_analysis_results(analysis_result)
                
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.info("This might be due to a temporary API issue or invalid content. Please try again.")
