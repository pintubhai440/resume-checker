# ======================================================================================
# 🚀 AI RESUME CHECKER - THE DEFINITIVE & COMPLETE CODE v5.0
# Author: Gemini (in collaboration with the user)
# Description: This is the final, complete, and excellent version of the application.
#              It combines a stable AI backend (LangChain), the user's preferred UI,
#              and robust features for a professional-grade experience.
# ======================================================================================

# --- 1. IMPORT ALL REQUIRED LIBRARIES ---
import streamlit as st
import requests
import fitz  # PyMuPDF
from typing import List, Optional

# LangChain and Pydantic for a robust, stable, and structured AI interaction
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError

# --- 2. DEFINE THE STRICT DATA STRUCTURE FOR THE AI (THE 'BRAIN'S BLUEPRINT) ---
# This Pydantic model is the secret to making our app reliable. LangChain will
# force the AI to return a response that perfectly matches this structure.
class ResumeAnalysis(BaseModel):
    relevance_score: int = Field(description="Relevance of resume to the job (0-100).")
    skills_match: str = Field(description="Percentage string of skills match (e.g., '85%').")
    years_experience: str = Field(description="Candidate's relevant years of experience (e.g., '5+ Years').")
    education_level: str = Field(description="Alignment of education ('High', 'Medium', 'Low', 'Not Specified').")
    matched_skills: List[str] = Field(description="A list of key matching skills.")
    missing_skills: List[str] = Field(description="A list of up to 3 critical missing skills.")
    uses_action_verbs: bool = Field(description="True if the resume effectively uses strong action verbs.")
    has_quantifiable_results: bool = Field(description="True if the resume shows measurable achievements.")
    recommendation_summary: str = Field(description="A concise, 2-3 sentence expert summary for the final recommendation.")
    recommendation_score: int = Field(description="Overall final recommendation score for the candidate (0-100).")

# --- 3. CORE LOGIC FUNCTIONS ---

def get_stable_gemini_analysis(job_desc: str, resume_text: str) -> Optional[dict]:
    """
    This is the main AI function. It uses LangChain with a very low temperature
    to ensure the results are stable and consistent.
    """
    try:
        # Step 1: Initialize the AI Model with low temperature for stability
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.1, # <<< YEH RESULT KO STABLE KARTA HAI
            convert_system_message_to_human=True
        )

        # Step 2: Create the Pydantic Output Parser
        parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)

        # Step 3: Create the detailed Prompt Template
        prompt_template = PromptTemplate(
            template="""
            You are a highly analytical and experienced Senior Technical Recruitment Manager.
            Your task is to conduct an in-depth, unbiased analysis of a candidate's resume against a job description.
            Your analysis must be critical, fair, and based ONLY on the text provided.

            {format_instructions}

            Here is the data for your analysis:
            JOB DESCRIPTION: ```{job_description}```
            RESUME TEXT: ```{resume_text}```
            """,
            input_variables=["job_description", "resume_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Step 4: Create and Invoke the LangChain Chain
        chain = prompt_template | model | parser
        response = chain.invoke({"job_description": job_desc, "resume_text": resume_text})
        
        # Convert the Pydantic object to a standard Python dictionary before returning
        return response.model_dump()

    except Exception as e:
        st.error(f"An error occurred during AI analysis. Please check your API key and input. Error: {e}")
        return None

def fetch_resume_from_github(github_url: str) -> Optional[str]:
    """
    Fetches and extracts text from a resume file (PDF or MD) in a GitHub repository.
    """
    try:
        parts = github_url.strip("/").split("/")
        if len(parts) < 2:
            st.error("Invalid GitHub URL. Please use the format: https://github.com/username/repo")
            return None
        username, repo = parts[-2], parts[-1]
        
        api_url = f"https://api.github.com/repos/{username}/{repo}/contents/"
        response = requests.get(api_url)
        response.raise_for_status()
        
        repo_files = response.json()
        resume_filenames = ["resume.pdf", "resume.md", "README.md"]
        resume_file_info = None

        for filename in resume_filenames:
            for file_info in repo_files:
                if file_info['name'].lower() == filename:
                    resume_file_info = file_info
                    break
            if resume_file_info:
                break
        
        if not resume_file_info:
            st.warning("Could not find a standard resume file (resume.pdf, resume.md, or README.md).")
            return None

        file_response = requests.get(resume_file_info['download_url'])
        file_response.raise_for_status()

        if resume_file_info['name'].lower().endswith('.pdf'):
            with fitz.open(stream=file_response.content, filetype="pdf") as doc:
                return "".join(page.get_text() for page in doc)
        else:
            return file_response.text
    except Exception as e:
        st.error(f"Error fetching from GitHub: {e}")
        return None

def generate_report_text(analysis: dict) -> str:
    """
    Generates a downloadable .txt report from the analysis results.
    """
    score = analysis.get('recommendation_score', 0)
    if score >= 75: verdict = "Highly Recommended"
    elif score >= 50: verdict = "Worth Considering"
    else: verdict = "Not a Strong Fit"

    return f"""
    AI RESUME ANALYSIS REPORT
    ===========================
    FINAL VERDICT: {verdict} ({score}%)
    
    EXECUTIVE SUMMARY:
    {analysis.get('recommendation_summary', 'N/A')}
    
    KEY METRICS:
    - AI Relevance Score: {analysis.get('relevance_score', 0)}%
    - Skills Match: {analysis.get('skills_match', 'N/A')}
    - Years of Experience: {analysis.get('years_experience', 'N/A')}
    - Education Level: {analysis.get('education_level', 'N/A')}
    
    SKILLS ANALYSIS:
    - Matched Skills: {', '.join(analysis.get('matched_skills', ['N/A']))}
    - Missing Skills: {', '.join(analysis.get('missing_skills', ['N/A']))}
    
    RESUME QUALITY:
    - Uses Strong Action Verbs: {'Yes' if analysis.get('uses_action_verbs') else 'No'}
    - Has Quantifiable Results: {'Yes' if analysis.get('has_quantifiable_results') else 'No'}
    """

# --- 4. UI AND APPLICATION FLOW ---

def setup_page_and_styles():
    """
    Sets up the Streamlit page configuration and injects the user's preferred CSS.
    """
    st.set_page_config(layout="wide", page_title="Stable AI Resume Checker", page_icon="🚀")
    # --- HERE IS YOUR PREFERRED DARK CARD DESIGN ---
    st.markdown("""
    <style>
        .card {
            background-color: #262730; /* Dark background */
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid #444;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            color: #FAFAFA; /* Bright text color for readability */
        }
        .card h5 {
            margin-top: 0;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            font-size: 1.2em;
            color: #FFFFFF; /* White heading color */
        }
        .card.matched {
            background-color: rgba(4, 170, 109, 0.2); /* Green tint */
            border-left: 5px solid #04AA6D;
        }
        .card.missing {
            background-color: rgba(255, 193, 7, 0.2); /* Yellow tint */
            border-left: 5px solid #FFC107;
        }
        .card.recommendation {
            background-color: rgba(0, 123, 255, 0.2); /* Blue tint */
            border-left: 5px solid #007bff;
        }
    </style>
    """, unsafe_allow_html=True)

def render_results(analysis_result: dict):
    """
    Displays the analysis results in a structured and visually appealing format.
    """
    st.divider()
    st.header("📊 Analysis Results")

    score = analysis_result.get('recommendation_score', 0)
    if score >= 75: color, text = "green", "Highly Recommended"
    elif score >= 50: color, text = "orange", "Worth Considering"
    else: color, text = "red", "Not a Strong Fit"
    st.subheader(f"Final Verdict: :{color}[{text} ({score}%)]")
    st.progress(score / 100)
    
    st.markdown("### Key Metrics")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("AI Relevance Score", f"{analysis_result.get('relevance_score', 0)}%")
    m_col2.metric("Skills Match", analysis_result.get('skills_match', 'N/A'))
    m_col3.metric("Years' Experience", analysis_result.get('years_experience', 'N/A'))
    m_col4.metric("Education Level", analysis_result.get('education_level', 'N/A'))

    st.markdown("### Skills Analysis")
    st.markdown(f"<div class='card matched'><h5>✅ Matched Skills</h5><p>{', '.join(analysis_result.get('matched_skills', ['N/A']))}</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card missing'><h5>❌ Missing Skills</h5><p>{', '.join(analysis_result.get('missing_skills', ['N/A']))}</p></div>", unsafe_allow_html=True)

    st.markdown("### 📝 Resume Quality")
    q_col1, q_col2 = st.columns(2)
    q_col1.metric("Uses Strong Action Verbs?", "✅ Yes" if analysis_result.get('uses_action_verbs') else "❌ No")
    q_col2.metric("Has Quantifiable Results?", "✅ Yes" if analysis_result.get('has_quantifiable_results') else "❌ No")

    st.markdown("### 💡 Recommendation")
    st.markdown(f"<div class='card recommendation'><p>{analysis_result.get('recommendation_summary', 'N/A')}</p></div>", unsafe_allow_html=True)

    st.divider()
    report_data = generate_report_text(analysis_result)
    st.download_button(label="📥 Download Full Report", data=report_data, file_name="AI_Resume_Analysis_Report.txt", mime="text/plain", use_container_width=True)


def main():
    """
    The main function that runs the Streamlit application.
    """
    setup_page_and_styles()
    
    # Initialize session state to prevent ghost results on refresh
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    st.title("🚀 Stable & Styled AI Resume Checker")
    st.write("Using Gemini 2.5 Pro and LangChain for consistent, expert-level analysis.")

    # API Key check
    try:
        # LangChain automatically uses the GOOGLE_API_KEY environment variable/secret.
        if "GOOGLE_API_KEY" not in st.secrets:
            raise Exception("API Key not found")
    except Exception:
        st.error("🤫 Google API Key not found. Please add it to your Streamlit secrets.")
        st.stop()
    
    # --- INPUT SECTION ---
    job_desc = st.text_area("📋 Paste the full Job Description here", height=250)
    st.write("---")
    input_method = st.radio("Choose Resume Source:", ("Paste Text", "From GitHub URL"), horizontal=True)
    
    resume_text = None
    if input_method == "Paste Text":
        resume_text = st.text_area("✍️ Paste the full Resume text here", height=300)
    else:
        github_url = st.text_input("🔗 Enter public GitHub Repository URL", placeholder="https://github.com/username/repo-name")
        if github_url:
            with st.spinner(f"Fetching resume..."):
                resume_text = fetch_resume_from_github(github_url)
            if resume_text:
                st.success("Successfully fetched resume!")
                st.expander("Preview Fetched Resume").text(resume_text[:1000] + "...")

    # --- ANALYSIS TRIGGER & LOGIC ---
    if st.button("✨ Analyze Now ✨", use_container_width=True, type="primary"):
        # Validate inputs before calling the AI
        if not job_desc or not resume_text:
            st.warning("Please provide both Job Description and Resume content.")
        elif len(job_desc) < 50 or len(resume_text) < 50:
             st.warning("Input text is too short for a meaningful analysis. Please provide full content.")
        else:
            with st.spinner("🤖 Gemini Pro is performing a deep, stable analysis via LangChain..."):
                st.session_state.analysis_result = get_stable_gemini_analysis(job_desc, resume_text)

    # --- DISPLAY RESULTS ---
    # Only show results if they exist in the session state
    if st.session_state.analysis_result:
        render_results(st.session_state.analysis_result)

# --- APPLICATION ENTRY POINT ---
if __name__ == "__main__":
    main()

