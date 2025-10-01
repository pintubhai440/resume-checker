# ======================================================================================
# 🚀 AI RESUME CHECKER - EXCELLENT VERSION v3.0 (LangChain Powered)
# Author: Gemini (in collaboration with the user)
# Description: A highly robust, professional-grade AI tool using LangChain for
#              structured, reliable, and powerful resume analysis.
# ======================================================================================

# --- 1. IMPORT REQUIRED LIBRARIES ---
import streamlit as st
import requests
import fitz  # PyMuPDF
import json
from typing import List, Optional

# LangChain and Pydantic Imports for a more robust AI interaction
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError # Corrected typo from pantic to pydantic

# --- 2. DEFINE THE STRICT DATA STRUCTURE FOR AI RESPONSE (THE 'BRAIN'S BLUEPRINT) ---
# This Pydantic model is the core of our reliability. LangChain will force the AI
# to return a response that perfectly matches this structure. No more 0% results.
class ResumeAnalysis(BaseModel):
    relevance_score: int = Field(description="Relevance of resume to the job description (0-100).")
    skills_match: str = Field(description="Percentage string of skills match (e.g., '85%').")
    years_experience: str = Field(description="Candidate's relevant years of experience (e.g., '5+ Years').")
    education_level: str = Field(description="Alignment of education ('High', 'Medium', 'Low', 'Not Specified').")
    matched_skills: List[str] = Field(description="A list of key skills that match the job description.")
    missing_skills: List[str] = Field(description="A list of up to 3 critical skills missing from the resume.")
    uses_action_verbs: bool = Field(description="True if the resume effectively uses strong action verbs.")
    has_quantifiable_results: bool = Field(description="True if the resume shows measurable, data-driven achievements.")
    recommendation_summary: str = Field(description="A concise, 2-3 sentence expert summary explaining the final recommendation.")
    recommendation_score: int = Field(description="Overall final recommendation score for the candidate (0-100).")

# --- 3. CORE LOGIC FUNCTIONS ---

def get_gemini_analysis_with_langchain(job_desc: str, resume_text: str) -> Optional[dict]:
    """
    This is the new, powerful function that uses LangChain to call the AI.
    It chains together the prompt, the model, and a strict output parser.
    """
    try:
        # Step 1: Initialize the AI Model
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.2,
            convert_system_message_to_human=True
        )

        # Step 2: Create a Pydantic Output Parser
        # This parser knows how to read our ResumeAnalysis class and will
        # automatically provide formatting instructions to the AI.
        parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)

        # Step 3: Create the Prompt Template
        # This template is more robust than a simple f-string.
        prompt_template = PromptTemplate(
            template="""
            You are an exceptionally skilled Senior Technical Recruitment Manager with 15 years of experience.
            Your task is to conduct an in-depth, unbiased analysis of a candidate's resume against a given job description.
            Your analysis must be critical, fair, and based ONLY on the text provided.

            {format_instructions}

            Here is the data for your analysis:
            JOB DESCRIPTION: ```{job_description}```
            RESUME TEXT: ```{resume_text}```
            """,
            input_variables=["job_description", "resume_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Step 4: Create the LangChain Chain
        # This chain connects the prompt, the model, and the parser in a sequence.
        chain = prompt_template | model | parser

        # Step 5: Invoke the chain and get the structured result
        response = chain.invoke({
            "job_description": job_desc,
            "resume_text": resume_text
        })
        
        return response.model_dump() # Convert Pydantic object to a dictionary

    except ValidationError as e:
        st.error("Error: The AI's response did not match the required format. This can happen with very complex or unusual inputs. Please try again.")
        st.expander("See Technical Details").write(e)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while running the AI analysis: {e}")
        return None

def fetch_resume_from_github(github_url: str) -> Optional[str]:
    """
    Ye function GitHub repository se resume file ko dhoond kar uska text nikalta hai.
    """
    try:
        parts = github_url.strip("/").split("/")
        if len(parts) < 2:
            st.error("Invalid GitHub URL format. Please use 'https://github.com/username/repo'.")
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
            st.warning("Could not find a resume file (resume.pdf, resume.md, or README.md) in the repository's root directory.")
            return None

        file_response = requests.get(resume_file_info['download_url'])
        file_response.raise_for_status()

        if resume_file_info['name'].lower().endswith('.pdf'):
            with fitz.open(stream=file_response.content, filetype="pdf") as doc:
                return "".join(page.get_text() for page in doc)
        else:
            return file_response.text
    except requests.exceptions.HTTPError:
        st.error("Could not access the GitHub repository. Please check if the URL is correct and the repository is public.")
        return None
    except Exception as e:
        st.error(f"An error occurred while fetching from GitHub: {e}")
        return None

def generate_report_text(analysis: dict) -> str:
    """
    Download ke liye text report banata hai.
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

# --- 4. UI COMPONENTS ---

def setup_page():
    """Page configuration aur custom CSS."""
    st.set_page_config(layout="wide", page_title="Excellent AI Resume Checker", page_icon="🚀")
    st.markdown("""
    <style>
        .card { border-radius: 8px; padding: 20px; margin-bottom: 20px; border: 1px solid #444; }
        .matched { border-left: 5px solid #04AA6D; }
        .missing { border-left: 5px solid #FFC107; }
        .recommendation { border-left: 5px solid #007bff; }
    </style>
    """, unsafe_allow_html=True)

def render_results(analysis_result: dict):
    """
    AI se mile results ko screen par sundar tarike se dikhata hai.
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

# --- 5. MAIN APPLICATION LOGIC ---

def main():
    setup_page()
    
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    st.title("🚀 Excellent AI Resume Checker")
    st.write("An advanced tool to analyze resumes against job descriptions using Gemini 2.5 Pro and LangChain.")

    try:
        # NOTE: Make sure to set the GOOGLE_API_KEY in Streamlit secrets
        # genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        # LangChain automatically uses the GOOGLE_API_KEY environment variable/secret.
        pass
    except Exception as e:
        st.error(f"🤫 Google API Key not found. Please add it to your Streamlit secrets. Error: {e}")
        st.stop()

    job_desc = st.text_area("📋 Paste the full Job Description here", height=250)
    
    st.write("---")

    input_method = st.radio("Choose Resume Source:", ("Paste Text", "From GitHub URL"), horizontal=True)
    
    resume_text = None
    if input_method == "Paste Text":
        resume_text = st.text_area("✍️ Paste the full Resume text here", height=300)
    else:
        github_url = st.text_input("🔗 Enter public GitHub Repository URL", placeholder="https://github.com/username/repo-name")
        if github_url:
            with st.spinner(f"Fetching resume from '{github_url}'..."):
                resume_text = fetch_resume_from_github(github_url)
            if resume_text:
                st.success("Successfully fetched resume!")
                st.expander("Preview Fetched Resume").text(resume_text[:1000] + "...")

    if st.button("✨ Analyze with LangChain ✨", use_container_width=True, type="primary"):
        if not job_desc or not resume_text:
            st.warning("Please provide both Job Description and Resume content.")
        elif len(job_desc) < 50 or len(resume_text) < 50:
             st.warning("Input text is too short for a meaningful analysis. Please provide full content.")
        else:
            with st.spinner("🤖 Gemini Pro is performing a deep, expert-level analysis via LangChain..."):
                st.session_state.analysis_result = get_gemini_analysis_with_langchain(job_desc, resume_text)

    if st.session_state.analysis_result:
        render_results(st.session_state.analysis_result)

if __name__ == "__main__":
    main()

