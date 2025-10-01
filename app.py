# ======================================================================================
# 🚀 ADVANCED AI RESUME CHECKER - ENHANCED & OPTIMIZED v6.0
# Author: Gemini (in collaboration with the user)
# Description: Professional-grade resume analyzer with consistent results, 
#              enhanced UI, and robust error handling.
# ======================================================================================

# --- 1. IMPORT ALL REQUIRED LIBRARIES ---
import streamlit as st
import requests
import fitz  # PyMuPDF
import hashlib
import time
from typing import List, Optional
import json

# LangChain and Pydantic for robust, stable, and structured AI interaction
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError

# --- 2. DEFINE THE STRICT DATA STRUCTURE FOR THE AI (THE 'BRAIN'S BLUEPRINT) ---
class ResumeAnalysis(BaseModel):
    relevance_score: int = Field(description="Relevance of resume to the job (0-100).")
    skills_match: str = Field(description="Percentage string of skills match (e.g., '85%').")
    years_experience: str = Field(description="Candidate's relevant years of experience (e.g., '5+ Years').")
    education_level: str = Field(description="Alignment of education ('High', 'Medium', 'Low', 'Not Specified').")
    matched_skills: List[str] = Field(description="A list of key matching skills (max 8 items).")
    missing_skills: List[str] = Field(description="A list of up to 3 critical missing skills.")
    uses_action_verbs: bool = Field(description="True if the resume effectively uses strong action verbs.")
    has_quantifiable_results: bool = Field(description="True if the resume shows measurable achievements.")
    recommendation_summary: str = Field(description="A concise, 2-3 sentence expert summary for the final recommendation.")
    recommendation_score: int = Field(description="Overall final recommendation score for the candidate (0-100).")
    analysis_id: str = Field(description="Unique identifier for this analysis session")

# --- 3. CORE LOGIC FUNCTIONS WITH ENHANCED STABILITY ---

def create_analysis_id(job_desc: str, resume_text: str) -> str:
    """Create a unique hash for consistent result caching"""
    content = f"{job_desc[:500]}{resume_text[:500]}"
    return hashlib.md5(content.encode()).hexdigest()[:12]

def get_stable_gemini_analysis(job_desc: str, resume_text: str) -> Optional[dict]:
    """
    Enhanced AI function with result caching and improved prompt engineering
    for consistent results across multiple runs.
    """
    try:
        # Create unique analysis ID for caching
        analysis_id = create_analysis_id(job_desc, resume_text)
        
        # Check cache first
        cache_key = f"analysis_{analysis_id}"
        if cache_key in st.session_state:
            st.info("📊 Loading cached analysis for consistent results...")
            return st.session_state[cache_key]

        # Step 1: Initialize the AI Model with very low temperature for maximum stability
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",  # Using faster, more consistent model
            temperature=0.05,  # Very low temperature for maximum consistency
            max_retries=3,
            timeout=60
        )

        # Step 2: Create the Pydantic Output Parser
        parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)

        # Step 3: Create the enhanced, more specific Prompt Template
        prompt_template = PromptTemplate(
            template="""
            You are a highly analytical and experienced Senior Technical Recruitment Manager.
            Your task is to conduct an in-depth, unbiased analysis of a candidate's resume against a job description.
            
            CRITICAL INSTRUCTIONS FOR CONSISTENCY:
            1. Be extremely objective and data-driven in your assessment
            2. Use the exact same scoring criteria for identical inputs
            3. Focus on factual matches between resume content and job requirements
            4. Ignore writing style variations unless they impact clarity
            5. Use consistent thresholds for all scoring categories
            
            {format_instructions}
            
            ANALYSIS GUIDELINES:
            - Relevance Score: Base on keyword matches, experience alignment, and role fit
            - Skills Match: Calculate percentage of job-required skills found in resume
            - Years Experience: Extract from resume or estimate based on roles
            - Education Level: Match against job requirements (High=Direct match, Medium=Related field, Low=Unrelated)
            - Matched Skills: List only skills explicitly mentioned in both resume and job description
            - Missing Skills: List only critical skills from job description missing in resume
            - Action Verbs: Look for words like "led", "managed", "developed", "implemented"
            - Quantifiable Results: Look for numbers, percentages, metrics, achievements
            
            Here is the data for your analysis:
            JOB DESCRIPTION: ```{job_description}```
            RESUME TEXT: ```{resume_text}```
            
            ANALYSIS ID: {analysis_id}
            """,
            input_variables=["job_description", "resume_text", "analysis_id"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Step 4: Create and Invoke the LangChain Chain
        chain = prompt_template | model | parser
        response = chain.invoke({
            "job_description": job_desc, 
            "resume_text": resume_text,
            "analysis_id": analysis_id
        })
        
        # Convert to dict and add analysis_id
        result = response.model_dump()
        result['analysis_id'] = analysis_id
        
        # Cache the result for consistent future retrievals
        st.session_state[cache_key] = result
        st.session_state['last_analysis_id'] = analysis_id
        
        return result

    except Exception as e:
        st.error(f"🔧 An error occurred during AI analysis. Please try again. Error: {str(e)}")
        return None

def fetch_resume_from_github(github_url: str) -> Optional[str]:
    """
    Enhanced GitHub resume fetcher with better file detection.
    """
    try:
        # Clean and validate URL
        github_url = github_url.strip().rstrip('/')
        if 'github.com' not in github_url:
            st.error("❌ Please enter a valid GitHub URL (e.g., https://github.com/username/repo)")
            return None
            
        parts = github_url.split('github.com/')[-1].split('/')
        if len(parts) < 2:
            st.error("❌ Invalid GitHub URL format")
            return None
            
        username, repo = parts[0], parts[1]
        
        api_url = f"https://api.github.com/repos/{username}/{repo}/contents/"
        
        with st.spinner("🔍 Searching for resume files..."):
            response = requests.get(api_url, timeout=10)
            if response.status_code != 200:
                st.error(f"❌ Could not access repository. Status: {response.status_code}")
                return None
        
            repo_files = response.json()
            if not isinstance(repo_files, list):
                st.error("❌ Unexpected response from GitHub API")
                return None
        
        # Expanded list of possible resume filenames
        resume_filenames = [
            "resume.pdf", "resume.md", "README.md", "cv.pdf", "CV.pdf",
            "RESUME.md", "Resume.pdf", "Resume.md", "CV.md", "cv.md"
        ]
        
        resume_file_info = None
        found_files = []

        for file_info in repo_files:
            if file_info.get('type') == 'file':
                filename = file_info['name'].lower()
                if any(resume_name in filename for resume_name in ['resume', 'cv', 'readme']):
                    found_files.append(file_info)
                if filename in [f.lower() for f in resume_filenames]:
                    resume_file_info = file_info
                    break
        
        # If no exact match, use the first relevant file found
        if not resume_file_info and found_files:
            resume_file_info = found_files[0]
        
        if not resume_file_info:
            st.warning("⚠️ No standard resume file found. Looking for files containing 'resume', 'cv', or 'readme'.")
            # Show available files
            if repo_files:
                available_files = [f['name'] for f in repo_files if f.get('type') == 'file'][:5]
                st.info(f"📁 Available files: {', '.join(available_files)}")
            return None

        # Download and process file
        file_url = resume_file_info.get('download_url')
        if not file_url:
            st.error("❌ Could not get download URL for the file")
            return None
            
        file_response = requests.get(file_url, timeout=10)
        file_response.raise_for_status()

        filename = resume_file_info['name']
        st.success(f"✅ Found: {filename}")

        if filename.lower().endswith('.pdf'):
            with fitz.open(stream=file_response.content, filetype="pdf") as doc:
                text = "".join(page.get_text() for page in doc)
                if len(text.strip()) < 50:
                    st.warning("⚠️ The PDF appears to be mostly empty or image-based. Text extraction may be limited.")
                return text
        else:
            return file_response.text
            
    except requests.exceptions.Timeout:
        st.error("⏰ Request timeout. Please check your internet connection and try again.")
        return None
    except Exception as e:
        st.error(f"❌ Error fetching from GitHub: {str(e)}")
        return None

def generate_report_text(analysis: dict) -> str:
    """
    Generates a comprehensive downloadable report.
    """
    score = analysis.get('recommendation_score', 0)
    if score >= 80: 
        verdict = "🏆 Highly Recommended"
        emoji = "🎯"
    elif score >= 60: 
        verdict = "👍 Worth Considering" 
        emoji = "📊"
    elif score >= 40: 
        verdict = "🤔 Maybe Consider"
        emoji = "⚖️"
    else: 
        verdict = "❌ Not a Strong Fit"
        emoji = "💡"

    return f"""
AI RESUME ANALYSIS REPORT
===========================
ANALYSIS ID: {analysis.get('analysis_id', 'N/A')}
TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}

FINAL VERDICT: {emoji} {verdict} ({score}%)

EXECUTIVE SUMMARY:
{analysis.get('recommendation_summary', 'N/A')}

KEY METRICS:
• AI Relevance Score: {analysis.get('relevance_score', 0)}%
• Skills Match: {analysis.get('skills_match', 'N/A')}
• Years of Experience: {analysis.get('years_experience', 'N/A')}
• Education Level: {analysis.get('education_level', 'N/A')}

SKILLS ANALYSIS:
• Matched Skills: {', '.join(analysis.get('matched_skills', ['N/A']))}
• Missing Skills: {', '.join(analysis.get('missing_skills', ['N/A']))}

RESUME QUALITY:
• Uses Strong Action Verbs: {'✅ Yes' if analysis.get('uses_action_verbs') else '❌ No'}
• Has Quantifiable Results: {'✅ Yes' if analysis.get('has_quantifiable_results') else '❌ No'}

RECOMMENDATION:
{analysis.get('recommendation_summary', 'N/A')}
"""

# --- 4. ENHANCED UI COMPONENTS ---

def setup_page_and_styles():
    """
    Sets up the Streamlit page with enhanced dark theme design.
    """
    st.set_page_config(
        layout="wide", 
        page_title="Advanced AI Resume Checker", 
        page_icon="🚀",
        initial_sidebar_state="collapsed"
    )
    
    # Enhanced Dark Theme with better spacing and colors
    st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        }
        .card {
            background-color: #1e1e2e;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid #444;
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.3);
            color: #e2e2e2;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 24px 0 rgba(0,0,0,0.4);
        }
        .card h5 {
            margin-top: 0;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            font-size: 1.2em;
            color: #ffffff;
            font-weight: 600;
        }
        .card.matched {
            background: linear-gradient(135deg, rgba(4, 170, 109, 0.15) 0%, #1e1e2e 100%);
            border-left: 5px solid #04AA6D;
        }
        .card.missing {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, #1e1e2e 100%);
            border-left: 5px solid #FFC107;
        }
        .card.recommendation {
            background: linear-gradient(135deg, rgba(0, 123, 255, 0.15) 0%, #1e1e2e 100%);
            border-left: 5px solid #007bff;
        }
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #ff4b4b 0%, #ffa500 50%, #04AA6D 100%);
        }
        .metric-container {
            background-color: #2d2d44;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #007bff;
        }
        .analysis-id {
            background: #2d2d44;
            padding: 8px 12px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.9em;
            border: 1px solid #444;
        }
    </style>
    """, unsafe_allow_html=True)

def render_verdict_section(score: int):
    """Render the verdict section with enhanced visual appeal"""
    if score >= 80: 
        color, text, icon = "green", "Highly Recommended", "🏆"
    elif score >= 60: 
        color, text, icon = "orange", "Worth Considering", "👍"
    elif score >= 40: 
        color, text, icon = "yellow", "Maybe Consider", "🤔"
    else: 
        color, text, icon = "red", "Not a Strong Fit", "❌"
    
    st.subheader(f"{icon} Final Verdict: :{color}[{text} ({score}%)]")
    st.progress(score / 100)

def render_results(analysis_result: dict):
    """
    Enhanced results display with better organization and visuals.
    """
    st.divider()
    st.header("📊 Detailed Analysis Results")
    
    # Analysis ID for reference
    if 'analysis_id' in analysis_result:
        st.caption(f"Analysis ID: `{analysis_result['analysis_id']}` - Identical inputs will produce consistent results")
    
    # Verdict Section
    score = analysis_result.get('recommendation_score', 0)
    render_verdict_section(score)
    
    # Key Metrics in a more compact layout
    st.markdown("### 📈 Key Metrics")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"<div class='metric-container'>🤖 AI Relevance<br><h3>{analysis_result.get('relevance_score', 0)}%</h3></div>", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"<div class='metric-container'>🔧 Skills Match<br><h3>{analysis_result.get('skills_match', 'N/A')}</h3></div>", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"<div class='metric-container'>⏳ Experience<br><h3>{analysis_result.get('years_experience', 'N/A')}</h3></div>", unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"<div class='metric-container'>🎓 Education<br><h3>{analysis_result.get('education_level', 'N/A')}</h3></div>", unsafe_allow_html=True)

    # Skills Analysis with enhanced cards
    st.markdown("### 🛠️ Skills Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='card matched'>
            <h5>✅ Matched Skills ({len(analysis_result.get('matched_skills', []))})</h5>
            <p>{' • '.join(analysis_result.get('matched_skills', ['N/A']))}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='card missing'>
            <h5>❌ Missing Skills ({len(analysis_result.get('missing_skills', []))})</h5>
            <p>{' • '.join(analysis_result.get('missing_skills', ['N/A']))}</p>
        </div>
        """, unsafe_allow_html=True)

    # Resume Quality Assessment
    st.markdown("### 📝 Resume Quality Assessment")
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        action_verbs = analysis_result.get('uses_action_verbs', False)
        st.markdown(f"<div class='card'><h5>💪 Action Verbs</h5><h3>{'✅ Effectively Used' if action_verbs else '❌ Needs Improvement'}</h3></div>", unsafe_allow_html=True)
    with q_col2:
        quantifiable = analysis_result.get('has_quantifiable_results', False)
        st.markdown(f"<div class='card'><h5>📊 Quantifiable Results</h5><h3>{'✅ Well Demonstrated' if quantifiable else '❌ Lacking Metrics'}</h3></div>", unsafe_allow_html=True)

    # Recommendation
    st.markdown("### 💡 Expert Recommendation")
    st.markdown(f"<div class='card recommendation'><p>{analysis_result.get('recommendation_summary', 'N/A')}</p></div>", unsafe_allow_html=True)

    # Download Report
    st.divider()
    report_data = generate_report_text(analysis_result)
    st.download_button(
        label="📥 Download Comprehensive Analysis Report", 
        data=report_data, 
        file_name=f"Resume_Analysis_Report_{analysis_result.get('analysis_id', 'unknown')}.txt", 
        mime="text/plain", 
        use_container_width=True
    )

def main():
    """
    The main function that runs the enhanced Streamlit application.
    """
    setup_page_and_styles()
    
    # Initialize session state for consistent results
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'last_inputs_hash' not in st.session_state:
        st.session_state.last_inputs_hash = None

    # Header with enhanced design
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #ffffff; margin-bottom: 10px;'>🚀 Advanced AI Resume Checker</h1>
        <p style='color: #a0a0a0; font-size: 1.2em;'>Professional-grade analysis with consistent, accurate results</p>
    </div>
    """, unsafe_allow_html=True)

    # API Key check
    try:
        if "GOOGLE_API_KEY" not in st.secrets:
            raise Exception("API Key not found")
    except Exception:
        st.error("🔑 Google API Key not found. Please add it to your Streamlit secrets.")
        st.info("💡 Add your API key in Streamlit Cloud under Settings → Secrets")
        st.stop()
    
    # --- ENHANCED INPUT SECTION ---
    st.markdown("### 📋 Job Description")
    job_desc = st.text_area(
        "Paste the complete Job Description here", 
        height=200,
        placeholder="Copy and paste the entire job description including requirements, responsibilities, and qualifications...",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### 📄 Resume Content")
    input_method = st.radio(
        "Choose Resume Source:", 
        ("📝 Paste Text", "🔗 GitHub Repository"), 
        horizontal=True,
        label_visibility="collapsed"
    )
    
    resume_text = None
    if input_method == "📝 Paste Text":
        resume_text = st.text_area(
            "Paste the complete Resume text here", 
            height=250,
            placeholder="Copy and paste the entire resume text...",
            label_visibility="collapsed"
        )
    else:
        github_url = st.text_input(
            "Enter public GitHub Repository URL", 
            placeholder="https://github.com/username/repository-name"
        )
        if github_url:
            with st.spinner("🔄 Fetching resume from GitHub..."):
                resume_text = fetch_resume_from_github(github_url)
            if resume_text:
                st.success("✅ Successfully fetched resume content!")
                with st.expander("👀 Preview Fetched Resume (First 1000 characters)"):
                    st.text(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)

    # --- ENHANCED ANALYSIS TRIGGER WITH INPUT VALIDATION ---
    st.markdown("---")
    
    current_inputs_hash = hashlib.md5(f"{job_desc}{resume_text}".encode()).hexdigest()
    is_same_input = st.session_state.last_inputs_hash == current_inputs_hash
    
    if st.button("✨ Run Advanced Analysis ✨", use_container_width=True, type="primary"):
        # Enhanced input validation
        if not job_desc or not resume_text:
            st.warning("⚠️ Please provide both Job Description and Resume content.")
        elif len(job_desc) < 100:
            st.warning("📝 Job Description seems too short. Please provide a complete job description for accurate analysis.")
        elif len(resume_text) < 100:
            st.warning("📄 Resume content seems too short. Please provide a complete resume for meaningful analysis.")
        elif is_same_input and st.session_state.analysis_result:
            st.info("🔄 Same inputs detected. Loading cached analysis for consistency...")
            # Re-use cached result for identical inputs
        else:
            with st.spinner("🤖 AI is performing deep analysis with enhanced consistency algorithms..."):
                result = get_stable_gemini_analysis(job_desc, resume_text)
                if result:
                    st.session_state.analysis_result = result
                    st.session_state.last_inputs_hash = current_inputs_hash
                    st.success("✅ Analysis completed successfully!")
                else:
                    st.error("❌ Analysis failed. Please check your inputs and try again.")

    # --- DISPLAY RESULTS ---
    if st.session_state.analysis_result:
        render_results(st.session_state.analysis_result)

# --- APPLICATION ENTRY POINT ---
if __name__ == "__main__":
    main()
