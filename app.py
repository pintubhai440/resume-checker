# Required libraries are imported for the application
import streamlit as st
import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.prompts import PromptTemplate
from collections import Counter
import time
import datetime
import pandas as pd
from io import StringIO
import PyPDF2
import tempfile

# --- Helper Functions for Resume Quality Analysis ---

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return None

def extract_text_from_txt(txt_file):
    """Extract text from TXT file"""
    try:
        text = str(txt_file.read(), "utf-8")
        return text
    except Exception as e:
        st.error(f"Error reading TXT file: {str(e)}")
        return None

def get_word_count_status(text):
    """Analyze resume word count with context for freshers."""
    # Clean the text first - remove extra whitespaces and count actual words
    clean_text = re.sub(r'\s+', ' ', text.strip())
    word_count = len(clean_text.split())
    
    if word_count < 200:
        return f"⚠️ Too Short ({word_count} words)"
    elif 200 <= word_count <= 600:
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
    """Extracts and cleans a JSON object from a string with better error handling."""
    try:
        # Multiple patterns to catch JSON in different formats
        json_patterns = [
            r'\{[^{}]*\{[^{}]*\}[^{}]*\}',  # Nested objects
            r'\{.*\}',  # Simple objects
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                
                # Clean the JSON text
                json_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_text)
                json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
                json_text = re.sub(r'(\w+):', r'"\1":', json_text)  # Ensure keys are quoted
                
                return json_text
        return None
    except Exception as e:
        st.error(f"JSON cleaning error: {str(e)}")
        return None

def get_improvement_suggestions(job_description, analysis_result, llm):
    """Generate improvement suggestions using a separate, focused AI call."""
    try:
        # AI ko context dene ke liye analysis se summary banayein
        score = analysis_result.get('recommendation_score', 0)
        missing_skills = ", ".join(analysis_result.get('missing_skills', [])) or "None"
        summary = analysis_result.get('recommendation_summary', 'N/A')
        experience = analysis_result.get('years_experience', 'N/A')

        suggestion_prompt_template = """
        You are a constructive and encouraging career coach providing feedback to a job applicant.

        **CONTEXT OF THE ANALYSIS:**
        - **Job Description Summary:** {jd}
        - **Candidate's Recommendation Score:** {score}%
        - **Key Missing Skills Identified:** {missing_skills}
        - **Candidate's Experience Level:** {experience}
        - **Recruiter's Assessment Summary:** {summary}

        **YOUR TASK:**
        Based on the context above, provide actionable improvement suggestions for the candidate. Address them directly in the second person ("You should...", "Consider...").

        - **If the score is below 60% (Not a good fit):** Focus on the most critical gaps. Provide a clear, step-by-step roadmap for what they need to learn or do to qualify for such roles in the future. Be direct but supportive.
        - **If the score is 60% or higher (A good fit):** Focus on suggestions that will make them an even stronger candidate. Suggest advanced skills, relevant certifications, or ways to better showcase their achievements.

        **OUTPUT FORMAT (Strictly follow this):**
        1. Start with a brief, encouraging summary sentence (1 line).
        2. Follow with 2-4 specific, actionable bullet points in Markdown format.
        3. Keep the entire response concise and to the point. DO NOT add any extra text or explanations.

        **GENERATE SUGGESTIONS NOW:**
        """

        suggestion_prompt = PromptTemplate.from_template(suggestion_prompt_template)
        suggestion_chain = suggestion_prompt | llm

        response = suggestion_chain.invoke({
            "jd": job_description[:500],  # JD ka summary use karein
            "score": score,
            "missing_skills": missing_skills,
            "summary": summary,
            "experience": experience
        })

        return response.content

    except Exception as e:
        st.warning(f"Could not generate improvement suggestions: {str(e)}")
        return "Suggestions could not be generated at this time."

def validate_analysis_result(result):
    """Ensure the analysis result has all required fields with proper defaults."""
    required_fields = {
        'relevance_score': 0,
        'skills_match': 0,
        'years_experience': 'Not Specified',
        'education_level': 'Not Specified',
        'matched_skills': [],
        'missing_skills': [],
        'recommendation_summary': 'Analysis incomplete.',
        'uses_action_verbs': False,
        'has_quantifiable_results': False,
        'recommendation_score': 0,
        'improvement_suggestions': 'No suggestions generated.'
    }
    
    validated_result = required_fields.copy()
    validated_result.update(result)
    
    # Ensure scores are within bounds
    validated_result['relevance_score'] = max(0, min(100, validated_result['relevance_score']))
    validated_result['skills_match'] = max(0, min(100, validated_result['skills_match']))
    validated_result['recommendation_score'] = max(0, min(100, validated_result['recommendation_score']))
    
    return validated_result

def analyze_single_resume(resume_text, job_description, llm):
    """Analyze a single resume against job description"""
    try:
        current_year = datetime.datetime.now().year
        experience_rules = f"""**EXPERIENCE LEVEL CALCULATION (Current Year: {current_year}):**
- "Fresher": 0-1 years experience OR currently studying
- "Junior": 1-3 years experience 
- "Mid-Level": 3-6 years experience
- "Senior": 6+ years experience

**WORK EXPERIENCE CALCULATION:**
- Calculate total professional work experience from all jobs/internships
- Full-time roles count as actual duration
- Internships count as half the duration (e.g., 6-month internship = 3 months experience)
- Be REALISTIC and CONSERVATIVE in experience calculation
"""

        part1 = """CRITICAL INSTRUCTIONS: You MUST return ONLY a valid JSON object. No additional text, no explanations, no markdown.
You are an expert Senior Technical Recruiter. Analyze the RESUME against the JOB DESCRIPTION with brutal honesty.

**STRICT PRIORITY ORDER:**
1. **ELIGIBILITY CHECK FIRST**: Check graduation year and batch eligibility BEFORE anything else
2. **EXPERIENCE CALCULATION**: Calculate total work experience from ALL jobs/internships - BE REALISTIC
3. **TECHNICAL SKILLS**: Only count skills EXPLICITLY mentioned in resume
4. **NO INFERENCES**: If not written, it doesn't exist
5. **BE CRITICAL**: Identify weaknesses and missing skills

**BATCH ELIGIBILITY RULES:**
- If JD requires "2023 and earlier pass-outs" and candidate passed in 2024 -> NOT ELIGIBLE
- If JD requires "2023 and earlier pass-outs" and candidate passed in 2022 -> ELIGIBLE
- Only check graduation/passing year mentioned in resume

**EXPERIENCE CALCULATION RULES:**
- Sum ALL professional experience (jobs + internships)
- Internships count as 50% of their duration
- If no dates mentioned, assume no experience
- Be CONSERVATIVE - don't overestimate
"""

        part2 = """

**JOB DESCRIPTION:**
{jd}

**RESUME:**
{resume}

**ANALYSIS OUTPUT - RETURN ONLY THIS JSON:**
{{
    "relevance_score": 75,
    "skills_match": 80,
    "years_experience": "Junior",
    "education_level": "High",
    "matched_skills": ["Python", "SQL"],
    "missing_skills": ["NoSQL", "Cloud"],
    "recommendation_summary": "Good candidate with solid foundation but missing some advanced skills. Consider for junior role.",
    "uses_action_verbs": true,
    "has_quantifiable_results": false,
    "recommendation_score": 72
}}

**STRICT SCORING GUIDELINES - FOLLOW THESE EXACTLY:**

**RECOMMENDATION SCORE RANGES (MUST FOLLOW):**
- 90-100%: PERFECT MATCH - All requirements met exactly + extra qualifications
- 85-89%: EXCELLENT MATCH - All key requirements met, minor gaps
- 80-84%: STRONG MATCH - Most requirements met, some minor gaps
- 75-79%: GOOD MATCH - Solid match with some noticeable gaps
- 70-74%: DECENT MATCH - Meets basic requirements but has significant gaps
- 60-69%: AVERAGE MATCH - Partial match, major skill/experience gaps
- 50-59%: WEAK MATCH - Barely meets minimum requirements
- 40-49%: POOR MATCH - Major deficiencies
- 30-39%: VERY POOR MATCH - Critical gaps
- 20-29%: MINIMAL MATCH - Few requirements met
- 10-19%: ALMOST NO MATCH - Hardly any requirements met
- 0-9%: NO MATCH - Completely unsuitable

**SKILLS MATCH SCORING:**
- 100%: All JD skills explicitly mentioned (RARE)
- 90-99%: Almost all key skills mentioned
- 80-89%: Most key skills mentioned, missing 1-2 important ones
- 70-79%: Good skill overlap, missing several important skills
- 60-69%: Basic skills match, missing many key skills
- 50-59%: Limited skill overlap
- Below 50%: Poor skill match

**EXPERIENCE EVALUATION:**
- Calculate TOTAL realistic experience (jobs + 50% of internships)
- Compare against JD requirement STRICTLY
- If JD says "2+ years" and candidate has 1.5 years -> this is a GAP

**MISSING SKILLS IDENTIFICATION:**
- MUST identify at least 2-3 missing skills unless candidate is perfect
- Look for skills in JD that are NOT in resume
- Be specific about what's missing

**BE REALISTIC AND CRITICAL - VERY FEW CANDIDATES SHOULD SCORE ABOVE 85%**

**EDUCATION LEVELS:**
- "High": B.Tech/BE/Masters from recognized institute
- "Medium": Bachelor's degree from any college
- "Low": Diploma/No degree

RETURN ONLY THE JSON OBJECT:
"""
        final_prompt_text = part1 + experience_rules + part2
        
        analysis_prompt = PromptTemplate.from_template(final_prompt_text)
        analysis_chain = analysis_prompt | llm

        response = analysis_chain.invoke({"resume": resume_text, "jd": job_description})
        response_text = response.content
        
        # Debug: Show raw response
        with st.expander("🔧 Debug: Raw AI Response"):
            st.code(response_text)
        
        cleaned_json = clean_json_response(response_text)
        if not cleaned_json:
            st.error("❌ Could not extract JSON from AI response")
            return None
            
        analysis_result = json.loads(cleaned_json)
        analysis_result = validate_analysis_result(analysis_result)
        
        # Analysis ke basis par improvement suggestions generate karein
        with st.spinner('💡 Generating personalized improvement suggestions...'):
            suggestions = get_improvement_suggestions(job_description, analysis_result, llm)
            analysis_result['improvement_suggestions'] = suggestions
        
        # Add quality metrics
        analysis_result['word_count_status'] = get_word_count_status(resume_text)
        analysis_result['repetition_status'] = get_repetition_status(resume_text)
        
        return analysis_result
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

def display_detailed_result(analysis_result, candidate_name):
    """Display detailed analysis result for a single candidate"""
    
    recommendation_score = analysis_result.get('recommendation_score', 0)
    
    if recommendation_score >= 80:
        rec_color, rec_text = "green", "Highly Recommended"
    elif recommendation_score >= 60:
        rec_color, rec_text = "orange", "Worth Considering"
    elif recommendation_score >= 40:
        rec_color, rec_text = "red", "Not Recommended"
    else:
        rec_color, rec_text = "red", "Strongly Not Recommended"

    st.subheader(f"🎯 {candidate_name} - Final Verdict: :{rec_color}[{rec_text} ({recommendation_score}%)]")
    st.progress(recommendation_score / 100)

    # Key Metrics - Fixed layout
    st.markdown("### 📊 Key Metrics")
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    
    with res_col1:
        st.metric("AI Relevance Score", f"{analysis_result.get('relevance_score', 0)}%")
    with res_col2:
        st.metric("Skills Match", f"{analysis_result.get('skills_match', 0)}%")
    with res_col3:
        st.metric("Years' Experience", analysis_result.get('years_experience', 'Not Specified'))
    with res_col4:
        st.metric("Education Level", analysis_result.get('education_level', 'Not Specified'))

    # Skills Analysis - Fixed layout
    st.markdown("### 🔧 Skills Analysis")
    skill_col1, skill_col2 = st.columns(2)
    
    # Add CSS for better skill badges
    st.markdown("""
    <style>
    .skill-badge { 
        display: inline-block; 
        padding: 6px 12px; 
        margin: 4px; 
        font-size: 0.9em; 
        font-weight: 500; 
        border-radius: 15px; 
        text-align: center;
        white-space: nowrap;
    }
    .matched-skill { 
        background-color: #E0F2E9; 
        color: #0D6938; 
        border: 1px solid #A3D4B6; 
    }
    .missing-skill { 
        background-color: #FFF3D4; 
        color: #B47D00; 
        border: 1px solid #FFDDA0; 
    }
    .skills-container {
        max-height: 200px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #fafafa;
    }
    </style>
    """, unsafe_allow_html=True)

    with skill_col1:
        st.success("✅ Matched Skills")
        matched_skills = analysis_result.get('matched_skills', [])
        if matched_skills:
            skills_html = "".join([f'<span class="skill-badge matched-skill">{skill}</span>' for skill in matched_skills])
            st.markdown(f"<div class='skills-container' style='line-height: 2.0;'>{skills_html}</div>", unsafe_allow_html=True)
        else:
            st.info("No matching skills found")

    with skill_col2:
        st.warning("❗️ Critical Missing Skills")
        missing_skills = analysis_result.get('missing_skills', [])
        if missing_skills:
            skills_html = "".join([f'<span class="skill-badge missing-skill">{skill}</span>' for skill in missing_skills])
            st.markdown(f"<div class='skills-container' style='line-height: 2.0;'>{skills_html}</div>", unsafe_allow_html=True)
        else:
            st.info("No major skill gaps identified")

    # Professional Assessment
    st.markdown("### 💡 Professional Assessment")
    st.info(analysis_result.get('recommendation_summary', 'No analysis available.'))

    # Resume Quality Analysis - Fixed layout
    st.markdown("### 📝 Resume Quality Analysis")
    
    action_verbs = "✅ Yes" if analysis_result.get('uses_action_verbs') else "❌ No"
    quant_results = "✅ Yes" if analysis_result.get('has_quantifiable_results') else "❌ No"
    
    # Fixed metric cards with better styling
    st.markdown("""
    <style>
    .metric-card { 
        background-color: #F8F9FA; 
        border-radius: 10px; 
        padding: 15px; 
        text-align: center; 
        border: 1px solid #E0E0E0; 
        margin: 5px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card p.label { 
        font-size: 14px; 
        color: #555; 
        margin-bottom: 5px; 
        font-weight: 500; 
    }
    .metric-card p.value { 
        font-size: 16px; 
        font-weight: bold; 
        color: #333; 
        margin: 0; 
    }
    </style>
    """, unsafe_allow_html=True)

    quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
    
    with quality_col1:
        st.markdown(f'<div class="metric-card"><p class="label">Word Count</p><p class="value">{analysis_result.get("word_count_status", "N/A")}</p></div>', unsafe_allow_html=True)
    with quality_col2:
        st.markdown(f'<div class="metric-card"><p class="label">Keyword Repetition</p><p class="value">{analysis_result.get("repetition_status", "N/A")}</p></div>', unsafe_allow_html=True)
    with quality_col3:
        st.markdown(f'<div class="metric-card"><p class="label">Action Verbs</p><p class="value">{action_verbs}</p></div>', unsafe_allow_html=True)
    with quality_col4:
        st.markdown(f'<div class="metric-card"><p class="label">Quantifiable Results</p><p class="value">{quant_results}</p></div>', unsafe_allow_html=True)

    # Improvement Suggestions ko display karein
    st.markdown("### 🌱 Improvement Suggestions")
    suggestions = analysis_result.get('improvement_suggestions', 'No suggestions available.')
    if "Suggestions could not be generated" in suggestions or not suggestions:
        st.warning(suggestions or "No suggestions were generated for this candidate.")
    else:
        st.success(suggestions) # 'success' box se yeh constructive lagega

    st.divider()

# --- UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Resume Checker", page_icon="🚀")
st.title("🚀 AI Resume Checker")
st.write("Get consistent, accurate, and data-driven resume analysis with Gemini. This tool provides precise relevance score, skill gap analysis, and detailed evaluation.")

# --- API KEY & MODEL SETUP ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except (FileNotFoundError, KeyError):
    st.error("🤫 Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()

# --- TAB LAYOUT ---
tab1, tab2 = st.tabs(["📄 Single Resume Analysis", "📊 Batch Resume Analysis"])

with tab1:
    # --- SINGLE RESUME ANALYSIS ---
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.header("📄 Job Requirements")
        job_description = st.text_area("Paste the Job Description here", height=350, key="single_jd", 
                                     placeholder="Enter the complete job description with required skills, qualifications, and experience...")
    with col2:
        st.header("👤 Candidate's Resume")
        resume_option = st.radio("Choose input method:", 
                               ["📝 Paste Text", "📄 Upload PDF", "📁 Upload TXT"], 
                               key="single_input_method")
        
        if resume_option == "📝 Paste Text":
            resume_text = st.text_area("Paste the Resume Text here", height=300, key="single_resume", 
                                     placeholder="Enter the complete resume text including education, skills, experience, projects...")
        elif resume_option == "📄 Upload PDF":
            uploaded_pdf = st.file_uploader("Upload PDF Resume", type=['pdf'], key="single_pdf")
            if uploaded_pdf:
                resume_text = extract_text_from_pdf(uploaded_pdf)
                if resume_text:
                    st.success("✅ PDF uploaded successfully!")
                    with st.expander("View extracted text from PDF"):
                        st.text_area("Extracted Text", resume_text, height=200)
                else:
                    resume_text = ""
            else:
                resume_text = ""
        else:  # Upload TXT
            uploaded_txt = st.file_uploader("Upload TXT Resume", type=['txt'], key="single_txt")
            if uploaded_txt:
                resume_text = extract_text_from_txt(uploaded_txt)
                if resume_text:
                    st.success("✅ TXT file uploaded successfully!")
                    with st.expander("View text content"):
                        st.text_area("File Content", resume_text, height=200)
                else:
                    resume_text = ""
            else:
                resume_text = ""

    # --- ANALYSIS BUTTON & LOGIC FOR SINGLE RESUME ---
    if st.button("Analyze with Gemini AI", use_container_width=True, type="primary", key="single_analyze"):
        if not job_description.strip():
            st.warning("⚠️ Please provide the Job Description.")
        elif not resume_text.strip():
            st.warning("⚠️ Please provide the Resume text or upload a file.")
        else:
            with st.spinner('🔍 Gemini is performing a deep analysis... This might take a moment.'):
                # --- ORIGINAL MODEL FROM YOUR CODE ---
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-preview-05-20", 
                    temperature=0.1,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                
                analysis_result = analyze_single_resume(resume_text, job_description, llm)
                
                if analysis_result:
                    display_detailed_result(analysis_result, "Candidate")
                    
                    # Generate comprehensive report
                    report_text = f"""
RESUME ANALYSIS REPORT
=====================
CANDIDATE: Single Candidate Analysis
FINAL VERDICT: {analysis_result.get('recommendation_summary', 'No analysis available')}
RECOMMENDATION SCORE: {analysis_result.get('recommendation_score', 0)}%
RELEVANCE SCORE: {analysis_result.get('relevance_score', 0)}%
SKILLS MATCH: {analysis_result.get('skills_match', 0)}%
EXPERIENCE LEVEL: {analysis_result.get('years_experience', 'Not Specified')}
EDUCATION LEVEL: {analysis_result.get('education_level', 'Not Specified')}

MATCHED SKILLS: {', '.join(analysis_result.get('matched_skills', []))}
MISSING SKILLS: {', '.join(analysis_result.get('missing_skills', []))}

RESUME QUALITY:
- Word Count: {analysis_result.get('word_count_status', 'N/A')}
- Repetition: {analysis_result.get('repetition_status', 'N/A')}
- Action Verbs: {'Yes' if analysis_result.get('uses_action_verbs') else 'No'}
- Quantifiable Results: {'Yes' if analysis_result.get('has_quantifiable_results') else 'No'}

IMPROVEMENT SUGGESTIONS:
{analysis_result.get('improvement_suggestions', 'No suggestions available.')}

PROFESSIONAL ASSESSMENT:
{analysis_result.get('recommendation_summary', 'No analysis available')}

Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
                    
                    st.download_button(
                        label="📥 Download Comprehensive Report",
                        data=report_text,
                        file_name="detailed_resume_analysis_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

with tab2:
    # --- BATCH RESUME ANALYSIS ---
    st.header("📊 Batch Resume Analysis")
    st.write("Upload multiple resumes at once and get comparative analysis results.")
    
    batch_col1, batch_col2 = st.columns(2)
    
    with batch_col1:
        st.subheader("Job Requirements")
        batch_job_description = st.text_area("Paste the Job Description here", height=250, key="batch_jd",
                                           placeholder="Enter the job description for all candidates...")
    
    with batch_col2:
        st.subheader("Upload Resumes")
        uploaded_files = st.file_uploader("Choose PDF or TXT files with resumes", 
                                        type=['pdf', 'txt'], 
                                        accept_multiple_files=True,
                                        help="Upload multiple PDF or TXT files containing resumes")
        
        st.info("💡 **Instructions**: Upload multiple PDF or TXT files. Each file should contain one resume. Files should be named meaningfully (e.g., candidate_name.pdf)")

    if st.button("🚀 Analyze All Resumes", use_container_width=True, type="primary", key="batch_analyze"):
        if not batch_job_description.strip():
            st.warning("⚠️ Please provide the Job Description first.")
        elif not uploaded_files:
            st.warning("⚠️ Please upload at least one resume file.")
        else:
            with st.spinner(f'🔍 Analyzing {len(uploaded_files)} resumes with Gemini AI... This may take several minutes.'):
                # --- ORIGINAL MODEL FROM YOUR CODE ---
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-preview-05-20", 
                    temperature=0.1,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Update progress
                        progress = (i / len(uploaded_files))
                        progress_bar.progress(progress)
                        status_text.text(f"Analyzing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                        
                        # Extract text based on file type
                        if uploaded_file.type == "application/pdf":
                            resume_text = extract_text_from_pdf(uploaded_file)
                        else:  # text/plain
                            resume_text = extract_text_from_txt(uploaded_file)
                        
                        if not resume_text:
                            st.warning(f"⚠️ Could not extract text from {uploaded_file.name}")
                            continue
                            
                        # Analyze resume
                        analysis_result = analyze_single_resume(resume_text, batch_job_description, llm)
                        
                        if analysis_result:
                            # Add candidate identifier
                            candidate_name = uploaded_file.name.replace('.pdf', '').replace('.txt', '')
                            analysis_result['candidate_name'] = candidate_name
                            analysis_result['file_name'] = uploaded_file.name
                            analysis_result['file_type'] = uploaded_file.type
                            results.append(analysis_result)
                        
                        # Small delay to avoid rate limiting
                        time.sleep(1)
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        continue
                
                progress_bar.progress(1.0)
                status_text.text("Analysis complete!")
                
                if results:
                    st.success(f"✅ Successfully analyzed {len(results)} out of {len(uploaded_files)} resumes")
                    
                    # Show file type summary
                    pdf_count = len([r for r in results if r.get('file_type') == 'application/pdf'])
                    txt_count = len([r for r in results if r.get('file_type') == 'text/plain'])
                    
                    if pdf_count > 0 or txt_count > 0:
                        st.info(f"📊 File types processed: {pdf_count} PDF files, {txt_count} TXT files")
                    
                    # Create results dataframe for overview
                    df_data = []
                    for result in results:
                        df_data.append({
                            'Candidate': result['candidate_name'],
                            'File Type': 'PDF' if result.get('file_type') == 'application/pdf' else 'TXT',
                            'Recommendation Score': result['recommendation_score'],
                            'Relevance Score': result['relevance_score'],
                            'Skills Match %': result['skills_match'],
                            'Experience Level': result['years_experience'],
                            'Education Level': result['education_level'],
                            'Matched Skills Count': len(result['matched_skills']),
                            'Missing Skills Count': len(result['missing_skills']),
                            'Verdict': 'Highly Recommended' if result['recommendation_score'] >= 80 else 
                                      'Worth Considering' if result['recommendation_score'] >= 60 else 
                                      'Not Recommended' if result['recommendation_score'] >= 40 else 
                                      'Strongly Not Recommended'
                        })
                    
                    df = pd.DataFrame(df_data)
                    
                    # Sort by recommendation score (descending)
                    df = df.sort_values('Recommendation Score', ascending=False)
                    
                    # Display summary section
                    st.subheader("📈 Comparative Overview")
                    
                    # Display summary metrics
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    with metric_col1:
                        st.metric("Total Analyzed", len(results))
                    with metric_col2:
                        high_rec = len([r for r in results if r['recommendation_score'] >= 80])
                        st.metric("Highly Recommended", high_rec)
                    with metric_col3:
                        avg_score = df['Recommendation Score'].mean()
                        st.metric("Average Score", f"{avg_score:.1f}%")
                    with metric_col4:
                        top_score = df['Recommendation Score'].max()
                        st.metric("Top Score", f"{top_score:.1f}%")
                    
                    # Display results table
                    st.dataframe(df, use_container_width=True)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="batch_resume_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # INDIVIDUAL DETAILED RESULTS SECTION
                    st.markdown("---")
                    st.header("📋 Individual Detailed Results")
                    st.write("Below are the detailed analysis reports for each candidate:")
                    
                    # Create tabs for each candidate for better organization
                    candidate_tabs = st.tabs([f"👤 {result['candidate_name']} ({'PDF' if result.get('file_type') == 'application/pdf' else 'TXT'})" for result in results])
                    
                    for i, (result, tab) in enumerate(zip(results, candidate_tabs)):
                        with tab:
                            display_detailed_result(result, result['candidate_name'])
                            
                            # Individual download button for each candidate
                            report_text = f"""
RESUME ANALYSIS REPORT
=====================
CANDIDATE: {result['candidate_name']}
FILE TYPE: {'PDF' if result.get('file_type') == 'application/pdf' else 'TXT'}
FINAL VERDICT: {result.get('recommendation_summary', 'No analysis available')}
RECOMMENDATION SCORE: {result.get('recommendation_score', 0)}%
RELEVANCE SCORE: {result.get('relevance_score', 0)}%
SKILLS MATCH: {result.get('skills_match', 0)}%
EXPERIENCE LEVEL: {result.get('years_experience', 'Not Specified')}
EDUCATION LEVEL: {result.get('education_level', 'Not Specified')}

MATCHED SKILLS: {', '.join(result.get('matched_skills', []))}
MISSING SKILLS: {', '.join(result.get('missing_skills', []))}

RESUME QUALITY:
- Word Count: {result.get('word_count_status', 'N/A')}
- Repetition: {result.get('repetition_status', 'N/A')}
- Action Verbs: {'Yes' if result.get('uses_action_verbs') else 'No'}
- Quantifiable Results: {'Yes' if result.get('has_quantifiable_results') else 'No'}

IMPROVEMENT SUGGESTIONS:
{result.get('improvement_suggestions', 'No suggestions available.')}

PROFESSIONAL ASSESSMENT:
{result.get('recommendation_summary', 'No analysis available')}

Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
                            
                            st.download_button(
                                label=f"📥 Download {result['candidate_name']}'s Report",
                                data=report_text,
                                file_name=f"{result['candidate_name']}_resume_analysis_report.txt",
                                mime="text/plain",
                                key=f"download_{i}",
                                use_container_width=True
                            )
                    
                    # Batch download all individual reports
                    st.markdown("---")
                    st.subheader("📦 Batch Download All Reports")
                    
                    all_reports_zip = ""
                    for result in results:
                        report_text = f"""
RESUME ANALYSIS REPORT - {result['candidate_name']}
{"="*50}
CANDIDATE: {result['candidate_name']}
FILE TYPE: {'PDF' if result.get('file_type') == 'application/pdf' else 'TXT'}
FINAL VERDICT: {result.get('recommendation_summary', 'No analysis available')}
RECOMMENDATION SCORE: {result.get('recommendation_score', 0)}%
RELEVANCE SCORE: {result.get('relevance_score', 0)}%
SKILLS MATCH: {result.get('skills_match', 0)}%
EXPERIENCE LEVEL: {result.get('years_experience', 'Not Specified')}
EDUCATION LEVEL: {result.get('education_level', 'Not Specified')}

MATCHED SKILLS: {', '.join(result.get('matched_skills', []))}
MISSING SKILLS: {', '.join(result.get('missing_skills', []))}

RESUME QUALITY:
- Word Count: {result.get('word_count_status', 'N/A')}
- Repetition: {result.get('repetition_status', 'N/A')}
- Action Verbs: {'Yes' if result.get('uses_action_verbs') else 'No'}
- Quantifiable Results: {'Yes' if result.get('has_quantifiable_results') else 'No'}

IMPROVEMENT SUGGESTIONS:
{result.get('improvement_suggestions', 'No suggestions available.')}

PROFESSIONAL ASSESSMENT:
{result.get('recommendation_summary', 'No analysis available')}
{"="*50}

"""
                        all_reports_zip += report_text
                    
                    st.download_button(
                        label="📥 Download All Reports as Single File",
                        data=all_reports_zip,
                        file_name="all_candidates_resume_analysis_reports.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                else:
                    st.error("❌ No resumes were successfully analyzed. Please check your files and try again.")

# Add footer with information
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    <p>🔍 <strong>AI Resume Checker</strong> | Uses Gemini AI for precise resume analysis</p>
    <p>Supports both PDF and TXT file formats for resume analysis</p>
</div>
""", unsafe_allow_html=True)




