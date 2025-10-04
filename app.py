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
    """Analyze resume word count with context for freshers."""
    word_count = len(text.split())
    if word_count < 250:
        return f"⚠️ Too Short ({word_count} words)"
    elif 250 <= word_count <= 600:
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
        'recommendation_score': 0
    }
    
    validated_result = required_fields.copy()
    validated_result.update(result)
    
    # Ensure scores are within bounds
    validated_result['relevance_score'] = max(0, min(100, validated_result['relevance_score']))
    validated_result['skills_match'] = max(0, min(100, validated_result['skills_match']))
    validated_result['recommendation_score'] = max(0, min(100, validated_result['recommendation_score']))
    
    return validated_result

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
        with st.spinner('🔍 Gemini Pro is performing a deep analysis... This might take a moment.'):
            # --- BRAIN UPGRADE: Using a more powerful model for higher accuracy ---
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro-preview-03-25", 
                temperature=0.1,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            try:
                # --- FIXED PROMPT: Enhanced with ELIGIBILITY CRITERIA priority ---
                analysis_prompt_template = """
CRITICAL INSTRUCTIONS: You MUST return ONLY a valid JSON object. No additional text, no explanations, no markdown.

You are an expert Senior Technical Recruiter. Analyze the RESUME against the JOB DESCRIPTION with brutal honesty.

**STRICT PRIORITY ORDER:**
1. **ELIGIBILITY CHECK FIRST**: Check graduation year and batch eligibility BEFORE anything else
2. **EXPERIENCE LEVEL**: Determine based on graduation year and work experience
3. **TECHNICAL SKILLS**: Only count skills EXPLICITLY mentioned in resume
4. **NO INFERENCES**: If not written, it doesn't exist

**BATCH ELIGIBILITY RULES:**
- If JD requires "2023 and earlier pass-outs" and candidate passed in 2015 → NOT ELIGIBLE
- If JD requires "2023 and earlier pass-outs" and candidate passed in 2024 → NOT ELIGIBLE  
- Only 2023, 2022, 2021, etc. are eligible for "2023 and earlier" requirement

**EXPERIENCE LEVEL CALCULATION:**
- Current Year: 2024
- "Fresher": Graduated in 2023-2024 (0-1 years experience)
- "Junior": Graduated in 2021-2022 (1-3 years experience)  
- "Mid-Level": Graduated in 2018-2020 (3-6 years experience)
- "Senior": Graduated in 2017 or earlier (6+ years experience)

**JOB DESCRIPTION:**
{jd}

**RESUME:**
{resume}

**ANALYSIS OUTPUT - RETURN ONLY THIS JSON:**
{{
    "relevance_score": 85,
    "skills_match": 80,
    "years_experience": "Fresher",
    "education_level": "High",
    "matched_skills": ["Python", "SQL"],
    "missing_skills": ["Spark", "Tableau"],
    "recommendation_summary": "Candidate meets basic qualifications but lacks key technical skills. Consider for junior roles with training.",
    "uses_action_verbs": true,
    "has_quantifiable_results": true,
    "recommendation_score": 65
}}

**SCORING LOGIC:**
The recommendation_score should be a balanced reflection of the relevance_score, skills_match, and the severity of missing skills. For intern roles, missing one or two key technologies should lower the score but not necessarily result in a 'Not Recommended' verdict if the foundational skills are strong.

**SCORING GUIDELINES FOR INELIGIBLE CANDIDATES:**
- If NOT ELIGIBLE due to batch criteria → recommendation_score MUST be 0-25%
- If NOT ELIGIBLE due to experience mismatch → recommendation_score MUST be 0-30%
- If ELIGIBLE but missing key skills → recommendation_score 30-60%
- If GOOD match → recommendation_score 60-85%  
- If EXCELLENT match → recommendation_score 85-100%

**EDUCATION LEVELS:**
- "High": B.Tech/BE/Masters from recognized institute
- "Medium": Bachelor's degree
- "Low": Diploma/No degree

RETURN ONLY THE JSON OBJECT:
"""
                analysis_prompt = PromptTemplate.from_template(analysis_prompt_template)
                analysis_chain = analysis_prompt | llm

                response = analysis_chain.invoke({"resume": resume_text, "jd": job_description})
                response_text = response.content

                # Debug: Show raw response
                with st.expander("🔧 Debug: Raw AI Response"):
                    st.code(response_text)

                cleaned_json = clean_json_response(response_text)
                if not cleaned_json:
                    st.error("❌ AI response format error. Could not extract JSON data.")
                    st.stop()

                analysis_result = json.loads(cleaned_json)
                analysis_result = validate_analysis_result(analysis_result)

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
                elif recommendation_score >= 40:
                    rec_color, rec_text = "red", "Not Recommended"
                else:
                    rec_color, rec_text = "red", "Strongly Not Recommended"

                st.subheader(f"Final Verdict: :{rec_color}[{rec_text} ({recommendation_score}%)]")
                st.progress(recommendation_score / 100)

                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                with res_col1:
                    st.metric("AI Relevance Score", f"{analysis_result.get('relevance_score', 0)}%")
                with res_col2:
                    st.metric("Skills Match", f"{analysis_result.get('skills_match', 0)}%")
                with res_col3:
                    st.metric("Years' Experience", analysis_result.get('years_experience', 'Not Specified'))
                with res_col4:
                    st.metric("Education Level", analysis_result.get('education_level', 'Not Specified'))

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
                
                # Generate comprehensive report
                report_text = f"""
RESUME ANALYSIS REPORT
=====================

FINAL VERDICT: {rec_text} ({recommendation_score}%)

CANDIDATE PROFILE:
- Experience Level: {analysis_result.get('years_experience', 'Not Specified')}
- Education Level: {analysis_result.get('education_level', 'Not Specified')}
- Relevance Score: {analysis_result.get('relevance_score', 0)}%
- Skills Match: {analysis_result.get('skills_match', 0)}%

SKILLS ANALYSIS:
✅ Matched Skills: {', '.join(analysis_result.get('matched_skills', []))}
❌ Missing Skills: {', '.join(analysis_result.get('missing_skills', []))}

PROFESSIONAL ASSESSMENT:
{analysis_result.get('recommendation_summary', 'No analysis available.')}

RESUME QUALITY:
- Word Count: {word_count_status}
- Keyword Repetition: {repetition_status}
- Action Verbs: {action_verbs}
- Quantifiable Results: {quant_results}

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
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


