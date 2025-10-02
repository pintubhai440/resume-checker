# Required libraries are imported for the application
import streamlit as st
import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.prompts import PromptTemplate
from collections import Counter

# --- Helper Functions for Resume Quality Analysis ---
def get_word_count_status(text):
    """Analyze resume word count and provide status"""
    word_count = len(text.split())
    if word_count < 50: 
        return f"⚠️ Too Short ({word_count} words)"
    elif 50 <= word_count <= 1000: 
        return f"✅ Optimal Length ({word_count} words)"
    else: 
        return f"⚠️ Exceeded Max Limit ({word_count} words)"

def get_repetition_status(text):
    """Analyze keyword repetition in resume"""
    stop_words = {'the', 'in', 'or', 'and', 'a', 'an', 'to', 'is', 'of', 'for', 'with', 'on', 'it', 'i', 'was', 'are', 'as', 'at', 'be', 'by', 'that', 'this', 'from', 'my', 'we', 'our', 'you', 'your'}
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
st.set_page_config(layout="wide", page_title="AI Resume Checker", page_icon="🚀")
st.title("🚀 Advanced AI Resume Checker")
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
        with st.spinner('🔍 Gemini is performing comprehensive analysis... This might take 20-30 seconds.'):
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp", 
                temperature=0.1,  # Lower temperature for more consistent results
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            prompt_template_str = """
            You are an EXPERT AI hiring assistant with 10+ years of recruitment experience. Your task is to provide STRICTLY ACCURATE and REALISTIC analysis of a resume against a job description.

            **CRITICAL INSTRUCTIONS:**
            - Be EXTREMELY PRECISE with scoring - don't inflate scores unnecessarily
            - If resume has minimal or no relevant content, scores should reflect this realistically
            - Skills match percentage should be CALCULATED MATHEMATICALLY based on actual skills present
            - Only count skills that are EXPLICITLY mentioned in the resume
            - Do NOT assume or infer skills that are not clearly stated

            **ANALYSIS PROCESS (FOLLOW STRICTLY):**
            1. **SKILL EXTRACTION**: Extract ONLY skills that are explicitly mentioned in resume. No assumptions.
            2. **ELIGIBILITY CHECK**: Check graduation year, required qualifications mentioned in JD
            3. **EXPERIENCE CALCULATION**: Calculate years based on dates mentioned in resume
            4. **SKILL MATCH CALCULATION**: 
               - Count required skills from JD
               - Count matching skills from resume  
               - Calculate percentage: (matching_skills / required_skills) * 100
            5. **RELEVANCE SCORING**: Based on role alignment, experience relevance, qualification match
            6. **FINAL RECOMMENDATION**: Combine all factors for final score

            **SCORING GUIDELINES (BE STRICT):**
            - **0-30%**: Poor match (minimal relevant skills/experience)
            - **31-50%**: Below average (some relevant elements but major gaps)  
            - **51-70%**: Average match (moderate relevance with some gaps)
            - **71-85%**: Good match (strong alignment with minor gaps)
            - **86-100%**: Excellent match (near perfect alignment)

            **EDUCATION LEVEL CRITERIA:**
            - "High": Exact qualification match or higher
            - "Medium": Related field or partially matching
            - "Low": Does not meet educational requirements

            **RESPONSE FORMAT - STRICT JSON ONLY:**
            {{
                "relevance_score": integer (0-100),
                "skills_match": "X%", 
                "years_experience": "X years",
                "education_level": "High/Medium/Low",
                "matched_skills": ["skill1", "skill2", ...],
                "missing_skills": ["skill1", "skill2", ...],
                "recommendation_summary": "2-3 sentence realistic summary",
                "uses_action_verbs": true/false,
                "has_quantifiable_results": true/false,
                "recommendation_score": integer (0-100)
            }}

            Resume: {resume}
            Job Description: {jd}
            """

            prompt = PromptTemplate(input_variables=["resume", "jd"], template=prompt_template_str)
            chain = prompt | llm
            
            response_text = ""
            try:
                response = chain.invoke({"resume": resume_text, "jd": job_description})
                response_text = response.content.strip()
                
                # Improved JSON parsing
                json_match = re.search(r'\{[^{}]*\{.*\}[^{}]*\}|\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                    # Clean the JSON text
                    json_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_text)
                    analysis_result = json.loads(json_text)
                else:
                    st.error("❌ AI response format error. Could not extract JSON data.")
                    st.text_area("Raw AI Response for debugging:", response_text, height=200)
                    st.stop()
                
                # Validate required fields
                required_fields = ['relevance_score', 'skills_match', 'recommendation_score']
                for field in required_fields:
                    if field not in analysis_result:
                        st.error(f"❌ Missing required field: {field}")
                        st.stop()

                # Get quality metrics
                word_count_status = get_word_count_status(resume_text)
                repetition_status = get_repetition_status(resume_text)

                st.divider()
                st.header("📊 Detailed Analysis Results")

                # Final verdict with improved logic
                recommendation_score = analysis_result.get('recommendation_score', 0)
                if recommendation_score >= 80:
                    rec_color, rec_text = "green", "Highly Recommended"
                elif recommendation_score >= 60:
                    rec_color, rec_text = "orange", "Worth Considering"
                elif recommendation_score >= 40:
                    rec_color, rec_text = "red", "Marginal Fit"
                else:
                    rec_color, rec_text = "red", "Not Recommended"

                st.subheader(f"Final Verdict: :{rec_color}[{rec_text} ({recommendation_score}%)]")
                st.progress(recommendation_score / 100)

                # Key metrics in columns
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                with res_col1:
                    st.metric("AI Relevance Score", f"{analysis_result.get('relevance_score', 0)}%")
                with res_col2:
                    skills_match_value = analysis_result.get('skills_match', '0%')
                    st.metric("Skills Match", skills_match_value)
                with res_col3:
                    st.metric("Years' Experience", analysis_result.get('years_experience', 'Not specified'))
                with res_col4:
                    education_level = analysis_result.get('education_level', 'Not specified')
                    st.metric("Education Level", education_level)

                # Skills Analysis
                st.subheader("🔧 Skills Analysis")
                skill_col1, skill_col2 = st.columns(2)
                
                with skill_col1:
                    st.success("✅ Matched Skills")
                    matched_skills = analysis_result.get('matched_skills', [])
                    if matched_skills:
                        for skill in matched_skills:
                            st.write(f"• {skill}")
                    else:
                        st.write("No matching skills found")
                
                with skill_col2:
                    st.warning("❗️ Critical Missing Skills")
                    missing_skills = analysis_result.get('missing_skills', [])
                    if missing_skills:
                        for skill in missing_skills:
                            st.write(f"• {skill}")
                    else:
                        st.write("No major skill gaps identified")

                # Recommendation
                st.subheader("💡 Professional Assessment")
                recommendation = analysis_result.get('recommendation_summary', 'No analysis available.')
                st.info(recommendation)

                # Resume Quality Checks
                st.subheader("📝 Resume Quality Analysis")
                
                action_verbs = "✅ Yes" if analysis_result.get('uses_action_verbs') else "❌ No"
                quant_results = "✅ Yes" if analysis_result.get('has_quantifiable_results') else "❌ No"

                # Custom CSS for better metrics display
                st.markdown("""
                <style>
                .metric-card { 
                    background-color: #F8F9FA; 
                    border-radius: 10px; 
                    padding: 15px; 
                    text-align: center; 
                    border: 1px solid #E0E0E0;
                    margin: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
                    st.markdown(f'<div class="metric-card"><p class="label">Word Count</p><p class="value">{word_count_status}</p></div>', unsafe_allow_html=True)
                with quality_col2:
                    st.markdown(f'<div class="metric-card"><p class="label">Keyword Repetition</p><p class="value">{repetition_status}</p></div>', unsafe_allow_html=True)
                with quality_col3:
                    st.markdown(f'<div class="metric-card"><p class="label">Action Verbs</p><p class="value">{action_verbs}</p></div>', unsafe_allow_html=True)
                with quality_col4:
                    st.markdown(f'<div class="metric-card"><p class="label">Quantifiable Results</p><p class="value">{quant_results}</p></div>', unsafe_allow_html=True)

                st.divider()
                
                # Download Report
                report_text = f"""
ADVANCED RESUME ANALYSIS REPORT
================================

FINAL ASSESSMENT: {rec_text} ({recommendation_score}%)

KEY METRICS:
- AI Relevance Score: {analysis_result.get('relevance_score', 0)}%
- Skills Match: {analysis_result.get('skills_match', 'N/A')}
- Years of Experience: {analysis_result.get('years_experience', 'N/A')}
- Education Level: {analysis_result.get('education_level', 'N/A')}

PROFESSIONAL ASSESSMENT:
{analysis_result.get('recommendation_summary', '')}

RESUME QUALITY ANALYSIS:
- Word Count: {word_count_status}
- Keyword Repetition: {repetition_status}
- Uses Action Verbs: {"Yes" if analysis_result.get('uses_action_verbs') else "No"}
- Shows Quantifiable Results: {"Yes" if analysis_result.get('has_quantifiable_results') else "No"}

MATCHED SKILLS:
{chr(10).join(['• ' + skill for skill in analysis_result.get('matched_skills', ['None identified'])])}

MISSING CRITICAL SKILLS:
{chr(10).join(['• ' + skill for skill in analysis_result.get('missing_skills', ['None identified'])])}

---
Generated by Advanced AI Resume Checker
                """
                
                st.download_button(
                    label="📥 Download Comprehensive Report",
                    data=report_text,
                    file_name="detailed_resume_analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            except json.JSONDecodeError as e:
                st.error("❌ Failed to parse AI response. The analysis couldn't be completed.")
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
