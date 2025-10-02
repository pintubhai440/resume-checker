# Required libraries are imported for the application
import streamlit as st
import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.prompts import PromptTemplate
from collections import Counter
import time # Added for potential retries or delays if needed

# --- Helper Functions for Resume Quality Analysis ---

def get_word_count_status(text):
    """Analyze resume word count and provide status"""
    word_count = len(text.split())
    if word_count < 150:
        return f"⚠️ Too Short ({word_count} words)"
    elif 150 <= word_count <= 800:
        return f"✅ Optimal Length ({word_count} words)"
    else:
        return f"⚠️ Too Long ({word_count} words)"

def get_repetition_status(text):
    """Analyze keyword repetition in resume with an improved stop-word list."""
    stop_words = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
        'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do',
        'does', 'doing', 'don', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have',
        'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into',
        'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of',
        'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same',
        'she', 'should', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves',
        'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very',
        'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you',
        'your', 'yours', 'yourself', 'yourselves', 'experience', 'work', 'project', 'company', 'team', 'role', 'worked',
        'responsibilities', 'development', 'used', 'using'
    }
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = [word for word in clean_text.split() if word not in stop_words and not word.isdigit()]

    if len(words) < 20:  # Not enough content to analyze
        return "✅ Good keyword distribution"
        
    word_counts = Counter(words)
    if not word_counts:
        return "✅ Good keyword distribution"

    total_words = len(words)
    most_common_word, count = word_counts.most_common(1)[0]
    repetition_percentage = (count / total_words) * 100
    
    if repetition_percentage > 4.5:
        return f"⚠️ High repetition of '{most_common_word.title()}'"
    return "✅ Good keyword distribution"

def clean_json_response(response_text):
    """Extracts and cleans a JSON object from a string."""
    # Find the start and end of the JSON object
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not json_match:
        return None
    
    json_text = json_match.group(0)
    # Remove control characters and trailing commas
    json_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_text)
    json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
    return json_text

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
                model="gemini-1.5-flash",
                temperature=0.2, # Slightly increased for better descriptive text
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # --- STEP 1: Extract Skills from JD and Resume using a focused prompt ---
            try:
                st.write("Step 1: Extracting skills from documents...")
                skill_extraction_prompt_template = """
                Analyze the Job Description and Resume to identify key skills.
                Return ONLY a raw JSON object with two keys: "required_skills" from the Job Description and "candidate_skills" from the Resume.
                Extract skills accurately and concisely.

                JOB DESCRIPTION:
                {jd}

                RESUME:
                {resume}

                EXAMPLE OUTPUT FORMAT:
                {{
                    "required_skills": ["Python", "Data Analysis", "Machine Learning", "SQL", "Tableau"],
                    "candidate_skills": ["Python", "SQL", "Excel", "Project Management"]
                }}
                """
                skill_prompt = PromptTemplate.from_template(skill_extraction_prompt_template)
                skill_chain = skill_prompt | llm
                
                skill_response_str = skill_chain.invoke({"resume": resume_text, "jd": job_description}).content
                cleaned_skill_json = clean_json_response(skill_response_str)

                if not cleaned_skill_json:
                    st.error("Error: Could not extract skills from the AI response in Step 1.")
                    st.text_area("Raw Skill Extraction Response:", skill_response_str)
                    st.stop()

                skill_data = json.loads(cleaned_skill_json)
                required_skills = skill_data.get("required_skills", [])
                candidate_skills = skill_data.get("candidate_skills", [])

                if not required_skills:
                    st.warning("Could not identify specific required skills in the Job Description. Skill analysis will be limited.")
            except Exception as e:
                st.error(f"Failed during skill extraction phase (Step 1): {e}")
                st.stop()

            # --- STEP 2: Calculate Skill Match in Python for 100% Accuracy ---
            st.write("Step 2: Performing accurate skill comparison...")
            matched_skills = []
            missing_skills = []
            skills_match_percentage = 0
            
            if required_skills:
                required_skills_set = {skill.lower().strip() for skill in required_skills}
                candidate_skills_set = {skill.lower().strip() for skill in candidate_skills}

                matched_skills_set = required_skills_set.intersection(candidate_skills_set)
                missing_skills_set = required_skills_set.difference(candidate_skills_set)
                
                matched_skills = sorted([s.title() for s in matched_skills_set])
                missing_skills = sorted([s.title() for s in missing_skills_set])
                
                skills_match_percentage = round((len(matched_skills_set) / len(required_skills_set)) * 100) if required_skills_set else 0
            
            # --- STEP 3: Get Qualitative Analysis from LLM with Pre-calculated Data ---
            try:
                st.write("Step 3: Generating final AI-powered assessment...")
                analysis_prompt_template = """
                You are an expert HR analyst. Analyze the resume against the job description based on the pre-calculated skill analysis provided.
                Provide your final evaluation ONLY in a raw JSON format.

                **JOB DESCRIPTION:**
                {jd}

                **RESUME:**
                {resume}

                **PRE-CALCULATED SKILL ANALYSIS (Use this as ground truth):**
                - Matched Skills: {matched_skills}
                - Missing Critical Skills: {missing_skills}
                - Skill Match Percentage: {skills_match_percentage}%

                **YOUR TASK:**
                Based on all the provided information, complete the following JSON structure.
                - Be realistic and critical in your scoring.
                - The recommendation summary should be concise and actionable for a hiring manager.
                - Calculate `years_experience` from the resume.
                - Determine `education_level` match ('High', 'Medium', 'Low').
                - Evaluate `uses_action_verbs` and `has_quantifiable_results` (true/false).

                **RETURN ONLY RAW JSON IN THIS EXACT FORMAT:**
                {{
                    "relevance_score": 85,
                    "years_experience": "3 years",
                    "education_level": "High",
                    "recommendation_summary": "The candidate is a strong fit due to their solid experience in key areas like Python and SQL. While they lack experience in Cloud Computing, their foundational skills make them a trainable asset. Recommended for an interview.",
                    "uses_action_verbs": true,
                    "has_quantifiable_results": false,
                    "recommendation_score": 80
                }}
                """
                analysis_prompt = PromptTemplate.from_template(analysis_prompt_template)
                analysis_chain = analysis_prompt | llm
                
                response = analysis_chain.invoke({
                    "resume": resume_text,
                    "jd": job_description,
                    "matched_skills": ", ".join(matched_skills) if matched_skills else "None",
                    "missing_skills": ", ".join(missing_skills) if missing_skills else "None",
                    "skills_match_percentage": skills_match_percentage
                })
                response_text = response.content
                
                cleaned_analysis_json = clean_json_response(response_text)
                if not cleaned_analysis_json:
                    st.error("❌ AI response format error in final analysis (Step 3).")
                    st.text_area("Raw AI Analysis Response for debugging:", response_text)
                    st.stop()
                    
                analysis_result = json.loads(cleaned_analysis_json)
                
                # --- DISPLAY RESULTS ---
                # Add the accurately calculated skill data to the final result for display
                analysis_result['skills_match'] = f"{skills_match_percentage}%"
                analysis_result['matched_skills'] = matched_skills
                analysis_result['missing_skills'] = missing_skills

                # Get quality metrics
                word_count_status = get_word_count_status(resume_text)
                repetition_status = get_repetition_status(resume_text)

                st.divider()
                st.header("📊 Detailed Analysis Results")

                # Final verdict
                recommendation_score = analysis_result.get('recommendation_score', 0)
                if recommendation_score >= 80:
                    rec_color, rec_text = "green", "Highly Recommended"
                elif recommendation_score >= 60:
                    rec_color, rec_text = "orange", "Worth Considering"
                else:
                    rec_color, rec_text = "red", "Not Recommended"

                st.subheader(f"Final Verdict: :{rec_color}[{rec_text} ({recommendation_score}%)]")
                st.progress(recommendation_score / 100)

                # Key metrics in columns
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                with res_col1:
                    st.metric("AI Relevance Score", f"{analysis_result.get('relevance_score', 0)}%")
                with res_col2:
                    st.metric("Skills Match", analysis_result.get('skills_match', '0%'))
                with res_col3:
                    st.metric("Years' Experience", analysis_result.get('years_experience', 'N/A'))
                with res_col4:
                    st.metric("Education Level", analysis_result.get('education_level', 'N/A'))

                # Skills Analysis
                st.subheader("🔧 Skills Analysis")
                
                # Custom CSS for skill tags to make them look like nice badges
                st.markdown("""
                <style>
                .skill-tag-container {
                    line-height: 2.2;
                }
                .skill-tag {
                    display: inline-block;
                    padding: 4px 12px;
                    margin: 4px 3px;
                    font-size: 14px;
                    background-color: #FFF0F0;
                    color: #D32F2F;
                    border: 1px solid #FFCDD2;
                    border-radius: 16px;
                    font-weight: 500;
                }
                .skill-tag-matched {
                    display: inline-block;
                    padding: 4px 12px;
                    margin: 4px 3px;
                    font-size: 14px;
                    background-color: #E8F5E9;
                    color: #2E7D32;
                    border: 1px solid #C8E6C9;
                    border-radius: 16px;
                    font-weight: 500;
                }
                </style>
                """, unsafe_allow_html=True)
                
                skill_col1, skill_col2 = st.columns(2)
                
                with skill_col1:
                    st.success("✅ Matched Skills")
                    if analysis_result['matched_skills']:
                        # Generate HTML for each skill tag and join them together
                        skills_html = "".join([f'<span class="skill-tag-matched">{skill}</span>' for skill in analysis_result['matched_skills']])
                        st.markdown(f'<div class="skill-tag-container">{skills_html}</div>', unsafe_allow_html=True)
                    else:
                        st.write("No matching skills found.")
                
                with skill_col2:
                    st.warning("❗️ Critical Missing Skills")
                    if analysis_result['missing_skills']:
                        # Generate HTML for each skill tag and join them together
                        skills_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in analysis_result['missing_skills']])
                        st.markdown(f'<div class="skill-tag-container">{skills_html}</div>', unsafe_allow_html=True)
                    else:
                        st.write("No major skill gaps identified.")

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
                
                # Download Report
                report_text = f"""
ADVANCED RESUME ANALYSIS REPORT
================================

FINAL ASSESSMENT: {rec_text} ({recommendation_score}%)

KEY METRICS:
- AI Relevance Score: {analysis_result.get('relevance_score', 'N/A')}%
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
                st.error(f"❌ JSON parsing error in final analysis (Step 3): {str(e)}")
                st.text_area("Raw AI Response for debugging:", response_text, height=200)
            except Exception as e:
                st.error(f"❌ An unexpected error occurred in final analysis (Step 3): {str(e)}")
                st.text_area("Raw AI Response for debugging:", response_text, height=200)

# Add footer with information
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    <p>🔍 <strong>Advanced AI Resume Checker</strong> | Uses Gemini AI for precise resume analysis</p>
    <p>Provides realistic scoring based on actual content matching between resume and job requirements</p>
</div>
""", unsafe_allow_html=True)
