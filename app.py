import streamlit as st
import os
import json
import re 
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from collections import Counter

# --- Helper Functions for Resume Analysis ---

def get_word_count_status(text):
    """Checks the word count and returns a detailed status message."""
    word_count = len(text.split())
    if word_count < 50:
        return f"⚠️ Too Short ({word_count} words)"
    elif 50 <= word_count <= 1000:
        return f"✅ Optimal Length ({word_count} words)"
    else: # More than 1000 words
        return f"⚠️ Exceeded Max Limit ({word_count} words)"

def get_repetition_status(text):
    """
    Checks for keyword repetition, ignoring common stop words and punctuation,
    to ensure good keyword distribution.
    """
    stop_words = {
        'the', 'in', 'or', 'and', 'a', 'an', 'to', 'is', 'of', 'for', 'with', 'on', 'it', 'i', 'was',
        'are', 'as', 'at', 'be', 'by', 'that', 'this', 'from', 'my', 'we', 'our', 'you', 'your'
    }
    # Clean text by removing punctuation and converting to lowercase
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = [word for word in clean_text.split() if word not in stop_words]
    
    if not words:
        return "✅ Good keyword distribution"
    
    word_counts = Counter(words)
    total_words = len(words)
    # Find the most common word and its frequency
    most_common_word, count = word_counts.most_common(1)[0]
    repetition_percentage = (count / total_words) * 100
    
    # Flag if a single keyword makes up more than 5% of the text
    if repetition_percentage > 5:
        return f"⚠️ High repetition of '{most_common_word}'"
    return "✅ Good keyword distribution"

# --- UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Resume Checker")
st.title("🚀 AI Resume Checker")
st.write("Analyze and score resumes against job requirements with Gemini AI. This tool provides a relevance score, skill gap analysis, and more.")

# --- API KEY & MODEL SETUP ---
# CRITICAL: Use Streamlit's secrets management for the API key.
# This is a secure way to handle sensitive information.
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("🤫 Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()
    
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- LAYOUT ---
col1, col2 = st.columns(2, gap="large")
with col1:
    st.header("📄 Job Requirements")
    job_description = st.text_area("Paste the Job Description here", height=350, label_visibility="collapsed")
with col2:
    st.header("👤 Candidate's Resume")
    resume_text = st.text_area("Paste the Resume Text here", height=350, label_visibility="collapsed")

# --- ANALYSIS BUTTON & LOGIC ---
if st.button("Analyze with Gemini AI", use_container_width=True, type="primary"):
    
    if not resume_text or not job_description:
        st.warning("Please provide both the Job Description and the Resume text.")
    else:
        with st.spinner('Gemini is performing a deep analysis... This might take a moment.'):
            # Initialize the generative AI model
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                temperature=0.3,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                },
            )
            
            # Define the prompt template for the AI
            prompt_template_str = """
            You are an expert AI hiring assistant. Analyze the following resume and job description.
            
            IMPORTANT CONTEXT: The current date is September 21, 2025. When you see an experience range like "2022 - Present", you must calculate the duration precisely. From a start date in 2022 to the current date, the experience is nearly 3 years.
            
            Provide ONLY a raw JSON response with the following keys. Do not include any other text, formatting, or markdown backticks.
            - "relevance_score": An integer score from 0 to 100 representing how well the resume matches the job description.
            - "skills_match": A percentage string like "85%".
            - "years_experience": A string representing the candidate's total relevant experience (e.g., "Almost 3 years" or "5+ years").
            - "education_level": A brief alignment description like "High", "Medium", or "Low".
            - "matched_skills": A list of up to 7 skills the candidate has that are required by the job.
            - "missing_skills": A list of up to 3 critical skills the candidate is missing.
            - "recommendation_summary": A 2-sentence summary of the candidate's fit.
            - "uses_action_verbs": A boolean (true or false).
            - "has_quantifiable_results": A boolean (true or false).
            - "recommendation_score": An integer from 0 to 100, representing the overall confidence in recommending this candidate based on all factors combined (relevance, skills, experience, and resume quality).

            Resume: {resume}
            Job Description: {jd}
            """

            prompt = PromptTemplate(input_variables=["resume", "jd"], template=prompt_template_str)
            chain = RunnableSequence(prompt, llm)
            
            try:
                response = chain.invoke({"resume": resume_text, "jd": job_description})
                response_text = response.content
                
                # Clean the response to ensure it's valid JSON
                start_index = response_text.find('{')
                end_index = response_text.rfind('}') + 1
                if start_index != -1 and end_index != -1:
                    json_text = response_text[start_index:end_index]
                    analysis_result = json.loads(json_text)
                else:
                    raise ValueError("No valid JSON found in the AI's response.")
                
                # Perform additional checks
                word_count_status = get_word_count_status(resume_text)
                repetition_status = get_repetition_status(resume_text)

                st.divider()
                st.header("📊 Analysis Results")

                # --- NEW FEATURE: RECOMMENDATION SCORE ---
                recommendation_score = analysis_result.get('recommendation_score', 0)
                if recommendation_score >= 75:
                    rec_color = "green"
                    rec_text = "Highly Recommended"
                elif recommendation_score >= 50:
                    rec_color = "orange"
                    rec_text = "Worth Considering"
                else:
                    rec_color = "red"
                    rec_text = "Not a Strong Fit"

                st.subheader(f"Final Verdict: :{rec_color}[{rec_text}]")
                st.progress(recommendation_score)
                # --- END OF NEW FEATURE ---

                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                res_col1.metric("AI Relevance Score", f"{analysis_result.get('relevance_score', 0)}%")
                res_col2.metric("Skills Match", analysis_result.get('skills_match', 'N/A'))
                res_col3.metric("Years' Experience", analysis_result.get('years_experience', 'N/A'))
                res_col4.metric("Education Level", analysis_result.get('education_level', 'N/A'))

                st.subheader("Skills Analysis")
                skill_col1, skill_col2 = st.columns(2)
                with skill_col1:
                    st.success("✅ Matched Skills")
                    st.write(", ".join(analysis_result.get('matched_skills', ["Not found"])))
                with skill_col2:
                    st.warning("❗️ Missing Skills")
                    st.write(", ".join(analysis_result.get('missing_skills', ["None found"])))

                st.subheader("💡 Recommendation")
                st.info(analysis_result.get('recommendation_summary', 'No summary available.'))
                
                st.subheader("Resume Quality Checks")
                action_verbs = "✅ Yes" if analysis_result.get('uses_action_verbs') else "⚠️ No"
                quant_results = "✅ Yes" if analysis_result.get('has_quantifiable_results') else "⚠️ No"
                
                add_col1, add_col2, add_col3, add_col4 = st.columns(4)
                with add_col1:
                    st.metric("Word Count", word_count_status)
                with add_col2:
                    st.metric("Keyword Repetition", repetition_status)
                with add_col3:
                    st.metric("Uses Action Verbs?", action_verbs)
                with add_col4:
                    st.metric("Quantifiable Results?", quant_results)
                
                st.divider()

                # --- DOWNLOAD REPORT ---
                report_text = f"""
AI RESUME ANALYSIS REPORT
=========================
FINAL VERDICT: {rec_text} ({recommendation_score}%)
AI RELEVANCE SCORE: {analysis_result.get('relevance_score', 0)}%
SKILLS MATCH: {analysis_result.get('skills_match', 'N/A')}
YEARS' EXPERIENCE: {analysis_result.get('years_experience', 'N/A')}

RECOMMENDATION:
{analysis_result.get('recommendation_summary', '')}

RESUME QUALITY CHECKS:
- Word Count: {word_count_status}
- Repetition: {repetition_status}
- Uses Action Verbs: {"Yes" if analysis_result.get('uses_action_verbs') else "No"}
- Shows Quantifiable Results: {"Yes" if analysis_result.get('has_quantifiable_results') else "No"}

MATCHED SKILLS:
- {', '.join(analysis_result.get('matched_skills', []))}

MISSING SKILLS:
- {', '.join(analysis_result.get('missing_skills', []))}
"""
                st.download_button(
                    label="⬇️ Download Full Report",
                    data=report_text,
                    file_name="resume_analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            except json.JSONDecodeError:
                st.error("Error: Could not decode the JSON response from the AI. The model may have returned an unexpected format.")
                st.text_area("Raw AI Response for debugging:", response_text, height=150)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

