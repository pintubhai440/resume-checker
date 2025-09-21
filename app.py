import streamlit as st
import os
import json
import re 
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from collections import Counter

# Final Word Count Function
def get_word_count_status(text):
    """Checks the word count and returns a detailed status message based on new limits."""
    word_count = len(text.split())
    if word_count < 50:
        return f"⚠️ Too Short ({word_count} words)"
    elif word_count < 150:
        return f"✅ Optimal Length ({word_count} words)"
    elif word_count <= 400:
        return f"✅ Optimal Length ({word_count} words)"
    elif word_count <= 1000:
        return f"✅ Optimal Length ({word_count} words)"
    else: # More than 1200 words
        return f"⚠️ Exceeded Max Limit ({word_count} words)"

# Final Repetition Function
def get_repetition_status(text):
    """
    Checks for keyword repetition, ignoring common stop words and punctuation.
    """
    stop_words = {
        'the', 'in', 'or', 'and', 'a', 'an', 'to', 'is', 'of', 'for', 'with', 'on', 'it', 'i', 'was',
        'are', 'as', 'at', 'be', 'by', 'that', 'this', 'from', 'my', 'my', 'we', 'our', 'you', 'your'
    }
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

# === UI SETUP ===
st.set_page_config(layout="wide")
st.title("🚀 AI Resume Checker")
st.write("This advanced tool to analyze and score resumes against job requirements.")

# === API KEY SETUP ===
# Your API key is now included here.
API_KEY = "AIzaSyBbAMtle5v_GKSzQOtSb-oX9HF09lHnEzY" 

# === LAYOUT ===
col1, col2 = st.columns(2, gap="large")
with col1:
    st.header("📄 Job Requirements")
    job_description = st.text_area("Job Description", height=350, label_visibility="collapsed")
with col2:
    st.header("👤 Resume Content")
    resume_text = st.text_area("Paste Resume Text", height=350, label_visibility="collapsed")

# === ANALYSIS BUTTON & LOGIC ===
if st.button("Analyze with Gemini AI", use_container_width=True, type="primary"):
    
    # This check is still here as a safeguard, but it will pass now.
    if API_KEY == "YOUR_OWN_API_KEY_HERE" or not API_KEY:
        st.error("Please enter your Google API Key in the code to proceed.")
    elif not resume_text or not job_description:
        st.warning("Please provide both Job Description and Resume.")
    else:
        os.environ["GOOGLE_API_KEY"] = API_KEY
        
        with st.spinner('Gemini is performing a deep analysis...'):
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
            
            prompt_template_str = """
            Analyze the following resume and job description.
            
            IMPORTANT CONTEXT: The current date is September 21, 2025. When you see an experience range like "2022 - Present", you must calculate the duration precisely. From a start date in 2022 to the current date, the experience is nearly 3 years.
            
            Provide ONLY a raw JSON response with the following keys and no other text or formatting:
            - "relevance_score": An integer score from 0 to 100.
            - "skills_match": A percentage string like "85%".
            - "years_experience": A string representing the candidate's calculated years of experience. Be precise (e.g., "Almost 3 years" or "2.5+ years").
            - "education_level": A brief description like "High" or "Medium".
            - "matched_skills": A list of up to 7 matching skills.
            - "missing_skills": A list of up to 3 missing skills.
            - "recommendation_summary": A 2-sentence summary of the candidate's fit.
            - "uses_action_verbs": A boolean (true or false).
            - "has_quantifiable_results": A boolean (true or false).
            
            Resume: {resume}
            Job Description: {jd}
            """
            
            prompt = PromptTemplate(input_variables=["resume", "jd"], template=prompt_template_str)
            chain = RunnableSequence(prompt, llm)
            
            try:
                response = chain.invoke({"resume": resume_text, "jd": job_description})
                response_text = response.content
                
                start_index = response_text.find('{')
                end_index = response_text.rfind('}') + 1
                if start_index != -1 and end_index != -1:
                    json_text = response_text[start_index:end_index]
                    analysis_result = json.loads(json_text)
                else:
                    raise ValueError("No valid JSON found in the AI's response.")
                
                word_count_status = get_word_count_status(resume_text)
                repetition_status = get_repetition_status(resume_text)
                
                st.divider()
                st.header("📊 Analysis Results")
                
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                res_col1.metric("AI Relevance Score", f"{analysis_result.get('relevance_score', 0)}%")
                res_col2.metric("Skills Match", analysis_result.get('skills_match', 'N/A'))
                res_col3.metric("Years' Experience", analysis_result.get('years_experience', 'N/A'))
                res_col4.metric("Education Level", analysis_result.get('education_level', 'N/A'))

                st.subheader("Skills Analysis")
                skill_col1, skill_col2 = st.columns(2)
                with skill_col1:
                    st.success("✅ Matched Skills")
                    st.write(", ".join(analysis_result.get('matched_skills', [])))
                with skill_col2:
                    st.warning("❗️ Missing Skills")
                    st.write(", ".join(analysis_result.get('missing_skills', [])))

                st.subheader("💡 Recommendation")
                st.info(analysis_result.get('recommendation_summary', 'No summary available.'))
                
                st.subheader("Additional Checks")
                action_verbs = "✅ Yes" if analysis_result.get('uses_action_verbs') else "⚠️ No"
                quant_results = "✅ Yes" if analysis_result.get('has_quantifiable_results') else "⚠️ No"
                
                add_col1, add_col2, add_col3, add_col4 = st.columns(4)
                with add_col1:
                    st.info(f"**Word Count:** {word_count_status}")
                with add_col2:
                    st.info(f"**Repetition:** {repetition_status}")
                with add_col3:
                    st.info(f"**Uses Action Verbs:** {action_verbs}")
                with add_col4:
                    st.info(f"**Shows Quantifiable Results:** {quant_results}")
                
                st.divider()
                report_text = f"""
AI RESUME ANALYSIS REPORT
=========================
AI RELEVANCE SCORE: {analysis_result.get('relevance_score', 0)}%
SKILLS MATCH: {analysis_result.get('skills_match', 'N/A')}
YEARS' EXPERIENCE: {analysis_result.get('years_experience', 'N/A')}

RECOMMENDATION:
{analysis_result.get('recommendation_summary', '')}

ADDITIONAL CHECKS:
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
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")