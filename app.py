import streamlit as st
import os
import json
import re 
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.prompts import PromptTemplate
from collections import Counter

# --- Helper Functions for Resume Analysis ---

def get_word_count_status(text):
    """Checks the word count and returns a detailed status message."""
    word_count = len(text.split())
    if word_count < 50:
        return f"⚠️ Too Short ({word_count} words)"
    elif 50 <= word_count <= 1000:
        return f"✅ Optimal Length ({word_count} words)"
    else:
        return f"⚠️ Exceeded Max Limit ({word_count} words)"

def get_repetition_status(text):
    """Checks for keyword repetition to ensure good keyword distribution."""
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
st.set_page_config(layout="wide", page_title="AI Resume Checker")
st.title("🚀 AI Resume Checker")
st.write("Analyze a resume against a job description to get instant insights.")

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
    job_description = st.text_area("Job Description", height=350, label_visibility="collapsed")
with col2:
    st.header("👤 Resume Content")
    resume_text = st.text_area("Paste Resume Text", height=350, label_visibility="collapsed")

# --- ANALYSIS BUTTON & LOGIC ---
if st.button("Analyze with Gemini AI", use_container_width=True, type="primary"):
    if not resume_text or not job_description:
        st.warning("Please provide both Job Description and Resume text.")
    else:
        with st.spinner('Gemini is performing a deep analysis...'):
            try:
                # Initialize the model
                llm = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    temperature=0.3,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    },
                )
                
                prompt_template_str = """
                You are an expert AI hiring assistant. Analyze the resume and job description.
                The current date is September 21, 2025.
                Provide ONLY a raw JSON response with these keys:
                - "relevance_score": An integer (0-100).
                - "skills_match": A percentage string ("85%").
                - "years_experience": A string for relevant experience.
                - "education_level": A brief description ("High", "Medium", "Low").
                - "matched_skills": A list of up to 7 matching skills.
                - "missing_skills": A list of up to 3 critical missing skills.
                - "recommendation_summary": A 2-sentence summary.
                - "uses_action_verbs": A boolean.
                - "has_quantifiable_results": A boolean.
                - "recommendation_score": An integer (0-100) for overall recommendation confidence.

                Resume: {resume}
                Job Description: {jd}
                """
                
                # Create prompt and get response
                prompt = PromptTemplate(input_variables=["resume", "jd"], template=prompt_template_str)
                formatted_prompt = prompt.format(resume=resume_text, jd=job_description)
                
                # Get response from Gemini
                response = llm.invoke(formatted_prompt)
                
                # Check if response exists and has content
                if not response or not hasattr(response, 'content'):
                    st.error("No response received from Gemini AI")
                    st.stop()
                
                response_text = response.content
                
                # Debug: Show raw response (collapsible)
                with st.expander("Debug: Raw AI Response"):
                    st.text_area("Raw Response", response_text, height=200)
                
                # Extract JSON from response
                start_index = response_text.find('{')
                end_index = response_text.rfind('}') + 1

                if start_index != -1 and end_index != -1:
                    json_text = response_text[start_index:end_index]
                    
                    try:
                        analysis_result = json.loads(json_text)
                        
                        st.divider()
                        st.header("📊 Analysis Results")

                        recommendation_score = analysis_result.get('recommendation_score', 0)
                        if recommendation_score >= 75:
                            rec_color, rec_text = "green", "Highly Recommended"
                        elif recommendation_score >= 50:
                            rec_color, rec_text = "orange", "Worth Considering"
                        else:
                            rec_color, rec_text = "red", "Not a Strong Fit"

                        st.subheader(f"Final Verdict: :{rec_color}[{rec_text}]")
                        st.progress(recommendation_score / 100)

                        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                        res_col1.metric("AI Relevance Score", f"{analysis_result.get('relevance_score', 0)}%")
                        res_col2.metric("Skills Match", analysis_result.get('skills_match', 'N/A'))
                        res_col3.metric("Years' Experience", analysis_result.get('years_experience', 'N/A'))
                        res_col4.metric("Education Level", analysis_result.get('education_level', 'N/A'))

                        st.subheader("Skills Analysis")
                        skill_col1, skill_col2 = st.columns(2)
                        with skill_col1:
                            st.success("✅ Matched Skills")
                            matched_skills = analysis_result.get('matched_skills', ["Not found"])
                            if isinstance(matched_skills, list):
                                st.write(", ".join(matched_skills))
                            else:
                                st.write(str(matched_skills))
                        with skill_col2:
                            st.warning("❗️ Missing Skills")
                            missing_skills = analysis_result.get('missing_skills', ["None found"])
                            if isinstance(missing_skills, list):
                                st.write(", ".join(missing_skills))
                            else:
                                st.write(str(missing_skills))

                        st.subheader("💡 AI Recommendation")
                        st.info(analysis_result.get('recommendation_summary', 'No summary available.'))
                        
                        st.subheader("Resume Quality Checks")
                        word_count_status = get_word_count_status(resume_text)
                        repetition_status = get_repetition_status(resume_text)
                        action_verbs = "✅ Yes" if analysis_result.get('uses_action_verbs') else "⚠️ No"
                        quant_results = "✅ Yes" if analysis_result.get('has_quantifiable_results') else "⚠️ No"
                        
                        add_col1, add_col2, add_col3, add_col4 = st.columns(4)
                        add_col1.metric("Word Count", word_count_status.split()[0], word_count_status.split()[1]) 
                        add_col2.metric("Keyword Repetition", repetition_status.split()[0], " ".join(repetition_status.split()[1:]))
                        add_col3.metric("Uses Action Verbs?", action_verbs)
                        add_col4.metric("Quantifiable Results?", quant_results)

                        st.divider()
                        # Create report text
                        report_text = f"""
AI RESUME ANALYSIS REPORT
=========================
FINAL VERDICT: {rec_text} (Score: {recommendation_score}%)
AI RELEVANCE SCORE: {analysis_result.get('relevance_score', 0)}%
SKILLS MATCH: {analysis_result.get('skills_match', 'N/A')}
YEARS' EXPERIENCE: {analysis_result.get('years_experience', 'N/A')}

RECOMMENDATION:
{analysis_result.get('recommendation_summary', '')}

MATCHED SKILLS:
- {', '.join([str(skill) for skill in analysis_result.get('matched_skills', [])])}

MISSING SKILLS:
- {', '.join([str(skill) for skill in analysis_result.get('missing_skills', [])])}
"""
                        st.download_button(
                            label="⬇️ Download Full Report",
                            data=report_text,
                            file_name="resume_analysis_report.txt",
                            mime="text/plain"
                        )

                    except json.JSONDecodeError as e:
                        st.error(f"Failed to parse JSON response: {e}")
                        st.write("JSON text that failed to parse:", json_text)
                        
                else:
                    st.error("No valid JSON found in the AI's response.")
                    st.write("Please check the raw response in the debug section above.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                st.write("Please check your API key and try again.")
