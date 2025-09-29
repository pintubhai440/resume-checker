# Zaroori libraries ko import karna
import streamlit as st
import os
import json
import re 
# FIX: Google ki jagah OpenAI ki library import ki gayi hai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from collections import Counter

# --- Helper Functions (Resume ki quality check karne ke liye) ---

def get_word_count_status(text):
    """Shabdon ki ginti check karta hai."""
    word_count = len(text.split())
    if word_count < 50:
        return f"⚠️ Too Short ({word_count} words)"
    elif 50 <= word_count <= 1000:
        return f"✅ Optimal Length ({word_count} words)"
    else:
        return f"⚠️ Exceeded Max Limit ({word_count} words)"

def get_repetition_status(text):
    """Resume mein keywords ke repetition ko check karta hai."""
    stop_words = {'the', 'in', 'or', 'and', 'a', 'an', 'to', 'is', 'of', 'for', 'with', 'on', 'it', 'i', 'was', 'are', 'as', 'at', 'be', 'by', 'that', 'this', 'from', 'my', 'we', 'our', 'you', 'your'}
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = [word for word in clean_text.split() if word not in stop_words]
    if not words:
        return "✅ Good"
    word_counts = Counter(words)
    total_words = len(words)
    most_common_word, count = word_counts.most_common(1)[0]
    repetition_percentage = (count / total_words) * 100
    if repetition_percentage > 5:
        return f"⚠️ High: '{most_common_word}'"
    return "✅ Good"

# --- UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Resume Checker", page_icon="🚀")
st.title("🚀 AI Resume Checker")
st.write("Analyze a resume against a job description to get instant, powerful insights.")

# --- API KEY SETUP ---
# FIX: Ab hum OpenAI ki API key dhoondh rahe hain
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
except (FileNotFoundError, KeyError):
    st.error("🤫 OpenAI API Key not found. Please add it to your Streamlit secrets.")
    st.stop()

# --- LAYOUT ---
col1, col2 = st.columns(2, gap="large")
with col1:
    st.header("📄 Job Requirements")
    job_description = st.text_area("Job Description", height=350, label_visibility="collapsed", placeholder="Paste the job description here...", key="job_description_input")
with col2:
    st.header("👤 Resume Content")
    resume_text = st.text_area("Paste Resume Text", height=350, label_visibility="collapsed", placeholder="Paste the candidate's resume here...", key="resume_text_input")

# --- ANALYSIS BUTTON & LOGIC ---
if st.button("Analyze with AI", use_container_width=True, type="primary"):
    if not resume_text or not job_description:
        st.warning("Please provide both the Job Description and the Resume text.")
    else:
        with st.spinner('The AI is performing a deep analysis...'):
            # FIX: Google ki jagah OpenAI ka model set kiya gaya hai
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo", 
                temperature=0.3
            )
            
            prompt_template_str = """
            You are an expert AI hiring assistant. Analyze the resume against the job description.
            Provide ONLY a raw JSON response with these keys: "relevance_score", "skills_match", "years_experience", "education_level", "matched_skills", "missing_skills", "recommendation_summary", "uses_action_verbs", "has_quantifiable_results", "recommendation_score".

            Resume: {resume}
            Job Description: {jd}
            """
            prompt = PromptTemplate(input_variables=["resume", "jd"], template=prompt_template_str)
            
            chain = prompt | llm
            
            response_text = "" 
            try:
                response = chain.invoke({"resume": resume_text, "jd": job_description})
                response_text = response.content
                
                start_index = response_text.find('{')
                end_index = response_text.rfind('}') + 1

                if start_index != -1 and end_index != -1:
                    json_text = response_text[start_index:end_index]
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
                        st.write(", ".join(analysis_result.get('matched_skills', ["Not found"])))
                    with skill_col2:
                        st.warning("❗️ Missing Skills")
                        st.write(", ".join(analysis_result.get('missing_skills', ["None found"])))

                    st.subheader("💡 AI Recommendation")
                    st.info(analysis_result.get('recommendation_summary', 'No summary available.'))
                    
                    st.subheader("Resume Quality Checks")
                    word_count_status = get_word_count_status(resume_text)
                    repetition_status = get_repetition_status(resume_text)
                    action_verbs = "✅ Yes" if analysis_result.get('uses_action_verbs') else "⚠️ No"
                    quant_results = "✅ Yes" if analysis_result.get('has_quantifiable_results') else "⚠️ No"
                    
                    add_col1, add_col2, add_col3, add_col4 = st.columns(4)
                    add_col1.metric("Word Count", word_count_status.split()[0], " ".join(word_count_status.split()[1:]))
                    add_col2.metric("Repetition", repetition_status.split()[0], " ".join(repetition_status.split()[1:]))
                    add_col3.metric("Uses Action Verbs?", action_verbs)
                    add_col4.metric("Quantifiable Results?", quant_results)
                else:
                    st.error("AI did not return a valid JSON response. See raw response below.")
                    st.code(response_text, language="text")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.code(f"Raw AI response (if available):\n{response_text}", language="text")

