import streamlit as st
import os
import json
import re 
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from collections import Counter
import PyPDF2
import docx

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

def extract_text_from_file(uploaded_file):
    """Extracts text from uploaded PDF or DOCX file."""
    text = ""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

# --- UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Resume Checker")
st.title("🚀 AI Resume Checker & Candidate Tracker")
st.write("Analyze resumes against job requirements, and track all candidates in one place.")

# --- INITIALIZE SESSION STATE ---
if 'candidates' not in st.session_state:
    st.session_state.candidates = []

# --- API KEY & MODEL SETUP ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("🤫 Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()
    
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- LAYOUT ---
st.header("📄 Job Requirements")
job_description = st.text_area("Paste the Job Description here", height=200, label_visibility="collapsed")

st.divider()

st.header("👤 Add a New Candidate")
candidate_name = st.text_input("Enter Candidate's Name")
uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

# --- ANALYSIS BUTTON & LOGIC ---
if st.button("Analyze & Add Candidate", use_container_width=True, type="primary"):
    
    if not uploaded_file or not job_description or not candidate_name:
        st.warning("Please provide the Job Description, Candidate's Name, and upload a Resume file.")
    else:
        with st.spinner(f'Reading and analyzing {candidate_name}\'s resume...'):
            resume_text = extract_text_from_file(uploaded_file)
            
            if not resume_text:
                st.error("Could not extract text from the uploaded file. Please ensure the file is not empty or corrupted.")
            else:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", # <<< YAHAN BADLAV KIYA GAYA HAI
                    temperature=0.3,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    },
                )
                
                prompt_template_str = """
                You are an expert AI hiring assistant. Analyze the following resume and job description.
                IMPORTANT CONTEXT: The current date is September 21, 2025.
                Provide ONLY a raw JSON response with the following keys. Do not include any other text or formatting.
                - "relevance_score": An integer from 0 to 100.
                - "skills_match": A percentage string like "85%".
                - "years_experience": A string for the candidate's relevant experience.
                - "education_level": A brief alignment description like "High", "Medium", or "Low".
                - "matched_skills": A list of up to 7 matching skills.
                - "missing_skills": A list of up to 3 critical missing skills.
                - "recommendation_summary": A 2-sentence summary.
                - "uses_action_verbs": A boolean.
                - "has_quantifiable_results": A boolean.
                - "recommendation_score": An integer from 0 to 100 for overall recommendation confidence.

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
                        
                        st.session_state.candidates.append({
                            "name": candidate_name,
                            "analysis": analysis_result,
                            "resume": resume_text 
                        })
                        st.success(f"Successfully analyzed and added {candidate_name} to the list below.")
                    else:
                        raise ValueError("No valid JSON found in the AI's response.")

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

st.divider()

# --- DISPLAY SAVED CANDIDATES ---
st.header("✅ Analyzed Candidates")

if st.button("Clear Candidate History"):
    st.session_state.candidates = []
    st.rerun()

if not st.session_state.candidates:
    st.info("No candidates have been analyzed yet. Add a candidate above to see their results here.")
else:
    sorted_candidates = sorted(st.session_state.candidates, key=lambda x: x['analysis'].get('recommendation_score', 0), reverse=True)

    for candidate in sorted_candidates:
        analysis = candidate["analysis"]
        name = candidate["name"]
        
        recommendation_score = analysis.get('recommendation_score', 0)
        if recommendation_score >= 75:
            rec_color = "green"
            rec_text = "Highly Recommended"
        elif recommendation_score >= 50:
            rec_color = "orange"
            rec_text = "Worth Considering"
        else:
            rec_color = "red"
            rec_text = "Not a Strong Fit"
        
        with st.expander(f"**{name}** - Verdict: **:{rec_color}[{rec_text}]** (Score: {recommendation_score}%)"):
            
            word_count_status = get_word_count_status(candidate["resume"])
            repetition_status = get_repetition_status(candidate["resume"])

            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("AI Relevance Score", f"{analysis.get('relevance_score', 0)}%")
            res_col2.metric("Skills Match", analysis.get('skills_match', 'N/A'))
            res_col3.metric("Years' Experience", analysis.get('years_experience', 'N/A'))
            res_col4.metric("Education Level", analysis.get('education_level', 'N/A'))

            st.subheader("Skills Analysis")
            skill_col1, skill_col2 = st.columns(2)
            with skill_col1:
                st.success("✅ Matched Skills")
                st.write(", ".join(analysis.get('matched_skills', ["Not found"])))
            with skill_col2:
                st.warning("❗️ Missing Skills")
                st.write(", ".join(analysis.get('missing_skills', ["None found"])))

            st.subheader("💡 Recommendation")
            st.info(analysis.get('recommendation_summary', 'No summary available.'))
            
            st.subheader("Resume Quality Checks")
            action_verbs = "✅ Yes" if analysis.get('uses_action_verbs') else "⚠️ No"
            quant_results = "✅ Yes" if analysis.get('has_quantifiable_results') else "⚠️ No"
            
            add_col1, add_col2, add_col3, add_col4 = st.columns(4)
            with add_col1:
                st.metric("Word Count", word_count_status) 
            with add_col2:
                st.metric("Keyword Repetition", repetition_status)
            with add_col3:
                st.metric("Uses Action Verbs?", action_verbs)
            with add_col4:
                st.metric("Quantifiable Results?", quant_results)

