# Zaroori libraries ko import karna

import streamlit as st

import os

import json

import re 

from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory

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

# App ka layout, title, aur icon set karna

st.set_page_config(layout="wide", page_title="AI Resume Checker", page_icon="🚀")

st.title("🚀 AI Resume Checker")

st.write("Analyze a resume against a job description to get instant, powerful insights.")



# --- API KEY & MODEL SETUP ---

# Streamlit secrets se API key ko surakshit tarike se lena

try:

    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

except (FileNotFoundError, KeyError):

    st.error("🤫 Google API Key not found. Please add it to your Streamlit secrets.")

    st.stop()



# --- LAYOUT ---

# Do columns banana - ek job description ke liye, ek resume ke liye

col1, col2 = st.columns(2, gap="large")

with col1:

    st.header("📄 Job Requirements")

    job_description = st.text_area("Job Description", height=350, label_visibility="collapsed", placeholder="Paste the job description here...", key="job_desc")

with col2:

    st.header("👤 Resume Content")

    resume_text = st.text_area("Paste Resume Text", height=350, label_visibility="collapsed", placeholder="Paste the candidate's resume here...", key="resume_text")



# --- ANALYSIS BUTTON & LOGIC ---

if st.button("Analyze with Gemini AI", use_container_width=True, type="primary"):

    if not resume_text or not job_description:

        st.warning("Please provide both the Job Description and the Resume text.")

    else:

        with st.spinner('Gemini is performing a deep analysis... This might take a moment.'):

            # Google Gemini AI model ko set karna

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

            

            # AI ko batana ki use kya karna hai (ek detailed prompt)

            prompt_template_str = """

            You are an expert AI hiring assistant. Your task is to analyze a resume against a job description.

            Provide ONLY a raw JSON response with the following keys. Do not add any extra text or formatting before or after the JSON object.

            - "relevance_score": An integer (0-100).

            - "skills_match": A percentage string (e.g., "85%").

            - "years_experience": A string for the candidate's relevant years of experience.

            - "education_level": A brief description of educational alignment ("High", "Medium", "Low").

            - "matched_skills": A list of up to 7 matching skills.

            - "missing_skills": A list of up to 3 critical missing skills.

            - "recommendation_summary": A concise, 2-sentence summary.

            - "uses_action_verbs": A boolean.

            - "has_quantifiable_results": A boolean.

            - "recommendation_score": An integer (0-100) for the overall confidence in recommending this candidate.



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

                else:

                    st.error("AI did not return a valid JSON response. See raw response below.")

                    st.code(response_text, language="text")

            except Exception as e:

                st.error(f"An unexpected error occurred: {e}")

                st.code(f"Raw AI response (if available):\n{response_text}", language="text")
