https://resume-checker-z6wuznuikwk3wvsgwds6tz.streamlit.app/

This is a Streamlit web application that uses the Google Gemini Pro model to analyze a candidate's resume against a job description. It provides a comprehensive analysis, including a relevance score, skill gap analysis, and qualitative feedback on the resume's quality.

This tool is designed for recruiters, hiring managers, and job applicants who want to get an instant, AI-powered evaluation of a resume's fit for a specific role.

# **🚀 AI Resume Checker**

This is a Streamlit web application that uses the Google Gemini Pro model to analyze a candidate's resume against a job description. It provides a comprehensive analysis, including a relevance score, skill gap analysis, and qualitative feedback on the resume's quality.

This tool is designed for recruiters, hiring managers, and job applicants who want to get an instant, AI-powered evaluation of a resume's fit for a specific role.

## **✨ Features**

* **Relevance Score**: Get an AI-generated score (0-100%) on how well the resume matches the job requirements.  
* **Skill Matching**: See a percentage match for required skills.  
* **Experience Calculation**: The AI calculates the candidate's years of relevant experience.  
* **Skill Gap Analysis**: Instantly identify which key skills are present and which are missing from the resume.  
* **AI Recommendation**: Read a concise, two-sentence summary of the candidate's suitability.  
* **Resume Quality Checks**: The app checks for optimal word count, keyword repetition, use of action verbs, and quantifiable results.  
* **Downloadable Report**: Download a .txt file of the complete analysis.

## **🛠️ Tech Stack**

* **Language**: Python  
* **Framework**: Streamlit  
* **AI Model**: Google Gemini 1.5 Flash (gemini-1.5-flash-latest)  
* **Core Libraries**: langchain, langchain\_google\_genai

## **⚙️ Setup and Installation**

Follow these steps to run the application locally.

### **1\. Clone the Repository**

git clone \<your-repository-url\>  
cd \<repository-folder-name\>

### **2\. Create a Virtual Environment**

It's highly recommended to use a virtual environment to manage dependencies.

\# For Windows  
python \-m venv .venv  
.\\.venv\\Scripts\\activate

\# For macOS/Linux  
python3 \-m venv .venv  
source .venv/bin/activate

### **3\. Install Dependencies**

Install all the required Python packages from the requirements.txt file.

pip install \-r requirements.txt

### **4\. Set Up Your API Key**

This application requires a Google AI API key. For security, we use Streamlit's built-in secrets management.

1. Create a folder named .streamlit in your project's root directory.  
2. Inside this folder, create a file named secrets.toml.  
3. Add your API key to this file in the following format:  
   \# .streamlit/secrets.toml

   GOOGLE\_API\_KEY \= "AIzaSy...your...actual...api...key...here"

**Important:** The .gitignore file is configured to ignore the secrets.toml file, so your key will not be accidentally committed to GitHub.

## **▶️ How to Run the App**

Once you have completed the setup, run the following command in your terminal:

python -m streamlit run app.py

The application will open in your default web browser.

## **📝 How to Use**

1. Paste the complete job description into the "Job Requirements" text area.  
2. Paste the candidate's full resume text into the "Resume Content" text area.  
3. Click the "Analyze with Gemini AI" button.  
4. View the detailed analysis on the screen.  
5. Optionally, click the "Download Full Report" button to save the results.
