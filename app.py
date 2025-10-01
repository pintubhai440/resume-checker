# ======================================================================================
# 🚀 AI RESUME CHECKER - FINAL & EXCELLENT VERSION
# Author: Gemini (in collaboration with the user)
# Version: 2.0
# Description: An advanced AI-powered tool to analyze resumes against job descriptions,
#              featuring dual input methods (Text & GitHub) and robust validation.
# ======================================================================================

# --- 1. IMPORT REQUIRED LIBRARIES ---
import streamlit as st
import google.generativeai as genai
import requests
import fitz  # PyMuPDF
import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

# --- 2. DEFINE THE DATA STRUCTURE FOR AI RESPONSE (MAKES CODE STRONG) ---
# Pydantic ka use karke hum AI se aane wale response ke liye ek blueprint bana rahe hain.
# Isse agar AI galat format me data bhejta hai, toh hamara app crash nahi hoga.
class ResumeAnalysis(BaseModel):
    relevance_score: int = Field(description="Relevance of resume to the job description (0-100).")
    skills_match: str = Field(description="Percentage string of skills match.")
    years_experience: str = Field(description="Candidate's relevant years of experience.")
    education_level: str = Field(description="Alignment of education ('High', 'Medium', 'Low').")
    matched_skills: List[str] = Field(description="List of skills that match the job description.")
    missing_skills: List[str] = Field(description="List of critical skills missing from the resume.")
    uses_action_verbs: bool = Field(description="Boolean indicating if resume uses strong action verbs.")
    has_quantifiable_results: bool = Field(description="Boolean indicating if resume shows measurable results.")
    recommendation_summary: str = Field(description="A concise, 2-3 sentence expert summary.")
    recommendation_score: int = Field(description="Overall recommendation score (0-100).")

# --- 3. CORE LOGIC FUNCTIONS ---

def get_gemini_response(job_desc: str, resume_text: str) -> Optional[dict]:
    """
    Ye function AI model (Gemini 2.5 Pro) ko call karta hai.
    Ise ek bahut detailed 'brain' (prompt) diya gaya hai.
    """
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    # Ye setting AI ko zyada creative hone se rokti hai, taaki result consistent rahe.
    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        temperature=0.2
    )

    # AI ke liye naya, 'Excellent' prompt (uska 'Brain')
    prompt = f"""
    You are an exceptionally skilled Senior Technical Recruitment Manager with 15 years of experience.
    Your task is to conduct an in-depth, unbiased analysis of a candidate's resume against a given job description.
    Your analysis must be critical, fair, and based ONLY on the text provided.

    Follow these steps for your analysis:
    1.  **Skill Gap Analysis:** Identify which required skills are present and which are critically missing.
    2.  **Experience Evaluation:** Assess the years and relevance of the candidate's experience.
    3.  **Education Check:** Compare the candidate's education with the job requirements.
    4.  **Resume Quality Check:** Specifically look for the use of strong action verbs (e.g., "Managed", "Developed", "Led") and quantifiable, result-oriented achievements (e.g., "Increased sales by 20%").
    5.  **Final Synthesis:** Based on all the above points, provide a final recommendation score and a professional summary.

    You MUST return your entire analysis in a single, raw JSON object. Do not add any introductory text, markdown formatting, or explanations. The JSON must strictly follow this structure:
    {{
        "relevance_score": <integer>,
        "skills_match": "<percentage_string>",
        "years_experience": "<string>",
        "education_level": "<'High'|'Medium'|'Low'>",
        "matched_skills": [<string_list>],
        "missing_skills": [<string_list>],
        "uses_action_verbs": <boolean>,
        "has_quantifiable_results": <boolean>,
        "recommendation_summary": "<expert_summary_string>",
        "recommendation_score": <integer>
    }}

    Here is the data for your analysis:
    JOB DESCRIPTION: ```{job_desc}```
    RESUME TEXT: ```{resume_text}
