import os
import streamlit as st
import openai
import nltk
import language_tool_python
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from io import BytesIO
from docx import Document

# Load environment variables
load_dotenv()

# Initialize OpenAI and LanguageTool
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0.7, max_tokens=1500)
tool = language_tool_python.LanguageTool('en-US')

# Initialize NLTK
nltk.download('punkt')

# Function to extract text from a .docx file
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to analyze the resume using OpenAI and LangChain
def analyze_resume(text, job_role, level):
    # Define job keywords for scoring (mockup example)
    job_keywords = {
        "analyst": ["analysis", "data", "reporting", "Excel"],
        "data scientist": ["machine learning", "statistics", "Python", "data analysis"],
        "machine learning engineer": ["machine learning", "TensorFlow", "Python", "algorithms"],
        "prompt engineer": ["OpenAI", "natural language processing", "GPT", "prompts"],
        "backend developer": ["API", "Node.js", "Python", "databases"],
        "frontend developer": ["React", "CSS", "JavaScript", "HTML"],
        "fullstack developer": ["React", "Node.js", "API", "full stack"],
        "devops engineer": ["CI/CD", "AWS", "Docker", "Kubernetes"],
        "cloud engineer": ["cloud computing", "AWS", "Azure", "infrastructure"],
        "software engineer": ["software development", "C++", "Java", "Python"],
        "quality assurance": ["testing", "QA", "automation", "Selenium"],
    }

    level_weights = {
        "entry": 0.5,
        "mid": 1.0,
        "experienced": 1.5,
    }

    # Calculate ATS Score
    role_keywords = job_keywords.get(job_role, [])
    text_tokens = nltk.word_tokenize(text.lower())
    keyword_matches = sum(1 for word in text_tokens if word in role_keywords)
    max_matches = len(role_keywords)
    ats_score = (keyword_matches / max_matches) * 100 * level_weights[level]

    # Identify grammar mistakes
    grammar_mistakes = tool.check(text)
    grammar_issues = len(grammar_mistakes)

    # Suggest missing keywords
    missing_keywords = [kw for kw in role_keywords if kw not in text_tokens]

    # Generate detailed analysis using OpenAI
    prompt = f"""
    This is a resume analysis for the role of {job_role} at {level} level. The ATS score is {ats_score:.2f}.
    Identify improvements and suggest missing keywords. Mention grammar issues if any.
    """
    completion = llm(prompt_template=PromptTemplate.from_template(prompt))

    return ats_score, missing_keywords, grammar_issues, completion

# Streamlit UI
st.title("Resume Analyzer")
st.write("Upload your resume, choose the job role and level, and get insights on your resume.")

uploaded_file = st.file_uploader("Upload Resume (.docx format)", type="docx")
job_role = st.selectbox("Select Job Role", ["analyst", "data scientist", "machine learning engineer", "prompt engineer", "backend developer", "frontend developer", "fullstack developer", "devops engineer", "cloud engineer", "software engineer", "quality assurance"])
level = st.selectbox("Select Experience Level", ["entry", "mid", "experienced"])

if uploaded_file:
    text = extract_text_from_docx(uploaded_file)
    ats_score, missing_keywords, grammar_issues, detailed_analysis = analyze_resume(text, job_role, level)

    st.header("Analysis Results")
    st.subheader("ATS Score")
    st.write(f"Your ATS score is: {ats_score:.2f}%")

    st.subheader("Missing Keywords")
    if missing_keywords:
        st.write("Consider adding these keywords to improve your ATS score:")
        st.write(", ".join(missing_keywords))
    else:
        st.write("Great! Your resume contains all the essential keywords.")

    st.subheader("Grammar Issues")
    if grammar_issues > 0:
        st.write(f"Found {grammar_issues} potential grammar issues. Consider reviewing your resume.")
    else:
        st.write("No grammar issues detected.")

    st.subheader("Detailed Analysis")
    st.write(detailed_analysis)

