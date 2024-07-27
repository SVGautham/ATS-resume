import os
import streamlit as st
import nltk
import language_tool_python
from langchain_openai import OpenAI
from dotenv import load_dotenv
from docx import Document
import fitz  # PyMuPDF

# Load environment variables
load_dotenv()

# Initialize OpenAI and LanguageTool
llm = OpenAI(temperature=0.7, max_tokens=1500)
tool = language_tool_python.LanguageToolPublicAPI('en-US')

# Initialize NLTK
nltk.download('punkt')

# Define job keywords for scoring
job_keywords = {
    "analyst": ["analysis", "data", "reporting", "Excel", "SQL", "visualization"],
    "data scientist": ["machine learning", "statistics", "Python", "data analysis", "AI", "big data"],
    "machine learning engineer": ["machine learning", "TensorFlow", "Python", "algorithms", "neural networks", "deep learning"],
    "prompt engineer": ["OpenAI", "natural language processing", "GPT", "prompts", "AI", "language models"],
    "backend developer": ["API", "Node.js", "Python", "databases", "server-side", "REST"],
    "frontend developer": ["React", "CSS", "JavaScript", "HTML", "UI/UX", "responsive design"],
    "fullstack developer": ["React", "Node.js", "API", "full stack", "frontend", "backend"],
    "devops engineer": ["CI/CD", "AWS", "Docker", "Kubernetes", "automation", "infrastructure as code"],
    "cloud engineer": ["cloud computing", "AWS", "Azure", "infrastructure", "scalability", "cloud security"],
    "software engineer": ["software development", "C++", "Java", "Python", "algorithms", "data structures"],
    "quality assurance": ["testing", "QA", "automation", "Selenium", "test cases", "bug tracking"],
}

def calculate_ats_score(text, job_role, level):
    role_keywords = job_keywords.get(job_role, [])
    text_lower = text.lower()
    
    # Keyword relevance and frequency
    keyword_score = sum(text_lower.count(kw.lower()) for kw in role_keywords) / len(role_keywords)
    
    # Skills matching (using role-specific keywords)
    skills_score = sum(text_lower.count(kw.lower()) for kw in role_keywords) / len(role_keywords)
    
    # Education relevance (simplified)
    education_score = 1 if "degree" in text_lower or "bachelor" in text_lower else 0.5
    
    # Experience relevance (simplified)
    experience_score = 0.5 + (0.1 * text_lower.count("experience"))
    
    # Overall score
    raw_score = (keyword_score * 0.4 + skills_score * 0.3 + education_score * 0.15 + experience_score * 0.15) * 100
    
    # Apply level weight
    level_weights = {"entry": 0.8, "mid": 1.0, "experienced": 1.2}
    ats_score = max(1, min(raw_score * level_weights[level], 99))
    
    return ats_score

def analyze_keywords(text, job_role):
    role_keywords = job_keywords.get(job_role, [])
    text_lower = text.lower()
    keyword_analysis = []
    for kw in role_keywords:
        count = text_lower.count(kw.lower())
        if count > 0:
            keyword_analysis.append(f"'{kw}' found {count} time(s)")
        else:
            keyword_analysis.append(f"'{kw}' not found - consider adding")
    return keyword_analysis

def analyze_context(text, job_role):
    role_keywords = job_keywords.get(job_role, [])
    sentences = nltk.sent_tokenize(text)
    context_analysis = []
    for kw in role_keywords:
        for sentence in sentences:
            if kw.lower() in sentence.lower():
                context_analysis.append(f"Context for '{kw}': {sentence}")
                break
    return context_analysis

def check_grammar(text):
    sentences = nltk.sent_tokenize(text)
    grammar_issues = []
    for i, sentence in enumerate(sentences):
        mistakes = tool.check(sentence)
        if mistakes:
            grammar_issues.append({
                'sentence_number': i + 1,
                'sentence': sentence,
                'corrections': [f"{m.context}  ->  {m.replacements[0] if m.replacements else 'No suggestion'}" for m in mistakes]
            })
    return grammar_issues

def analyze_resume(text, job_role, level):
    ats_score = calculate_ats_score(text, job_role, level)
    keyword_analysis = analyze_keywords(text, job_role)
    context_analysis = analyze_context(text, job_role)
    grammar_issues = check_grammar(text)
    
    # Generate detailed analysis using OpenAI
    prompt_template = """
    This is a resume analysis for the role of {job_role} at {level} level. The ATS score is {ats_score:.2f}.
    Identify improvements and suggest missing keywords. Mention grammar issues if any.
    """
    prompt = prompt_template.format(job_role=job_role, level=level, ats_score=ats_score)
    try:
        response = llm.invoke(input=prompt)
        detailed_analysis = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        detailed_analysis = f"An error occurred: {str(e)}"

    return ats_score, keyword_analysis, context_analysis, grammar_issues, detailed_analysis

# Streamlit UI
st.title("Resume Analyzer")
st.write("Upload your resume, choose the job role and level, and get insights on your resume.")

uploaded_file = st.file_uploader("Upload Resume (.docx or .pdf format)", type=["docx", "pdf"])
job_role = st.selectbox("Select Job Role", list(job_keywords.keys()))
level = st.selectbox("Select Experience Level", ["entry", "mid", "experienced"])

if st.button("Analyze Resume"):
    if uploaded_file:
        if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif uploaded_file.type == "application/pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
        
        ats_score, keyword_analysis, context_analysis, grammar_issues, detailed_analysis = analyze_resume(text, job_role, level)

        st.header("Analysis Results")
        st.subheader("ATS Score")
        st.write(f"Your ATS score is: {ats_score:.2f}%")

        st.subheader("Keyword Analysis")
        for analysis in keyword_analysis:
            st.write(analysis)

        st.subheader("Keyword Context")
        for context in context_analysis:
            st.write(context)

        st.subheader("Grammar Issues")
        if grammar_issues:
            st.write(f"Found {len(grammar_issues)} sentences with potential grammar issues. Consider reviewing your resume:")
            for issue in grammar_issues:
                st.markdown(f"- Sentence {issue['sentence_number']}: {issue['sentence']}")
                st.markdown("  Suggested corrections:")
                for correction in issue['corrections']:
                    st.markdown(f"    - {correction}")
        else:
            st.write("No grammar issues detected.")

        st.subheader("Detailed Analysis")
        st.write(detailed_analysis)
    else:
        st.write("Please upload a resume file to analyze.")
