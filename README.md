# 📄 AI-Powered CV Analyzer

A Streamlit application that leverages LLMs (e.g., Gemini Pro via LangChain) to analyze resumes (in `.docx` or `.pdf` format) and generate actionable summaries, ATS scores, interview topics, and personalized resume improvement suggestions.

---

## 🚀 Features

- ✅ Upload `.docx` or `.pdf` resume files
- 🧠 Uses Google Gemini Pro (via LangChain) to summarize key information:
  - Name, Email, Skills, Experience, Last Company
- 📈 Calculates an estimated ATS (Applicant Tracking System) score
- 💡 Recommends personalized interview topics based on resume content
- 🛠 Suggests detailed improvements to enhance resume impact
- 📊 Displays summary in clean tabbed layout (ATS, Skills, Suggestions)
- 📥 Export final summary as a downloadable PDF
- 🔍 Optional JD upload to extract missing keywords from job descriptions

---

## 🧑‍💻 Tech Stack

- **Frontend/UI:** [Streamlit](https://streamlit.io/)
- **LLM Integration:** [LangChain](https://www.langchain.com/) + [Google Gemini Pro](https://ai.google.dev/gemini-api)
- **Document Parsing:** 
  - `Docx2txtLoader` for `.docx`
  - `PyPDFLoader` for `.pdf`
- **Prompt Engineering:** LangChain's `PromptTemplate` + `Refine` summarization chain
- **Text Chunking:** `CharacterTextSplitter`
- **PDF Export:** Python `fpdf` module

---

## 🛠 Setup Instructions

### 1. Clone the repository
```
git clone https://github.com/your-username/cv-analyzer.git
cd cv-analyzer
```


### 2. Create virtual environment & install dependencies
```
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

### 3. Set up .env for API keys
Create a .env file with:
GOOGLE_API_KEY=your_google_generative_ai_key

### 4. Run the Streamlit app
```
streamlit run cv_analyser.py
```

## Sample Use Case
1. Upload your latest resume as a .docx or .pdf.

2. View:

- Structured summary of your profile

- ATS score and how to improve it

- Suggested topics for interviews

- Resume formatting and content suggestions

- Optionally upload a JD (Job Description) to get personalized keywords to include.

- Export final summary as a downloadable PDF.
