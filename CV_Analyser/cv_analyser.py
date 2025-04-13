# CV Analyser using LangChain and Streamlit
# This script allows users to upload their CVs in DOCX or PDF format, processes the documents,
# and provides insights such as ATS score, key skills, and suggestions for improvement.
# It also allows users to compare their CV with a job description to identify missing keywords.
# The script uses LangChain for document processing and OpenAI's GPT-3.5 for summarization and analysis.
import os
import tempfile
import textwrap
from dotenv import load_dotenv
from langchain.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
import requests
import markdown
from fpdf import FPDF

load_dotenv()

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            return temp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def download_file_from_gdrive(link):
    try:
        import re
        file_id_match = re.search(r"/d/([\w-]+)", link)
        if not file_id_match:
            st.error("Invalid Google Drive link format.")
            return None
        file_id = file_id_match.group(1)
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(download_url)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
                f.write(response.content)
                return f.name
        else:
            st.error("Failed to download file. Make sure the link is publicly accessible.")
            return None
    except Exception as e:
        st.error(f"Failed to access Google Drive link: {e}")
        return None

def process_docx(path_or_file):
    import shutil
    try:
        # If it's a file-like object (from Streamlit), save it to a temporary file
        if hasattr(path_or_file, "read"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                shutil.copyfileobj(path_or_file, tmp)
                path = tmp.name
        else:
            path = path_or_file

        loader = Docx2txtLoader(path)
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        return splitter.split_documents(documents)
    except Exception as e:
        st.error(f"DOCX Processing Error: {e}")
        return []

def process_pdf(path):
    try:
        loader = PyPDFLoader(path)
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        return splitter.split_documents(documents)
    except Exception as e:
        st.error(f"PDF Processing Error: {e}")
        return []

def export_to_pdf(text, file_name="cv_summary.pdf"):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for line in text.split("\n"):
            pdf.multi_cell(0, 10, line)

        path = os.path.join(tempfile.gettempdir(), file_name)
        pdf.output(path)
        return path
    except Exception as e:
        st.error(f"Failed to export PDF: {e}")
        return None

def main():
    st.title("📄 AI-Powered CV Analyzer")

    # method = st.radio("Choose input method:", ["Upload File", "Google Drive Link"])
    uploaded_file = st.file_uploader("Upload your Resume", type=["docx", "pdf"], key="resume_upload")

    file_path = None
    file_name = ""
    texts = []

    # if method == "Upload File":
    if uploaded_file:
        # uploaded_file = st.file_uploader("Upload your Resume", type=["docx", "pdf"])
        # if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)
        file_name = uploaded_file.name
        st.success("✅ File uploaded successfully.")

    # elif method == "Google Drive Link":
    #     link = st.text_input("Paste Google Drive DOCX link")
    #     if link:
    #         file_path = download_file_from_gdrive(link)
    #         file_name = link.split("/")[-2] + ".docx"
    #         if file_path:
    #             st.success("✅ File accessed successfully.")

    if file_path:
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".docx":
            texts = process_docx(file_path)
        elif ext == ".pdf":
            texts = process_pdf(file_path)
        else:
            st.error("❌ Unsupported file type")
            return

        st.subheader("📃 File Name")
        st.write(file_name)

        # st.subheader("🔍 Sample Resume Content (First 300 characters of each chunk)")
        # for doc in texts:
        #     st.markdown(f"```\n{doc.page_content[:300]}...\n```")

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

        prompt_template = """
        You have been given a Resume to analyse.
        Based on the provided resume text:
        - Extract: Name, Email, Key Skills, Last Company, Experience Summary.
        - Give an estimated ATS (Applicant Tracking System) score out of 100.
        - Recommend topics to study for interviews based on the profile.
        - Provide 3 personalized suggestions to improve the resume.
        
        Resume:
        {text}
        """

        refine_template = """
        Previous extracted details: {existing_answer}
        Refine them based on new context below:
        ------------
        {text}
        ------------
        Final Refined Output with following format:
        Name:
        Email:
        Key Skills:
        Last Company:
        Experience Summary:
        ATS Score:
        Interview Topics:
        Resume Improvement Suggestions:
        """

        prompt = PromptTemplate.from_template(prompt_template)
        refine_prompt = PromptTemplate.from_template(refine_template)

        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
        )

        with st.spinner("Analyzing resume..."):
            result = chain.invoke({"input_documents": texts})

        summary = result['output_text']

        st.subheader("📌 Resume Summary and Insights")

        def format_as_table(summary_text):
            import re
            sections = {
                "Name": "",
                "Email": "",
                "Key Skills": "",
                "Last Company": "",
                "Experience Summary": "",
                "ATS Score": "",
                "Interview Topics": "",
                "Resume Improvement Suggestions": ""
            }
            current_section = None
            for line in summary_text.split("\n"):
                line = line.strip()
                # Match lines like "**Name:** value"
                match = re.match(r"\*\*(.+?):\*\*\s*(.*)", line)
                if match:
                    key = match.group(1).strip()
                    if key in sections:
                        current_section = key
                        sections[current_section] = match.group(2).strip() + "\n"
                    continue
                elif current_section:
                    sections[current_section] += line + "\n"
            return sections

        formatted_sections = format_as_table(summary)

        tab1, tab2, tab3 = st.tabs(["📊 ATS & Experience", "🧠 Skills & Topics", "✅ Resume Suggestions"])

        with tab1:
            st.markdown("### Name")
            st.markdown(formatted_sections['Name'].strip())
            st.markdown("### Email")
            st.markdown(formatted_sections['Email'].strip())
            st.markdown("### Last Company")
            st.markdown(formatted_sections['Last Company'].strip())
            st.markdown("### Experience Summary")
            st.markdown(formatted_sections['Experience Summary'].strip())
            st.markdown("### ATS Score")
            st.markdown(formatted_sections['ATS Score'].strip())

        with tab2:
            st.markdown("### Key Skills")
            st.code(formatted_sections['Key Skills'].strip())
            st.markdown("### Interview Topics")
            st.markdown(formatted_sections['Interview Topics'].strip())

        with tab3:
            st.markdown("### Resume Improvement Suggestions")
            st.markdown(formatted_sections['Resume Improvement Suggestions'].strip())

        if st.button("📄 Export Summary to PDF"):
            def export_to_pdf(text, file_name="cv_summary.pdf"):
                try:
                    from fpdf import FPDF
                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)

                    for line in text.split("\n"):
                        line = line.encode("latin-1", "replace").decode("latin-1")  # safely encode unsupported chars
                        pdf.multi_cell(0, 10, line)

                    path = os.path.join(tempfile.gettempdir(), file_name)
                    pdf.output(path)
                    return path
                except Exception as e:
                    st.error(f"Failed to export PDF: {e}")
                    return None

            pdf_path = export_to_pdf(summary)
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name="CV_Summary.pdf",
                        mime="application/pdf"
                    )

        provide_jd = st.checkbox("Do you want to upload a Job Description (JD)?")
        if provide_jd:
            jd_file = st.file_uploader("Upload JD (DOCX or PDF)", type=["docx", "pdf"], key="jd")
            if jd_file:
                jd_path = save_uploaded_file(jd_file)
                jd_texts = process_docx(jd_path) if jd_path.endswith(".docx") else process_pdf(jd_path)
                jd_combined = "\n".join([doc.page_content for doc in jd_texts])
                resume_combined = "\n".join([doc.page_content for doc in texts])
                
                compare_prompt = PromptTemplate.from_template("""
                Given the following resume and job description:

                Resume:
                {resume}

                Job Description:
                {jd}

                Identify and list keywords from the JD that are missing in the resume but relevant to the candidate's profile.
                Format as bullet points.
                """)
                with st.spinner("Analyzing JD vs Resume..."):
                    compare_input = compare_prompt.format(resume=resume_combined, jd=jd_combined)
                    jd_response = llm.invoke(compare_input)

                st.subheader("📈 Suggested Keywords to Add from JD")
                st.markdown(jd_response)

if __name__ == "__main__":
    main()