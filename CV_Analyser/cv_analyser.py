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
import re

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
        # Extract document ID from various Google Drive link formats
        doc_id = None
        # Format: https://docs.google.com/document/d/DOCUMENT_ID/edit
        doc_match = re.search(r"document/d/([a-zA-Z0-9_-]+)", link)
        if doc_match:
            doc_id = doc_match.group(1)
        else:
            # Format: https://drive.google.com/file/d/DOCUMENT_ID/view
            drive_match = re.search(r"file/d/([a-zA-Z0-9_-]+)", link)
            if drive_match:
                doc_id = drive_match.group(1)
        
        if not doc_id:
            st.error("Could not extract document ID from the provided link.")
            return None
            
        # For Google Docs, use the export feature to get docx format
        if "document/d/" in link:
            download_url = f"https://docs.google.com/document/d/{doc_id}/export?format=docx"
        else:
            # For other file types, use the direct download link
            download_url = f"https://drive.google.com/uc?export=download&id={doc_id}"
        
        # Add headers to mimic a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(download_url, headers=headers)
        
        if response.status_code == 200:
            # Check content type or try to determine file type
            content_type = response.headers.get('Content-Type', '')
            
            # Determine file extension based on content type or URL
            if 'application/pdf' in content_type:
                ext = '.pdf'
            elif 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
                ext = '.docx'
            elif "document/d/" in link:
                # Default to docx for Google Docs
                ext = '.docx'
            else:
                # Default extension based on binary content check
                if response.content.startswith(b'%PDF'):
                    ext = '.pdf'
                else:
                    ext = '.docx'
                
            # Save the file
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                f.write(response.content)
                temp_path = f.name
                
            # Validate the file depending on its type
            if ext == '.docx':
                try:
                    import zipfile
                    # Test if it's a valid zip file (DOCX files are zip archives)
                    with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                        # Just check if we can read the zip - we don't need to extract
                        pass
                except zipfile.BadZipFile:
                    st.error("Downloaded file is not a valid DOCX file.")
                    os.unlink(temp_path)  # Delete the invalid file
                    return None
            elif ext == '.pdf':
                # Basic PDF validation - check for PDF header
                if not response.content.startswith(b'%PDF'):
                    st.error("Downloaded file is not a valid PDF file.")
                    os.unlink(temp_path)  # Delete the invalid file
                    return None
                    
            return temp_path
        else:
            st.error(f"Failed to download file (Status code: {response.status_code}). Make sure the link is publicly accessible.")
            return None
    except Exception as e:
        st.error(f"Failed to access Google Drive link: {str(e)}")
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
            # Handle encoding issues
            line = line.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 10, line)

        path = os.path.join(tempfile.gettempdir(), file_name)
        pdf.output(path)
        return path
    except Exception as e:
        st.error(f"Failed to export PDF: {e}")
        return None


def main():
    st.title("📄 AI-Powered CV Analyzer")
    
    # Use session state to preserve data between reruns
    if 'resume_analyzed' not in st.session_state:
        st.session_state.resume_analyzed = False
    if 'resume_summary' not in st.session_state:
        st.session_state.resume_summary = None
    if 'formatted_sections' not in st.session_state:
        st.session_state.formatted_sections = None
    if 'texts' not in st.session_state:
        st.session_state.texts = []
    
    # Only show file input if resume hasn't been analyzed yet
    if not st.session_state.resume_analyzed:
        method = st.radio("Choose input method:", ["Upload File", "Google Drive Link"])
        
        file_path = None
        file_name = ""
        texts = []

        if method == "Upload File":
            uploaded_file = st.file_uploader("Upload your Resume", type=["docx", "pdf"], key="resume_upload")
            if uploaded_file:
                file_path = save_uploaded_file(uploaded_file)
                file_name = uploaded_file.name
                st.success("✅ File uploaded successfully.")

        elif method == "Google Drive Link":
            link = st.text_input("Paste Google Drive document link")
            if link and st.button("Process Google Drive Document"):
                with st.spinner("Downloading file from Google Drive..."):
                    file_path = download_file_from_gdrive(link)
                    if file_path:
                        file_name = os.path.basename(file_path)
                        st.success("✅ File accessed successfully.")

        if file_path:
            ext = os.path.splitext(file_path)[-1].lower()
            
            # Process the file based on extension
            try:
                if ext == ".docx":
                    texts = process_docx(file_path)
                elif ext == ".pdf":
                    texts = process_pdf(file_path)
                else:
                    st.error("❌ Unsupported file type")
                    return
                    
                # Check if texts were successfully extracted
                if not texts:
                    st.error("❌ No text could be extracted from the document. Please check file format and try again.")
                    return
                    
                st.subheader("📃 File Name")
                st.write(file_name)
                
                # Optional: Show a sample of extracted text to verify
                with st.expander("Show extracted text sample"):
                    st.write(texts[0].page_content[:200] + "...")

                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

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
                
                # Store the resume data in session state
                st.session_state.resume_summary = summary
                st.session_state.texts = texts
                
                def format_as_table(summary_text):
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
                        # Match lines like "Name:" or "**Name:**"
                        match = re.match(r"\*\*(.+?):\*\*\s*(.*)", line)
                        if not match:
                            match = re.match(r"(.+?):\s*(.*)", line)
                        
                        if match:
                            key = match.group(1).strip()
                            if key in sections:
                                current_section = key
                                sections[current_section] = match.group(2).strip() + "\n"
                                continue
                        
                        if current_section:
                            sections[current_section] += line + "\n"
                    return sections

                st.session_state.formatted_sections = format_as_table(summary)
                st.session_state.resume_analyzed = True
                
                # Force a rerun to refresh the UI with the analyzed resume
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("If using Google Drive, make sure the document sharing is set to 'Anyone with the link can view'")
    
    # Always show the resume analysis if it exists
    if st.session_state.resume_analyzed:
        st.subheader("📌 Resume Summary and Insights")
        
        tab1, tab2, tab3 = st.tabs(["📊 ATS & Experience", "🧠 Skills & Topics", "✅ Resume Suggestions"])

        with tab1:
            st.markdown("### Name")
            st.markdown(st.session_state.formatted_sections['Name'].strip())
            st.markdown("### Email")
            st.markdown(st.session_state.formatted_sections['Email'].strip())
            st.markdown("### Last Company")
            st.markdown(st.session_state.formatted_sections['Last Company'].strip())
            st.markdown("### Experience Summary")
            st.markdown(st.session_state.formatted_sections['Experience Summary'].strip())
            st.markdown("### ATS Score")
            st.markdown(st.session_state.formatted_sections['ATS Score'].strip())

        with tab2:
            st.markdown("### Key Skills")
            st.code(st.session_state.formatted_sections['Key Skills'].strip())
            st.markdown("### Interview Topics")
            st.markdown(st.session_state.formatted_sections['Interview Topics'].strip())

        with tab3:
            st.markdown("### Resume Improvement Suggestions")
            st.markdown(st.session_state.formatted_sections['Resume Improvement Suggestions'].strip())

        if st.button("📄 Export Summary to PDF"):
            pdf_path = export_to_pdf(st.session_state.resume_summary)
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name="CV_Summary.pdf",
                        mime="application/pdf"
                    )
        
        # Button to start over
        if st.button("Start Over (Upload a New Resume)"):
            st.session_state.resume_analyzed = False
            st.session_state.resume_summary = None
            st.session_state.formatted_sections = None
            st.session_state.texts = []
            st.rerun()

        # JD analysis section
        provide_jd = st.checkbox("Do you want to upload a Job Description (JD)?")
        if provide_jd:
            jd_method = st.radio("Choose JD input method:", ["Upload File", "Google Drive Link"], key="jd_method")
            
            jd_path = None
            
            if jd_method == "Upload File":
                jd_file = st.file_uploader("Upload JD (DOCX or PDF)", type=["docx", "pdf"], key="jd")
                if jd_file:
                    jd_path = save_uploaded_file(jd_file)
            else:
                jd_link = st.text_input("Paste Google Drive JD link")
                if jd_link and st.button("Process JD from Google Drive"):
                    with st.spinner("Downloading JD from Google Drive..."):
                        jd_path = download_file_from_gdrive(jd_link)
            
            if jd_path:
                try:
                    jd_texts = process_docx(jd_path) if jd_path.endswith(".docx") else process_pdf(jd_path)
                    if not jd_texts:
                        st.error("❌ No text could be extracted from the JD. Please check file format and try again.")
                        return
                        
                    jd_combined = "\n".join([doc.page_content for doc in jd_texts])
                    resume_combined = "\n".join([doc.page_content for doc in st.session_state.texts])
                    
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
                    
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
                    st.markdown(jd_response.content)
                except Exception as e:
                    st.error(f"Error processing JD: {str(e)}")
                    st.info("If using Google Drive, make sure the document sharing is set to 'Anyone with the link can view'")

if __name__ == "__main__":
    main()