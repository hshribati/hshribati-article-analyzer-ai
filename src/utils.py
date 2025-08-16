# src/utils.py
from io import BytesIO
from bs4 import BeautifulSoup
import docx
import PyPDF2

def extract_text_from_file(uploaded_file):
    """
    Takes a Streamlit uploaded file and returns extracted text.
    Supports: PDF, DOCX, TXT, HTML
    """
    filename = uploaded_file.name.lower()
    bytes_data = uploaded_file.read()
    
    if filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(BytesIO(bytes_data))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

    elif filename.endswith(".docx"):
        doc = docx.Document(BytesIO(bytes_data))
        text = "\n".join([p.text for p in doc.paragraphs])
        return text

    elif filename.endswith(".txt"):
        return bytes_data.decode("utf-8", errors="ignore")

    elif filename.endswith((".html", ".htm")):
        soup = BeautifulSoup(bytes_data, "html.parser")
        return soup.get_text(separator="\n")

    else:
        return ""
