import PyPDF2
from docx import Document
from bs4 import BeautifulSoup

def extract_text_from_file(file):
    """
    Extracts text from PDF, DOCX, TXT, or HTML files.
    """
    try:
        if file.type == "application/pdf":
            return extract_text_from_pdf(file)
        elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                           "application/msword"]:
            return extract_text_from_docx(file)
        elif file.type == "text/plain":
            return file.getvalue().decode("utf-8")
        elif file.type == "text/html":
            return extract_text_from_html(file)
        else:
            return ""
    except Exception as e:
        return f"[Error extracting text: {e}]"

# ----------------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ----------------------------
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([p.text for p in doc.paragraphs])
    return text

# ----------------------------
def extract_text_from_html(file):
    soup = BeautifulSoup(file.getvalue(), "html.parser")
    return soup.get_text(separator="\n")
