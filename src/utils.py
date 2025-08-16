import hashlib
from typing import List, Dict
from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document

def _make_id(name: str, content: bytes) -> str:
    h = hashlib.sha1()
    h.update(name.encode("utf-8"))
    h.update(content)
    return h.hexdigest()[:12]

def _read_pdf(content: bytes) -> str:
    reader = PdfReader(io_bytes:=content)
    # pypdf wants file-like; wrap bytes
    import io as _io
    f = _io.BytesIO(content)
    reader = PdfReader(f)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)

def _read_docx(content: bytes) -> str:
    import io as _io
    f = _io.BytesIO(content)
    doc = Document(f)
    return "\n".join(p.text for p in doc.paragraphs)

def _read_txt(content: bytes) -> str:
    # Best-effort decoding
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return content.decode(enc)
        except Exception:
            continue
    return content.decode("utf-8", errors="ignore")

def _read_html(content: bytes) -> str:
    html = _read_txt(content)
    soup = BeautifulSoup(html, "lxml")
    # remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())

def load_file_texts(uploaded_files) -> List[Dict[str, str]]:
    """
    Streamlit uploaded_files -> list of {id, name, text}
    """
    out = []
    for uf in uploaded_files:
        name = uf.name
        raw = uf.getvalue()
        ext = name.lower().split(".")[-1]
        if ext == "pdf":
            text = _read_pdf(raw)
        elif ext == "docx":
            text = _read_docx(raw)
        elif ext in ("html", "htm"):
            text = _read_html(raw)
        elif ext == "txt":
            text = _read_txt(raw)
        else:
            text = ""
        out.append({"id": _make_id(name, raw), "name": name, "text": text})
    # drop empties
    out = [d for d in out if d["text"].strip()]
    return out
