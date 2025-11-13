import pymupdf 
import re
from bs4 import BeautifulSoup
import unicodedata
from collections import Counter


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    
    # Only use BeautifulSoup if text looks like HTML
    if bool(re.search(r"<[^>]+>", text)):
        text = BeautifulSoup(text, "html.parser").get_text()
    
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Optional: remove non-ASCII
    return text


def extract_text(pdf_path, remove_headers_footers=True):
    doc = pymupdf.open(pdf_path)
    pages_text = []
    header_footer_candidates = []

    for page in doc:
        page_text = page.get_text("text") or ""
        lines = [clean_text(line.strip()) for line in page_text.split("\n") if len(line.strip()) > 10]
        if remove_headers_footers:
            header_footer_candidates.extend(lines[:2] + lines[-2:])
        pages_text.append((page.number + 1, lines))

    repetitive = set()
    if remove_headers_footers:
        counts = Counter(header_footer_candidates)
        repetitive = {line for line, c in counts.items() if c > 0.6 * len(pages_text)}

    text_docs = []
    for page_num, lines in pages_text:
        clean_lines = [ln for ln in lines if ln not in repetitive]
        text_docs.append({
            "metadata": {"page_number": page_num, "source": pdf_path},
            "text": " ".join(clean_lines)
        })

    doc.close()
    return text_docs

def pdfs_loader(paths):
    all_docs = []
    for pdf_path in paths:
        docs = extract_text(pdf_path)
        all_docs.extend(docs)
    return all_docs