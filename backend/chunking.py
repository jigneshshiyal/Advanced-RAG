import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

def get_text_splitter(strategy="recursive", chunk_size=800, chunk_overlap=150, model_name="gpt-4o"):
    if strategy == "token":
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, model_name=model_name)
    # default: character recursive (works well)
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

splitter = get_text_splitter(strategy="recursive", chunk_size=800, chunk_overlap=200)


def generate_chunk_id(text: str) -> str:
    """
    Generate a stable unique ID for a text chunk using file name, page, and text hash.
    """
    # short hash of text to ensure uniqueness even if chunk sizes vary
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    # return f"{source}__page{page_number}__chunk{chunk_index}__{text_hash}"
    return text_hash


def chunking_doc(documents, splitter=splitter):
    chunking_dataset = []
    for document in documents:
        texts = splitter.split_text(document["text"])
        for text in texts:
            chunking_dataset.append({
                "text": text,
                "metadata": document["metadata"],
                "id": generate_chunk_id(
                    text=text
                )
            })
    return chunking_dataset