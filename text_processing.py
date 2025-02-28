import re
from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def clean_chunk(chunk: str) -> str:
    """
    Clean a text chunk by removing copyright notices, page numbers, and other boilerplate text.
    """
    if not isinstance(chunk, str):
        return chunk
        
    # Remove copyright notice and related text
    copyright_patterns = [
        r"Copyright \d{4} © - .+ All Rights Reserved\.",
        r"No part of this document may be reproduced or utilized.+?author\.",
        r"---\s*Copyright.+?author\.\s*---",  # Matches the entire copyright block
        r"\[\d+\]:",  # Remove page/cell numbers in brackets
        r"^\d+$",     # Standalone page numbers
        r"\s*\.\.\.\s*"  # Remove ellipsis
    ]
    
    cleaned_text = chunk
    for pattern in copyright_patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.DOTALL | re.MULTILINE)
    
    # Remove empty lines and normalize whitespace
    cleaned_text = "\n".join(line.strip() for line in cleaned_text.splitlines() if line.strip())
    
    return cleaned_text

def strip_str(s: str) -> str:
    """
    Helper function for helping format strings returned by the LLM.
    Trims whitespace and finds the actual text content.
    """
    l, r = 0, len(s)-1
    beg_found = False
    for i in range(len(s)):
        if s[i].isalpha():
            if not beg_found:
                l = i
                beg_found = True
            else:
                r = i 
    r += 2
    return s[l:min(r, len(s))]

def chunk_markdown_document(markdown_doc: str) -> List[str]:
    """
    Split the markdown document into meaningful chunks.
    Uses a two-step process:
    1. Split by markdown headers
    2. Further split by size for large sections
    """
    # Define headers to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2")    
    ]

    # First split by markdown headers
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    
    markdown_doc_splits = markdown_splitter.split_text(markdown_doc)
    
    # Further split by size
    chunk_size = 1024
    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
    )
    
    chunked_document = text_splitter.split_documents(markdown_doc_splits)
    
    # Filter out empty or very short chunks
    chunks = [chunk.page_content for chunk in chunked_document 
              if len(re.sub(r'[^a-zA-Z0-9\s]', '', chunk.page_content)) > 100]
    
    return chunks