"""
Reader Module - Deep Content Extraction
========================================

Extracts the first ~1000 characters of text from common file formats.
This enables the semantic engine to "taste" document content, not just filenames.

Supported formats:
- .txt, .md, .py, .json: Direct text read
- .pdf: First page extraction via pypdf
- .docx: First paragraph(s) via python-docx

Usage:
    from sensory.reader import read_header
    
    content = read_header("document.pdf", limit=1000)
    if content:
        # Analyze the content
        pass
"""

import os
from typing import Optional


def read_header(filepath: str, limit: int = 1000) -> Optional[str]:
    """
    Extract the first ~limit characters of text from a file.
    
    Args:
        filepath: Path to the file to read
        limit: Maximum characters to extract (default 1000)
        
    Returns:
        Extracted text string, or None if unreadable
        
    Note:
        Speed is priority. Does not read entire file.
        Returns None on any error (locked, corrupted, unsupported).
    """
    if not os.path.exists(filepath):
        return None
    
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        # Text-based formats: direct read
        if ext in ('.txt', '.md', '.py', '.json', '.csv', '.log', '.yaml', '.yml'):
            return _read_text_file(filepath, limit)
        
        # PDF: extract from first page
        elif ext == '.pdf':
            return _read_pdf(filepath, limit)
        
        # DOCX: extract from first paragraphs
        elif ext == '.docx':
            return _read_docx(filepath, limit)
        
        # Unsupported format
        else:
            return None
            
    except Exception:
        # Any error = return None (don't crash)
        return None


def _read_text_file(filepath: str, limit: int) -> Optional[str]:
    """Read first `limit` characters from a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(limit)
            return content.strip() if content else None
    except Exception:
        return None


def _read_pdf(filepath: str, limit: int) -> Optional[str]:
    """Extract text from the first page of a PDF."""
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(filepath)
        if len(reader.pages) == 0:
            return None
        
        # Extract text from first page only (speed priority)
        first_page = reader.pages[0]
        text = first_page.extract_text()
        
        if not text:
            return None
        
        # Limit to requested characters
        return text[:limit].strip() if text else None
        
    except ImportError:
        print("[Reader] Warning: pypdf not installed. Run: pip install pypdf")
        return None
    except Exception:
        return None


def _read_docx(filepath: str, limit: int) -> Optional[str]:
    """Extract text from the first paragraphs of a DOCX file."""
    try:
        from docx import Document
        
        doc = Document(filepath)
        
        # Collect text until we hit the limit
        collected = []
        total_chars = 0
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                collected.append(text)
                total_chars += len(text) + 1  # +1 for space/newline
                
                if total_chars >= limit:
                    break
        
        if not collected:
            return None
        
        result = ' '.join(collected)
        return result[:limit].strip()
        
    except ImportError:
        print("[Reader] Warning: python-docx not installed. Run: pip install python-docx")
        return None
    except Exception:
        return None


# Convenience function to check if a file type is supported
def is_supported(filepath: str) -> bool:
    """Check if the file type is supported for content extraction."""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in ('.txt', '.md', '.py', '.json', '.csv', '.log', 
                   '.yaml', '.yml', '.pdf', '.docx')


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        content = read_header(test_file)
        if content:
            print(f"[Reader] Extracted {len(content)} characters:")
            print("-" * 40)
            print(content[:500])
            print("-" * 40)
        else:
            print(f"[Reader] Could not read: {test_file}")
    else:
        print("Usage: python reader.py <filepath>")
