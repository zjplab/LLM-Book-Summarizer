import re
from typing import List
from llama_index.core import Document
from llama_index.readers.file import PDFReader

def process_pdf_with_chapters(file_path: str) -> List[Document]:
    """
    Process a PDF file and organize it into chapter-based documents using LlamaIndex's PDF parser.
    
    Args:
        file_path (str): Path to the PDF file
    
    Returns:
        List[Document]: List of documents, each representing a chapter or section
    """
    # Initialize PDF reader
    pdf_reader = PDFReader()
    
    # Load the PDF documents
    documents = pdf_reader.load_data(file=file_path)
    
    # Combine all pages into a single text for processing
    full_text = "\n\n".join([doc.text for doc in documents])
    
    # Split into chapters using various heading patterns
    chapters = split_into_chapters(full_text)
    
    # Create Document objects for each chapter
    chapter_documents = []
    for i, (title, content) in enumerate(chapters):
        # Clean up the content
        cleaned_content = clean_chapter_content(content)
        
        # Create document with metadata
        doc = Document(
            text=cleaned_content,
            metadata={
                "chapter_number": i + 1,
                "chapter_title": title,
                "source": file_path,
                "document_type": "chapter"
            }
        )
        chapter_documents.append(doc)
    
    return chapter_documents

def split_into_chapters(text: str) -> List[tuple]:
    """
    Split text into chapters based on common heading patterns.
    
    Args:
        text (str): Full text of the document
    
    Returns:
        List[tuple]: List of (title, content) tuples
    """
    chapters = []
    
    # Common chapter heading patterns
    chapter_patterns = [
        r'^(Chapter\s+\d+[:\.\s].*?)$',           # Chapter 1: Title
        r'^(CHAPTER\s+\d+[:\.\s].*?)$',           # CHAPTER 1: TITLE
        r'^(\d+\.\s+.*?)$',                       # 1. Title
        r'^([A-Z][A-Z\s]{10,})$',                 # ALL CAPS TITLES (at least 10 chars)
        r'^(Part\s+[IVX]+[:\.\s].*?)$',          # Part I: Title
        r'^(Section\s+\d+[:\.\s].*?)$',          # Section 1: Title
    ]
    
    # Combine all patterns
    combined_pattern = '|'.join(f'({pattern})' for pattern in chapter_patterns)
    
    # Split text by lines and find chapter boundaries
    lines = text.split('\n')
    current_chapter_title = "Introduction"
    current_chapter_content = []
    
    for line in lines:
        line = line.strip()
        
        # Check if this line matches any chapter pattern
        if line and re.match(combined_pattern, line, re.MULTILINE | re.IGNORECASE):
            # Save previous chapter if it has content
            if current_chapter_content:
                chapters.append((current_chapter_title, '\n'.join(current_chapter_content)))
            
            # Start new chapter
            current_chapter_title = clean_chapter_title(line)
            current_chapter_content = []
        else:
            # Add line to current chapter
            if line:  # Skip empty lines
                current_chapter_content.append(line)
    
    # Add the last chapter
    if current_chapter_content:
        chapters.append((current_chapter_title, '\n'.join(current_chapter_content)))
    
    # If no chapters were found, treat the entire document as one chapter
    if not chapters:
        chapters = [("Complete Document", text)]
    
    # Filter out very short chapters (likely false positives)
    filtered_chapters = []
    for title, content in chapters:
        if len(content.strip()) > 200:  # At least 200 characters
            filtered_chapters.append((title, content))
    
    return filtered_chapters or [("Complete Document", text)]

def clean_chapter_title(title: str) -> str:
    """
    Clean and normalize chapter titles.
    
    Args:
        title (str): Raw chapter title
    
    Returns:
        str: Cleaned chapter title
    """
    # Remove extra whitespace
    title = re.sub(r'\s+', ' ', title.strip())
    
    # Remove common prefixes and clean up
    title = re.sub(r'^(Chapter|CHAPTER)\s*\d*[:\.\s]*', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^(Part|PART)\s*[IVX]*[:\.\s]*', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^(Section|SECTION)\s*\d*[:\.\s]*', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^\d+[\.\:\s]+', '', title)  # Remove leading numbers
    
    # Convert to title case if all caps
    if title.isupper() and len(title) > 10:
        title = title.title()
    
    # Ensure it's not empty
    if not title.strip():
        title = "Untitled Chapter"
    
    return title.strip()

def clean_chapter_content(content: str) -> str:
    """
    Clean chapter content by removing excessive whitespace and formatting artifacts.
    
    Args:
        content (str): Raw chapter content
    
    Returns:
        str: Cleaned chapter content
    """
    # Remove excessive whitespace
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Multiple newlines to double
    content = re.sub(r'[ \t]+', ' ', content)             # Multiple spaces to single
    
    # Remove page numbers and headers/footers (common patterns)
    content = re.sub(r'\n\s*\d+\s*\n', '\n', content)    # Standalone page numbers
    content = re.sub(r'\n\s*Page\s+\d+.*?\n', '\n', content, flags=re.IGNORECASE)
    
    # Remove common PDF artifacts
    content = re.sub(r'\s*\n\s*', '\n', content)         # Clean line breaks
    content = content.strip()
    
    return content

def estimate_reading_time(content: str) -> int:
    """
    Estimate reading time for content in minutes.
    
    Args:
        content (str): Text content
    
    Returns:
        int: Estimated reading time in minutes
    """
    words = len(content.split())
    reading_speed = 200  # Average words per minute
    return max(1, words // reading_speed)
