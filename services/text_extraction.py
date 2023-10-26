import fitz
import logging
import os

# Initialize logging
logging.basicConfig(filename='text_extraction.log', level=logging.INFO)

def extract_book_title_from_path(file_path: str) -> str:
    # Extract the filename from the path
    filename = os.path.basename(file_path)
    
    # Remove the file extension to get the title
    book_title = os.path.splitext(filename)[0]
    
    return book_title

def extract_text(file):
    try:
        # Use a with statement to ensure that resources are managed correctly
        with fitz.open(stream=file.read(), filetype="pdf") as pdf_document:
            text = ""
            for page in pdf_document:
                text += page.get_text()
                
            logging.info(f"Successfully extracted text from PDF.")
            
            return text

    except Exception as e:
        logging.error(f"An error occurred while extracting text: {e}")
        return None
    