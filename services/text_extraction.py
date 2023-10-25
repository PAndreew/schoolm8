import fitz
import logging

# Initialize logging
logging.basicConfig(filename='text_extraction.log', level=logging.INFO)

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
    