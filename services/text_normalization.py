import unicodedata
import re
import logging

# Initialize logging
logging.basicConfig(filename='text_normalization.log', level=logging.INFO)

def unicode_normalize(text: str) -> str:
    try:
        normalized_text = unicodedata.normalize('NFKD', text)
        logging.info("Successfully normalized text to Unicode.")
        return normalized_text
    except Exception as e:
        logging.error(f"An error occurred during Unicode normalization: {e}")
        return None

def remove_non_unicode(text: str) -> str:
    try:
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)
        logging.info("Successfully removed non-Unicode characters.")
        return cleaned_text
    except Exception as e:
        logging.error(f"An error occurred during removal of non-Unicode characters: {e}")
        return None

def lowercase(text: str) -> str:
    try:
        lower_text = text.lower()
        logging.info("Successfully converted text to lowercase.")
        return lower_text
    except Exception as e:
        logging.error(f"An error occurred during converting text to lowercase: {e}")
        return None
