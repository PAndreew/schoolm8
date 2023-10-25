import fitz
import logging
import spacy
import unicodedata
import re
from typing import Union

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextProcessor:
    def __init__(self, nlp_model):
        """
        Initialize TextProcessor with a specific NLP model.
        
        :param nlp_model: The NLP model to be used for text processing.
        """
        self.nlp_model = nlp_model

    def extract_text(self, file_path: str) -> Union[str, None]:
        """
        Extract text from a PDF file.
        
        :param file_path: Path to the PDF file.
        :return: Extracted text or None if an error occurs.
        """
        try:
            with fitz.open(file_path) as pdf_document:
                text = ''.join(page.get_text() for page in pdf_document)
            logging.info("Text successfully extracted from PDF.")
            return text
        except Exception as e:
            logging.error(f"An error occurred while extracting text from PDF: {e}")
            raise Exception(f"An error occurred while extracting text from PDF: {e}") from e

    def clean_text(self, raw_text: str) -> str:
        """
        Clean and preprocess the text.
        
        :param raw_text: The raw text to be cleaned.
        :return: Cleaned and preprocessed text.
        """
        try:
            text = self._remove_non_unicode(raw_text)
            text = self._unicode_normalize(text)
            text = self._lowercase(text)
            tokens = self._tokenize(text)
            tokens = self._remove_stopwords(tokens)
            tokens = self._remove_punctuation(tokens)
            cleaned_text = self._lemmatize(tokens)
            logging.info("Text successfully cleaned and preprocessed.")
            return cleaned_text
        except Exception as e:
            logging.error(f"An error occurred while cleaning text: {e}")
            raise Exception(f"An error occurred while cleaning text: {e}") from e

    def _unicode_normalize(self, text: str) -> str:
        """
        Normalize unicode characters in the text.
        
        :param text: The text to be normalized.
        :return: Unicode normalized text.
        """
        return unicodedata.normalize('NFKD', text)

    def _remove_non_unicode(self, text: str) -> str:
        """
        Remove non-unicode characters from the text.
        
        :param text: The text from which non-unicode characters should be removed.
        :return: Text with non-unicode characters removed.
        """
        return re.sub(r'[^\x00-\x7F]+', '', text)

    def _lowercase(self, text: str) -> str:
        """
        Convert the text to lowercase.
        
        :param text: The text to be converted to lowercase.
        :return: Lowercased text.
        """
        return text.lower()

    def _tokenize(self, text: str) -> list:
        """
        Tokenize the text using the NLP model.
        
        :param text: The text to be tokenized.
        :return: List of tokens.
        """
        return [token.text for token in self.nlp_model(text)]

    def _remove_stopwords(self, tokens: list) -> list:
        """
        Remove stopwords from the list of tokens.
        
        :param tokens: The list of tokens from which stopwords should be removed.
        :return: List of tokens with stopwords removed.
        """
        return [token for token in tokens if token.lower() not in self.nlp_model.Defaults.stop_words]

    def _remove_punctuation(self, tokens: list) -> list:
        """
        Remove punctuation from the list of tokens.
        
        :param tokens: The list of tokens from which punctuation should be removed.
        :return: List of tokens with punctuation removed.
        """
        return [re.sub(r'[^\w\s]', '', token) for token in tokens]

    def _lemmatize(self, tokens: list) -> str:
        """
        Lemmatize the list of tokens using the NLP model.
        
        :param tokens: The list of tokens to be lemmatized.
        :return: String of lemmatized tokens joined by spaces.
        """
        lemmatized_tokens = [token.lemma_ for token in self.nlp_model(' '.join(tokens))]
        return ' '.join(lemmatized_tokens)
