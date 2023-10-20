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
        self.nlp_model = nlp_model

    def extract_text(self, file_path: str) -> Union[str, None]:
        try:
            with fitz.open(file_path) as pdf_document:
                text = ''.join(page.get_text() for page in pdf_document)
            logging.info("Text successfully extracted from PDF.")
            return text
        except Exception as e:
            logging.error(f"An error occurred while extracting text from PDF: {e}")
            raise Exception(f"An error occurred while extracting text from PDF: {e}") from e

    def clean_text(self, raw_text: str) -> str:
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
        return unicodedata.normalize('NFKD', text)

    def _remove_non_unicode(self, text: str) -> str:
        return re.sub(r'[^\x00-\x7F]+', '', text)

    def _lowercase(self, text: str) -> str:
        return text.lower()

    def _tokenize(self, text: str) -> list:
        return [token.text for token in self.nlp_model(text)]

    def _remove_stopwords(self, tokens: list) -> list:
        return [token for token in tokens if token.lower() not in self.nlp_model.Defaults.stop_words]

    def _remove_punctuation(self, tokens: list) -> list:
        return [re.sub(r'[^\w\s]', '', token) for token in tokens]

    def _lemmatize(self, tokens: list) -> str:
        lemmatized_tokens = [token.lemma_ for token in self.nlp_model(' '.join(tokens))]
        return ' '.join(lemmatized_tokens)
