import spacy
from spacy.lang.hu.stop_words import STOP_WORDS
import re
import logging

# Initialize logging
logging.basicConfig(filename='text_analysis.log', level=logging.INFO)

# Load the spaCy model
try:
    nlp = spacy.load("hu_core_news_lg")
    logging.info("Successfully loaded spaCy model.")
except Exception as e:
    logging.error(f"An error occurred while loading spaCy model: {e}")

def clean_text(text):
    return text.replace('\n', ' ').replace('\r', '').strip()

def tokenize(text: str) -> list:
    try:
        doc = nlp(text)
        tokens = [token.text for token in doc]
        logging.info("Successfully tokenized the text.")
        return tokens
    except Exception as e:
        logging.error(f"An error occurred during tokenization: {e}")
        return None

import re

def remove_emails(text: str) -> str:
    """
    Remove email addresses from the text.

    Parameters:
        text (str): The input text.

    Returns:
        str: The text with email addresses removed.
    """
    return re.sub(r'\S*@\S*\s?', '', text)

def remove_urls(text: str) -> str:
    """
    Remove URLs from the text.

    Parameters:
        text (str): The input text.

    Returns:
        str: The text with URLs removed.
    """
    return re.sub(r'http\S+', '', text)

def remove_numbers(text: str) -> str:
    """
    Remove numbers from the text.

    Parameters:
        text (str): The input text.

    Returns:
        str: The text with numbers removed.
    """
    return re.sub(r'\d+', '', text)

def remove_stopwords(tokens: list) -> list:
    try:
        cleaned_tokens = [token for token in tokens if token not in STOP_WORDS]
        logging.info("Successfully removed stopwords.")
        return cleaned_tokens
    except Exception as e:
        logging.error(f"An error occurred during removing stopwords: {e}")
        return None

def remove_punctuation(tokens: list) -> list:
    try:
        cleaned_tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
        logging.info("Successfully removed punctuation.")
        return cleaned_tokens
    except Exception as e:
        logging.error(f"An error occurred during removing punctuation: {e}")
        return None

def lemmatize(tokens: list) -> list:
    try:
        lemmatized_tokens = [token.lemma_ for token in nlp(' '.join(tokens))]
        logging.info("Successfully lemmatized the tokens.")
        return lemmatized_tokens
    except Exception as e:
        logging.error(f"An error occurred during lemmatization: {e}")
        return None
