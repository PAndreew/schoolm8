import logging
import numpy as np
from sentence_transformers import SentenceTransformer, SentenceTransformerError
from typing import Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingModel:
    def __init__(self, model_name: str = 'paraphrase-distilroberta-base-v1'):
        try:
            self.model = SentenceTransformer(model_name)
            logging.info(f"Loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            logging.error(f"An error occurred while loading SentenceTransformer model: {e}")
            raise e

    def generate_embeddings(self, cleaned_text: str) -> Union[np.ndarray, None]:
        """
        Generates embeddings for the given cleaned text.

        Args:
            cleaned_text: Preprocessed and cleaned text string.

        Returns:
            A numpy array containing the generated embeddings, or None if generation fails.
        """
        try:
            # Generate embeddings
            embeddings = self.model.encode([cleaned_text], convert_to_numpy=True)

            logging.info("Successfully generated embeddings.")
            return embeddings[0]

        except SentenceTransformerError as e:
            logging.error(f"A SentenceTransformer specific error occurred while generating embeddings: {e}")
            return None
        except Exception as e:
            logging.error(f"An error occurred while generating embeddings: {e}")
            return None
