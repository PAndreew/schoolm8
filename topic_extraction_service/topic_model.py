import logging
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.exceptions import NotFittedError
from typing import List, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TopicModel:
    def __init__(self, n_topics: int = 20):
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer()
        self.lda = LatentDirichletAllocation(n_components=self.n_topics, random_state=0)

    def extract_topics(self, cleaned_text: str) -> Union[List[Tuple[str, float]], None]:
        """
        Extracts topics from the given cleaned text using LDA.

        Args:
            cleaned_text: Preprocessed and cleaned text string.

        Returns:
            A list of topics with their respective weights, or None if extraction fails.
        """
        try:
            # Vectorize the cleaned text
            data_vectorized = self.vectorizer.fit_transform([cleaned_text])

            # Perform LDA
            self.lda.fit(data_vectorized)

            # Get the topics
            feature_names = self.vectorizer.get_feature_names_out()
            topics = self._get_topics(feature_names)

            logging.info("Successfully extracted topics.")
            return topics

        except Exception as e:
            logging.error(f"An error occurred while extracting topics: {e}")
            return None

    def _get_topics(self, feature_names: List[str]) -> List[Tuple[str, float]]:
        """
        Internal method to get the topics from the LDA model.

        Args:
            feature_names: List of feature names from the vectorizer.

        Returns:
            A list of topics with their respective weights.
        """
        try:
            topics = []
            for topic_idx, topic in enumerate(self.lda.components_):
                topic_features = [(feature_names[i], topic[i]) for i in topic.argsort()[:-10 - 1:-1]]
                topics.append((f"Topic {topic_idx+1}", topic_features))
            
            return topics

        except NotFittedError as e:
            logging.error("The LDA model is not fitted yet.")
            raise e
        except Exception as e:
            logging.error(f"An error occurred while getting topics: {e}")
            raise e
