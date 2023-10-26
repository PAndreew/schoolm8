from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer

def perform_lda(tokens):

    # Convert list of cleaned tokens to a single string
    document = ' '.join(tokens)

    # Vectorize the text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([document])  # Note the list around `document`

    # Get the feature names (words in the vocabulary)
    feature_names = vectorizer.get_feature_names_out()

    # Perform LDA
    lda = LatentDirichletAllocation(n_components=20, random_state=0)
    lda.fit(X)

    return lda, feature_names  # Return topics and feature_names

def extract_topics(lda, feature_names, n_top_words=20):
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append({"topic_idx": topic_idx, "top_words": top_words})
    return topics

def generate_vector_representation(tokens):
    # Your existing vector generation code here
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    text_embeddings = model.encode(tokens, convert_to_tensor=True)
    return text_embeddings
