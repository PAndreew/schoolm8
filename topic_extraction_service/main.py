from fastapi import FastAPI
from pydantic import BaseModel
from text_processor import TextProcessor
from topic_model import TopicModel
from embedding_model import EmbeddingModel
from typing import Optional
import spacy

app = FastAPI()

class TopicRequest(BaseModel):
    file_path: str
    n_topics: Optional[int] = 20

@app.post("/extract_topics/")
async def extract_topics(request: TopicRequest):
    try:
        nlp_model = spacy.load("hu_core_news_lg")
        text_processor = TextProcessor(nlp_model)
        topic_model = TopicModel(n_topics=request.n_topics)
        embedding_model = EmbeddingModel()

        raw_text = text_processor.extract_text(request.file_path)
        cleaned_text = text_processor.clean_text(raw_text)

        topics = topic_model.extract_topics(cleaned_text)
        embeddings = embedding_model.generate_embeddings(cleaned_text)

        return {"success": True, "topics": topics, "embeddings": embeddings}

    except Exception as e:
        # Proper logging should be here
        return {"success": False, "error": str(e)}
