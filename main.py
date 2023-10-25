from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from tortoise import Tortoise
from tortoise.contrib.fastapi import register_tortoise

from database.db_operations import save_topics
from database.models import TopicDataMapping
from services.milvus_service import init_milvus, close_milvus, store_vectors
from services.text_extraction import extract_text
from services.text_normalization import unicode_normalize, remove_non_unicode, lowercase
from services.text_analysis import tokenize, remove_stopwords, remove_punctuation, lemmatize
from services.vectorization import perform_lda, generate_vector_representation
# from services.search_service import perform_search
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import uuid


app = FastAPI()

async def init():
    await Tortoise.init(
        db_url='sqlite://topics.db',
        modules={'models': ['database.models']}
    )
    await Tortoise.generate_schemas()

async def close():
    await Tortoise.close_connections()

app.add_event_handler("startup", init)
app.add_event_handler("shutdown", close)
app.add_event_handler("startup", init_milvus)
app.add_event_handler("shutdown", close_milvus)


class TopicData(BaseModel):
    book_id: str
    topics: list
    vector_ids: list  # These can be Milvus IDs for faster vector retrieval.

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/process_pdf/")
async def process_pdf(file: UploadFile = File(...), book_id: Optional[str] = None):
    try:
        # Step 1: Extract Text
        text = extract_text(file.file)
        if text is None:
            raise Exception("Text extraction failed")

        # Step 2: Normalize and Clean Text
        normalized_text = unicode_normalize(text)
        cleaned_text = remove_non_unicode(normalized_text)
        lower_text = lowercase(cleaned_text)

        # Step 3: Tokenization and Further Cleaning
        tokens = tokenize(lower_text)
        cleaned_tokens = remove_stopwords(tokens)
        cleaned_tokens = remove_punctuation(cleaned_tokens)
        lemmatized_tokens = lemmatize(cleaned_tokens)

        # Step 4: Perform LDA and Save Topics
        lda_result = perform_lda(lemmatized_tokens)
        book_id = book_id if book_id else str(uuid.uuid4())
        await save_topics(book_id, lda_result)

        # Step 5: Generate Vectors
        vectors = generate_vector_representation(lemmatized_tokens)
        vector_ids = store_vectors("text_collection", vectors)

        # Step 6: Store vector_ids with book_id
        topic_data = TopicData(book_id=book_id, topics=lda_result, vector_ids=vector_ids)
        await add_topics_endpoint(topic_data)
        
        return {"status": "success", "book_id": book_id}

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the PDF")

# Your existing code, including the Tortoise setup and other endpoints, remains the same.


@app.post("/extract_text/")
async def extract_text_endpoint(file: UploadFile = File(...)):
    # FastAPI's UploadFile automatically manages uploaded file, so we can pass its file object to extract_text
    text = extract_text(file.file)
    if text is not None:
        return {"text": text}
    else:
        return {"error": "An error occurred while extracting text from PDF."}

@app.post("/normalize_text/")
async def normalize_text_endpoint(text: str):
    normalized_text = unicode_normalize(text)
    if normalized_text is not None:
        return {"normalized_text": normalized_text}
    else:
        return {"error": "An error occurred while normalizing text."}

@app.post("/remove_non_unicode/")
async def remove_non_unicode_endpoint(text: str):
    cleaned_text = remove_non_unicode(text)
    if cleaned_text is not None:
        return {"cleaned_text": cleaned_text}
    else:
        return {"error": "An error occurred while removing non-Unicode characters."}

@app.post("/lowercase/")
async def lowercase_endpoint(text: str):
    lower_text = lowercase(text)
    if lower_text is not None:
        return {"lower_text": lower_text}
    else:
        return {"error": "An error occurred while converting text to lowercase."}

@app.post("/tokenize/")
async def tokenize_endpoint(text: str):
    tokens = tokenize(text)
    if tokens is not None:
        return {"tokens": tokens}
    else:
        return {"error": "An error occurred while tokenizing the text."}

@app.post("/remove_stopwords/")
async def remove_stopwords_endpoint(tokens: list):
    cleaned_tokens = remove_stopwords(tokens)
    if cleaned_tokens is not None:
        return {"cleaned_tokens": cleaned_tokens}
    else:
        return {"error": "An error occurred while removing stopwords."}

@app.post("/remove_punctuation/")
async def remove_punctuation_endpoint(tokens: list):
    cleaned_tokens = remove_punctuation(tokens)
    if cleaned_tokens is not None:
        return {"cleaned_tokens": cleaned_tokens}
    else:
        return {"error": "An error occurred while removing punctuation."}

@app.post("/lemmatize/")
async def lemmatize_endpoint(tokens: list):
    lemmatized_tokens = lemmatize(tokens)
    if lemmatized_tokens is not None:
        return {"lemmatized_tokens": lemmatized_tokens}
    else:
        return {"error": "An error occurred while lemmatizing the tokens."}

@app.post("/perform_lda/")
async def perform_lda_endpoint(tokens: list, book_id: str):
    try:
        lda_result = perform_lda(tokens)
        await save_topics(book_id, lda_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to perform LDA")
    return {"lda_result": str(lda_result)}

@app.post("/generate_vector_representation/")
async def generate_vector_representation_endpoint(tokens: list):
    vectors = generate_vector_representation(tokens)
    return {"vectors": vectors}

@app.post("/add_topics/")
async def add_topics_endpoint(topic_data: TopicDataMapping):
    # Logic to add topics to database
    return {"status": "success"}

# @app.post("/perform_search/")
# async def perform_search_endpoint(vector: list):
#     search_result = perform_search(vector)
#     return {"search_result": search_result}

@app.get("/get_topics/{book_id}")
async def get_topics_endpoint(book_id: str):
    # Logic to retrieve topics from database
    return {"topics": "Placeholder"}

register_tortoise(
    app,
    db_url='sqlite://topics.db',
    modules={'models': ['database.models']},
    generate_schemas=True,
    add_exception_handlers=True,
)

if __name__ == "__main__":
    # Use this for debugging purposes only
    logger.warning("Running in development mode. Do not run like this in production.")
    import uvicorn  # type: ignore

    uvicorn.run(app, host="localhost", port=8001, log_level="debug")