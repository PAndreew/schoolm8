from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Json
from tortoise import Tortoise
from tortoise.contrib.fastapi import register_tortoise
from tortoise.exceptions import DoesNotExist
from tortoise.functions import Count

from database.db_operations import save_topics
from database.models import TopicDataMapping as TopicDataMappingTortoise
from services.milvus_service import init_milvus, close_milvus, store_vectors
from services.text_extraction import extract_text, extract_book_title_from_path
from services.text_normalization import unicode_normalize, remove_non_unicode, lowercase
from services.text_analysis import tokenize, remove_stopwords, remove_punctuation, lemmatize, clean_text, remove_emails, remove_numbers, remove_urls
from services.vectorization import perform_lda, generate_vector_representation, extract_topics
from services.llm_generation import generate_composition_questions
# from services.search_service import perform_search
from fastapi.responses import JSONResponse
from typing import Optional, List
import logging
import uuid
import uvicorn
import sys

logger = logging.getLogger(__name__)

def configure_logging():
    # Configure your logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Also configure Uvicorn's log formatting if needed
    uvicorn_logger = logging.getLogger("uvicorn.access")
    console_formatter = uvicorn.logging.ColourizedFormatter(
        "{asctime} {levelprefix} : {message}",
        style="{", use_colors=True
    )
    uvicorn_logger.handlers[0].setFormatter(console_formatter)

app = FastAPI()

origins = [
    "http://localhost:8501",  # Allow requests from your Streamlit app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from the origins specified above
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

async def init():
    await Tortoise.init(
        db_url='sqlite://topics.db',
        modules={'models': ['database.models']}
    )
    await Tortoise.generate_schemas()

async def close():
    await Tortoise.close_connections()

app.add_event_handler("startup", init)
app.add_event_handler("startup", configure_logging)
app.add_event_handler("shutdown", close)
# app.add_event_handler("startup", init_milvus)
# app.add_event_handler("shutdown", close_milvus)


class TopicDataMapping(BaseModel):
    id: int
    book_id: str
    book_title: str
    topic: Json
    # vector_id: int

class CompositionRequest(BaseModel):
    topics: List[str]
    number_of_questions: int
    type_of_questions: str

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/generate_composition/")
async def generate_composition(request: CompositionRequest):
    topics = request.topics
    number_of_questions = request.number_of_questions
    type_of_questions = request.type_of_questions

    generated_text = generate_composition_questions(topics, number_of_questions, type_of_questions)
    
    return {"generated_text": generated_text}

@app.post("/process_pdf/")
async def process_pdf(file: UploadFile = File(...), book_id: Optional[str] = None, book_title: Optional[str] = None):
    try:
        # Step 0: Extract book title
        book_title = extract_book_title_from_path(file.filename)
        print(book_title)

        # Step 1: Extract Text
        text = extract_text(file.file)
        if text is None:
            raise Exception("Text extraction failed")
        # print(text)
        # Step 2: Normalize and Clean Text
        normalized_text = unicode_normalize(text)
        text_without_non_unicode = remove_non_unicode(normalized_text)
        lower_text = lowercase(text_without_non_unicode)
        cleaned_text = clean_text(lower_text)
        text_without_urls = remove_urls(cleaned_text)
        text_without_emails = remove_emails(text_without_urls)
        text_without_numbers = remove_numbers(text_without_emails)

        # Step 3: Tokenization and Further Cleaning
        tokens = tokenize(text_without_numbers)
        cleaned_tokens = remove_stopwords(tokens)
        cleaned_tokens = remove_punctuation(cleaned_tokens)
        lemmatized_tokens = lemmatize(cleaned_tokens)
        # print(lemmatized_tokens)

        # Step 4: Perform LDA and Save Topics
        lda, feature_names = perform_lda(lemmatized_tokens)
        # print(lda, feature_names)
        print(lda.components_.shape)
        print(len(feature_names))


        # Step 5: Extract and serialise the topics
        serialised_topics = extract_topics(lda, feature_names)
        print(serialised_topics)
        book_id = book_id if book_id else str(uuid.uuid4())
        await save_topics(book_id, book_title, serialised_topics)

        # # Step 6: Generate Vectors
        # vectors = generate_vector_representation(lemmatized_tokens)
        # vector_ids = store_vectors("text_collection", vectors)

        # # Step 6: Store vector_ids with book_id
        # topic_data = TopicDataMapping(book_id=book_id, topics=lda_result, vector_ids=vector_ids)
        # await add_topics_endpoint(topic_data)
        
        return {"status": "success", "book_id": book_id}

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the PDF")

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
    try:
        # Fetch topics from the database using Tortoise ORM
        query = await TopicDataMappingTortoise.filter(book_id=book_id).all()
        
        if not query:
            return {"status": "error", "message": "No topics found for the given book_id"}
        
        # Serialize the query result to JSON
        topics = [topic.topic for topic in query]
        
        return {"status": "success", "topics": topics}
        
    except DoesNotExist:
        return {"status": "error", "message": "Book ID does not exist"}
    except Exception as e:
        return {"status": "error", "message": f"An error occurred: {e}"}

@app.get("/books/")
async def get_books_endpoint():
    # try:
    # A simplified query to fetch distinct book titles.
    query = await TopicDataMappingTortoise.all().distinct().values('book_id', 'book_title')
    print(query)  # This will print the query result to the console

    if not query:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No books found")

    return {"status": "success", "books": query}
    # except DoesNotExist:
    #     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No books found")
    # except Exception as e:
    #     raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {e}")

register_tortoise(
    app,
    db_url='sqlite://topics.db',
    modules={'models': ['database.models']},
    generate_schemas=True,
    add_exception_handlers=True,
)

import requests
from typing import Dict, Any

def call_process_pdf_endpoint(url: str, file_path: str) -> Dict[str, Any]:
    """
    Calls the process_pdf FastAPI endpoint to upload a PDF file and process it.

    Parameters:
        url (str): The URL of the FastAPI endpoint.
        file_path (str): The file path of the PDF to be uploaded.

    Returns:
        Dict[str, Any]: The JSON response from the FastAPI endpoint.
    """
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to process PDF. Status code: {response.status_code}, Message: {response.text}")
            return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

if __name__ == "__main__":
    url = "http://127.0.0.1:8001/process_pdf/"
    # file_path = r"C:\Users\uif56391\Downloads\OH-KEM78TA_I__teljes.pdf"
    file_path = r"C:\Users\uif56391\Downloads\Resume_Andras_Pasztor.pdf"
    response = call_process_pdf_endpoint(url, file_path)
    print(response)
