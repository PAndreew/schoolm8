from fastapi import FastAPI, File, UploadFile
from services.text_extraction import extract_text
from services.text_normalization import unicode_normalize, remove_non_unicode, lowercase
from services.text_analysis import tokenize, remove_stopwords, remove_punctuation, lemmatize
from services.vectorization import perform_lda, generate_vector_representation
from services.search_service import perform_search

app = FastAPI()

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
    return {"normalized_text": normalized_text}

@app.post("/remove_non_unicode/")
async def remove_non_unicode_endpoint(text: str):
    cleaned_text = remove_non_unicode(text)
    return {"cleaned_text": cleaned_text}

@app.post("/lowercase/")
async def lowercase_endpoint(text: str):
    lower_text = lowercase(text)
    return {"lower_text": lower_text}

@app.post("/tokenize/")
async def tokenize_endpoint(text: str):
    tokens = tokenize(text)
    return {"tokens": tokens}

@app.post("/remove_stopwords/")
async def remove_stopwords_endpoint(tokens: list):
    cleaned_tokens = remove_stopwords(tokens)
    return {"cleaned_tokens": cleaned_tokens}

@app.post("/remove_punctuation/")
async def remove_punctuation_endpoint(tokens: list):
    cleaned_tokens = remove_punctuation(tokens)
    return {"cleaned_tokens": cleaned_tokens}

@app.post("/lemmatize/")
async def lemmatize_endpoint(tokens: list):
    lemmatized_tokens = lemmatize(tokens)
    return {"lemmatized_tokens": lemmatized_tokens}

@app.post("/perform_lda/")
async def perform_lda_endpoint(tokens: list):
    lda_result = perform_lda(tokens)
    return {"lda_result": lda_result}

@app.post("/generate_vector_representation/")
async def generate_vector_representation_endpoint(tokens: list):
    vectors = generate_vector_representation(tokens)
    return {"vectors": vectors}

@app.post("/perform_search/")
async def perform_search_endpoint(vector: list):
    search_result = perform_search(vector)
    return {"search_result": search_result}
