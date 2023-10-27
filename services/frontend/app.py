import requests
import streamlit as st

# Define the FastAPI backend URL
backend_url = "http://localhost:8001"

def fetch_books():
    response = requests.get(f"{backend_url}/books")
    return response.json().get("books", [])

def fetch_topics(book_id):
    response = requests.get(f"{backend_url}/get_topics/{book_id}")
    return response.json().get("topics", [])

# Streamlit app
st.title("Question Generator")

# Fetch and display available books
available_books = fetch_books()
book_options = {book['book_title']: book['book_id'] for book in available_books}
selected_book_title = st.selectbox("Select Book", list(book_options.keys()))

# Number of questions
num_questions = st.number_input("Number of Questions", min_value=1, max_value=100, value=10)

# Topics
if selected_book_title:
    selected_book_id = book_options[selected_book_title]
    available_topics = fetch_topics(selected_book_id)
    selected_topics = st.multiselect("Topics", available_topics)

# Type of questions
question_type = st.selectbox("Type of Questions", ["Egyszeres választós", "Többszörös választós", "Kifejtős"])

# Submit button
if st.button("Submit"):
    # Make a POST request to the FastAPI backend with the selected options
    payload = {
        "num_questions": num_questions,
        "topics": selected_topics,
        "question_type": question_type
    }
    response = requests.post(f"{backend_url}/generate_questions", json=payload)
    if response.status_code == 200:
        questions = response.json()
        st.write(questions)
    else:
        st.write("An error occurred.")
