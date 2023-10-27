import requests
import streamlit as st

# Define the FastAPI backend URL
backend_url = "http://localhost:8001"

# Fetch books from the FastAPI backend
def fetch_books():
    response = requests.get(f"{backend_url}/get_books")
    return response.json().get("books", [])

# Fetch topics from the FastAPI backend
def fetch_topics():
    response = requests.get(f"{backend_url}/get_topics")
    return response.json()

# Streamlit app
st.title("Question Generator")

# Fetch and display available books
available_books = fetch_books()
selected_book = st.selectbox("Select Book", available_books)

# Number of questions
num_questions = st.number_input("Number of Questions", min_value=1, max_value=100, value=10)

# Topics
available_topics = fetch_topics()
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
