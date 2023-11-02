import requests
import streamlit as st

# Define the FastAPI backend URL
backend_url = "http://localhost:8001"

def fetch_books():
    response = requests.get(f"{backend_url}/books")
    return response.json().get("books", [])

def fetch_topics(book_id):
    response = requests.get(f"{backend_url}/get_topics/{book_id}")
    topics = response.json().get("topics", [])
    # Flatten the list of top_words arrays into a single list of words
    top_words = [word for topic in topics for word in topic['top_words']]
    return top_words

# Streamlit app
st.title("Question Generator")

# Subject
subject = st.selectbox("Select Subject", ["Kémia"])

# Grade
grade = st.selectbox("Select Grade", ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6", "Grade 7", "Grade 8"])

# List of topics
topic_list = [
    "Biztonságos Kísérletezés Szabályai",
    "Anyagok Szerkezete és Tulajdonságai",
    "Elemek Periódusos Rendszere"
]

# Fetch and display available books
# if 'available_books' not in st.session_state:
#     st.session_state['available_books'] = fetch_books()
# book_options = {book['book_title']: book['book_id'] for book in st.session_state['available_books']}
# selected_book_title = st.selectbox("Select Book", list(book_options.keys()))

# Number of questions
num_questions = st.number_input("Number of Questions", min_value=1, max_value=100, value=10)

# Topics
selected_topics = st.multiselect("Topics", topic_list)  # Use the predefined topic_list

# Type of questions
question_type = st.selectbox("Type of Questions", ["Egyszeres választós", "Többszörös választós", "Kifejtős"])

# Submit button
if st.button("Submit"):
    # Make a POST request to the FastAPI backend with the selected options
    payload = {
        "number_of_questions": num_questions,
        "topics": selected_topics,
        "type_of_questions": question_type,
        "grade": grade
    }
    response = requests.post(f"{backend_url}/generate_composition", json=payload)
    if response.status_code == 200:
        generated_text = response.json().get('generated_text', '')
        st.write(generated_text)
    else:
        st.write("An error occurred.")
