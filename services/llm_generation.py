import openai
from dotenv import load_dotenv
import os

load_dotenv()

def generate_composition_questions(topics, number_of_questions, type_of_questions, grade):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    json_example = '''{
    "timestamp": "2023-10-26T14:30:00Z",
    "book_id": "123456789",
    "grade": 9,
    "questions": [
        {
        "question_number": 1,
        "question_type": "Egyszeres választós",
        "question_text": "Mi a Pitagorasz-tétel alapelve?",
        "choices": [
            "A derékszögű háromszög két befogójának négyzetösszege egyenlő az átfogó négyzetével.",
            "A derékszögű háromszög két befogójának négyzetösszege egyenlő az átfogó hosszával.",
            "A derékszögű háromszög területe egyenlő az átfogó hosszával.",
            "A derékszögű háromszög szögeinek összege 180 fok."
        ],
        "correct_choice": "A"
        },
        {
        "question_number": 2,
        "question_type": "Kifejtős",
        "question_text": "Adott egy derékszögű háromszög, aminek az egyik befogója 3 cm, a másik befogója 4 cm. Számítsd ki az átfogó hosszát a Pitagorasz-tétel segítségével."
        }
    ]
    }'''

    prompt = f"As an AI trained in educational assistance, I am tasked with helping students create school compositions. \
    Based on the provided topics: {topics}, I shall generate {number_of_questions} questions of type {type_of_questions} for {grade} students to assist in writing a composition. \
    I shall ALWAYS create the questions in Hungarian. \
    I shall ALWAYS respond in valid JSON format. \
    Here's a good example: {json_example}"


    response = openai.ChatCompletion.create(  
      model="gpt-3.5-turbo",
      messages=[{
          "role": "system",
          "content": "You are a helpful assistant."
      },{
          "role": "user",
          "content": prompt
      }]
    )

    return response['choices'][0]['message']['content'].strip()

