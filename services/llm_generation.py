import os
import openai
from dotenv import load_dotenv

load_dotenv()

def compile_prompts(topics, number_of_questions, type_of_questions, grade):
    """Prepare input prompts for the model. 
    It's necessary because the price shall be calculated separately for input and output."""
    json_example = '''{
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
    
    user_prompt = f"Based on the provided topics: {topics} generate {number_of_questions} questions of type {type_of_questions} for {grade} students."
    system_prompt = f"You are a helpful educational assistant. \
            Your mission is to create engaging and fun school compositions. \
            Your exercises test the students' understanding of the topics from different angles. \
            You ALWAYS create the questions in Hungarian. \
            You ALWAYS respond in valid JSON format. \
            Here's a good example: {json_example}"

    return system_prompt, user_prompt

def generate_composition_questions(system_prompt, user_prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(  
      model="gpt-3.5-turbo",
      messages=[{
          "role": "system",
          "content": system_prompt
        },
        {
          "role": "user",
          "content": user_prompt
        }]
    )

    return response['choices'][0]['message']['content'].strip()

