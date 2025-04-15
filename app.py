from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()
api_key = os.getenv("API_KEY")
client = genai.Client(api_key=api_key)

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define anime characters and their traits
characters = {
    "Naruto Uzumaki": "energetic, loyal, impulsive, never gives up",
    "Lelouch Lamperouge": "strategic, intelligent, calm, manipulative",
    "Goku": "brave, strong, carefree, determined",
    "Light Yagami": "smart, ambitious, justice-driven, secretive",
    "Hinata Hyuga": "shy, kind, loyal, observant",
    "Eren Yeager": "angry, vengeful, passionate, rebellious"
}

# Use a pretrained chatbot model to generate questions and options

response = client.models.generate_content(
    model="gemini-2.0-flash", 
    contents=[
        "Generate a list of personality quiz questions with four options each. "
        "The output should be in the following format:\n"
        "Question: <question text>\n"
        "Options:\n"
        "- <option 1>\n"
        "- <option 2>\n"
        "- <option 3>\n"
        "- <option 4>\n"
        "Repeat this structure for each question."
    ]
).text

# Debugging: Print the raw response to verify its content
print("Raw Response:", response)  # Debugging line to check the raw response

# Parse the generated text into separate lists for questions and options
questions = []
options = []

lines = response.split("\n")
current_question = None
current_options = []

for line in lines:
    line = line.strip()
    if line.startswith("Question:"):
        # Save the previous question and its options if any
        if current_question and current_options:
            questions.append(current_question)
            options.append(current_options)
        # Start a new question
        current_question = line[len("Question:"):].strip()
        current_options = []
    elif line.startswith("-"):
        # Add an option to the current question
        current_options.append(line[1:].strip())

# Add the last question and its options
if current_question and current_options:
    questions.append(current_question)
    options.append(current_options)

# Fallback: Check if questions and options are empty
if not questions or not options:
    print("Parsing failed. Please check the response format.")
    questions = ["Sample Question 1", "Sample Question 2"]  # Fallback questions
    options = [["Option A", "Option B", "Option C", "Option D"],  # Fallback options
               ["Option 1", "Option 2", "Option 3", "Option 4"]]

print("Questions:", questions)  # Debugging line to check extracted questions
print("Options:", options)      # Debugging line to check extracted options

@app.route('/')
def index():
    return render_template('quiz.html', questions=questions, options=options)

@app.route('/submit', methods=['POST'])
def submit():
    answers = [request.form.get(f'q{i}') for i in range(len(questions))]
    user_profile = ", ".join(answers)
    user_embedding = model.encode(user_profile, convert_to_tensor=True)

    best_match = None
    best_score = -1

    for name, traits in characters.items():
        char_embedding = model.encode(traits, convert_to_tensor=True)
        score = util.pytorch_cos_sim(user_embedding, char_embedding).item()
        if score > best_score:
            best_score = score
            best_match = name

    return render_template('result.html', character=best_match)

if __name__ == '__main__':
    app.run(debug=True)
