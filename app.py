from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

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
chatbot = pipeline("text-generation", model="gpt-2")

# Prompt the chatbot to generate questions and options
prompt = (
    "Generate a list of personality quiz questions with four options each. "
    "The questions should help identify traits like loyalty, intelligence, courage, and kindness."
)
generated_text = chatbot(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]

# Parse the generated text into a structured format
# Note: This parsing assumes the generated text is well-structured. Adjust as needed.
questions = []
for line in generated_text.split("\n"):
    if "?" in line:
        question = line.strip()
        options = []
    elif line.startswith("- "):
        options.append(line[2:].strip())
    if len(options) == 4:
        questions.append({"question": question, "options": options})

@app.route('/')
def index():
    return render_template('quiz.html', questions=questions)

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
