from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util

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

# Questions and options
questions = [
    {
        "question": "How do you handle conflict?",
        "options": ["Stay calm and find a solution", "Confront head-on", "Use strategy", "Let emotions drive me"]
    },
    {
        "question": "What motivates you the most?",
        "options": ["Protecting loved ones", "Becoming stronger", "Achieving justice", "Outsmarting opponents"]
    },
    {
        "question": "Pick a trait you value most:",
        "options": ["Loyalty", "Intelligence", "Courage", "Kindness"]
    },
    {
        "question": "In a team, what role do you play?",
        "options": ["Leader", "Supporter", "Strategist", "Wild card"]
    },
    {
        "question": "How do you deal with failure?",
        "options": ["Learn from it", "Try harder", "Blame others", "Get emotional"]
    }
]

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
