from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.0-flash", contents=["Generate a list of personality quiz questions with four options each. "
    "The questions should help identify traits like loyalty, intelligence, courage, and kindness."]
)
print(response.text)