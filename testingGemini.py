from google import genai

client = genai.Client(api_key="AIzaSyC2cqP6LarFWrEUFpywDOc6dVQGcGPiyCE")

response = client.models.generate_content(
    model="gemini-2.0-flash", contents=["Generate a list of personality quiz questions with four options each. "
    "The questions should help identify traits like loyalty, intelligence, courage, and kindness."]
)
print(response.text)