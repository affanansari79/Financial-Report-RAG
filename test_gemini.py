import os
from dotenv import load_dotenv
from google import genai

# Load the API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

# Connect to Gemini
client = genai.Client(api_key=api_key)

# Send a test question
response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Explain Retrieval-Augmented Generation (RAG) in 2 simple sentences."
)

print(response.text)