from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load environment variables from .env file
load_dotenv()

# Get the GEMINI_KEY environment variable
GEMINI_KEY = os.getenv("GEMINI_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_KEY)