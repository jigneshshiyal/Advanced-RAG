from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEY, GEMINI_MODEL

model = ChatGoogleGenerativeAI(model=GEMINI_MODEL, api_key=GEMINI_API_KEY)