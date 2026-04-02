from langchain_openai import ChatOpenAI
from core import settings


def get_open_ai_model():
    """Return OpenAI model instance"""
    return ChatOpenAI(
        model="gpt-5.2",
        temperature=0,
        api_key=settings["OPENAI_API_KEY"]
    )

