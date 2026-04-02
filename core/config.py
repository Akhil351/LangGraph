import os

from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

def get_settings()-> dict:
    return{
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "SERPAPI_API_KEY": os.getenv("SERPAPI_API_KEY")
    }

settings=get_settings()