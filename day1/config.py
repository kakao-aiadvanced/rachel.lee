import os
from dotenv import load_dotenv

load_dotenv()
SECRET_ENV = os.getenv("OPENAI_API_KEY")
print(SECRET_ENV)