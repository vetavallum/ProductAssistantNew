import os

from dotenv import load_dotenv
# Load environment variables from the .env file
load_dotenv()

# Access the environment variables just like you would with os.environ
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

print("Value of MY_VARIABLE:", OPENAI_API_KEY)
