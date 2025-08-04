from dotenv import load_dotenv
load_dotenv()

class LLMInterface:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
