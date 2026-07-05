from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

gemini = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
answer_gemini = gemini.invoke("Hello world")
print(answer_gemini.content)
