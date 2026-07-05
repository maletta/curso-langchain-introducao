import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI

load_dotenv()

# 1. Listar modelos disponíveis
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

print("Modelos disponíveis:")
for model_found in client.models.list().data:
    print(f"  - {model_found.id}")

# 2. Usar um deles com LangChain
model = ChatOpenAI(
    model="deepseek-v4-flash",  ## deepseek-v4-flash ||  deepseek-v4-pro
    api_key=lambda: os.getenv("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com/v1",
    temperature=0.5,
)

message = model.invoke("Hello world")
print(message.content)
