import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

system = ("system", "you are an assistant that answers questions in a {style} style")
user = ("user", "{question}")

chat_prompt = ChatPromptTemplate([system, user])

messages = chat_prompt.format_messages(style="funny", question="who is alan turing?")

for msg in messages:
    print(f"{msg.type}: {msg.content}")

model = ChatOpenAI(
    model="deepseek-v4-flash",  ## deepseek-v4-flash ||  deepseek-v4-pro
    api_key=lambda: os.getenv("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com/v1",
    temperature=0.5,
)

result = model.invoke(messages)

print(result.content)
