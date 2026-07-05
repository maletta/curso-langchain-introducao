import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="deepseek-v4-flash",  ## deepseek-v4-flash ||  deepseek-v4-pro
    api_key=lambda: os.getenv("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com/v1",
    temperature=0.5,
)

question_template = PromptTemplate(
    input_variables=["name"],
    template="Hello i'm {name}. Tell me a joke with my name",
)

chain = question_template | model  ## saída do question_template é a entrada do model

result = chain.invoke({"name": "Maletta"})

print(result.content)
