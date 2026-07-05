import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
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

question_template2 = PromptTemplate(
    input_variables=["square_result"],
    template="Tell me about the number {square_result}",
)


## decorando funções com chains, é preciso se atentar ao retorno, pois o output de um chain vai ser o input do próximo chain, então é preciso retornar um dicionário com a chave do próximo chain
@chain
def square(x: int) -> dict:
    return {"square_result": x * x}


chains = question_template | model
chains2 = square | question_template2 | model

result = chains2.invoke(10)
print(result.content)
