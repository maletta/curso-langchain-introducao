import os
from pprint import pprint

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

template_summary = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 4 words: \n ```{text}```",
)


template_translate = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to English: \n ```{initial_text}```",
)

# value = template_translate.invoke({"initial_text": "Hola, ¿cómo estás?"})
# print(value)
# print(type(value))
# pprint(vars(value)) # imprime a estrutura do objeto, mostrando seus atributos e valores

llm_en = ChatOpenAI(
    model="deepseek-v4-flash",  ## deepseek-v4-flash ||  deepseek-v4-pro
    api_key=lambda: os.getenv("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com/v1",
    temperature=0,
)

translate = template_translate | llm_en | StrOutputParser()
pipeline = {"text": translate} | template_summary | llm_en | StrOutputParser()

result = pipeline.invoke(
    {
        "initial_text": "LangChain é um framework para desenvolvimento de aplicações de IA"
    }
)

print(result)
