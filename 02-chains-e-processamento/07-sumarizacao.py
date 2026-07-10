import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Texto de exemplo — pequeno, mas vamos simular um cenário real
long_text = """
The midnight oil has burned to ash and grey,
As silent shadows stretch across the floor.
The weary mind resists the coming day,
And lingers by the heavy, closed bedroom door.
A sea of thoughts, a turbulent, wild stream,
Where memories and undone tasks collide,
We navigate the space between a dream,
And all the secrets that the dark can hide.

But time, a steady river, never waits,
It pulls the velvet curtain of the night.
And opens up the morning's golden gates,
To flood the world with soft, forgiving light.
The birds begin their faint, melodic song,
A gentle choir waking up the trees.
They whisper that the darkness won't last long,
Carried away upon a waking breeze.

So let the heavy burdens of the past,
Dissolve like mist beneath the rising sun.
The longest, coldest nights can never last,
A brand-new chapter has at last begun.
Take a deep breath and step into the glow,
The world is waiting, fresh and clean and new.
Leave all the doubts that held you back below,
The sky is bright, and open just for you.
"""

sep = "-" * 15

llm = ChatOpenAI(
    model="deepseek-v4-flash",  ## deepseek-v4-flash ||  deepseek-v4-pro
    api_key=lambda: os.getenv("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com/v1",
    temperature=0.5,
)

# ═══════════════════════════════════════════════════════════════
# ESTRATÉGIA 2: MAP_REDUCE — divide, resume cada parte, combina
# Útil quando o texto NÃO cabe na janela de contexto
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*50}")
print("ESTRATÉGIA 2: MAP_REDUCE (divide e combina)")
print(f"{'='*50}\n")

# Dividir o texto em chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.create_documents([long_text])

# LCEL map stage: summarize each chunk
map_prompt = PromptTemplate.from_template(
    "Write a concise summary of the following text:\n{context}"
)

# Lang Chain Expression Language LCEL
map_chain = map_prompt | llm | StrOutputParser()


# usado para preparar os inputs para o map_stage, já que o map_chain espera um dict com a chave "context" e não objeto Document
# List comprehension (parecido com [].map de javascript) é a nota de Python para criar uma lista a partir de outra lista, aplicando uma função ou expressão a cada elemento da lista original. Aqui, estamos criando uma nova lista de dicionários, onde cada dicionário contém o conteúdo de cada chunk de texto sob a chave "context". Isso é necessário porque o map_chain espera receber um dicionário com a chave "context" em vez de um objeto Document diretamente.
def prepare_inputs(docs: List[Document]) -> List[dict]:
    return [{"context": doc.page_content} for doc in docs]


prepare_map_inputs = RunnableLambda(prepare_inputs)
map_stage = prepare_map_inputs | map_chain.map()

# LCEL reduce stage: combine the summaries into a final summary
reduce_prompt = PromptTemplate.from_template(
    "Combine the following summaries into a single concise summary:\n{context}"
)
reduce_chain = reduce_prompt | llm | StrOutputParser()


def combine_summaries(summaries: List[str]) -> dict:
    combined_summary = "\n".join(summaries)
    print(f"Combining summaries:\n{combined_summary}")
    return {"context": combined_summary}


prepare_reduce_input = RunnableLambda(combine_summaries)
pipeline = map_stage | prepare_reduce_input | reduce_chain

result = pipeline.invoke(chunks)
print(f"\n{'='*15} RESULTADO FINAL {'='*15}\n")
print(result)
