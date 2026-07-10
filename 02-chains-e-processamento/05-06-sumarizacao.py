import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
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
# ESTRATÉGIA 1: STUFF — texto inteiro de uma vez
# Útil quando o texto cabe na janela de contexto, tender a ser mais
# preciso, rápido e barato
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*50}")
print("ESTRATÉGIA 1: STUFF (texto inteiro)")
print(f"{'='*50}\n")

template_stuff = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following poem concisely:\n\n{text}",
)

chain_stuff = template_stuff | llm | StrOutputParser()
result_stuff = chain_stuff.invoke({"text": long_text})
print(result_stuff)

# ═══════════════════════════════════════════════════════════════
# ESTRATÉGIA 2: MAP_REDUCE — divide, resume cada parte, combina
# Útil quando o texto NÃO cabe na janela de contexto
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*50}")
print("ESTRATÉGIA 2: MAP_REDUCE (divide e combina)")
print(f"{'='*50}\n")

# 2a. Dividir o texto em chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=70,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.create_documents([long_text])
print(f"Texto dividido em {len(chunks)} chunks:\n")

# 2b. MAP: resumir cada chunk individualmente
template_map = PromptTemplate(
    input_variables=["text"],
    template="Write a brief summary of this poem excerpt:\n\n{text}",
)
chain_map = template_map | llm | StrOutputParser()

summaries = []
for i, chunk in enumerate(chunks):
    summary = chain_map.invoke({"text": chunk.page_content})
    summaries.append(summary)
    print(f"Chunk {i+1} ({len(chunk.page_content)} chars):")
    print(f"  {chunk.page_content[:60].replace(chr(10), ' ')}...")
    print(f"  Resumo: {summary}")
    print()

# 2c. REDUCE: combinar todos os resumos num resumo final
template_reduce = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are given multiple summaries of parts of a poem.\n"
        "Combine them into a single, coherent summary of the whole poem:\n\n"
        "{text}"
    ),
)
chain_reduce = template_reduce | llm | StrOutputParser()

combined = "\n\n".join(summaries)
final_summary = chain_reduce.invoke({"text": combined})

print(f"{sep} RESUMO FINAL (MAP_REDUCE) {sep}")
print(final_summary)

# ═══════════════════════════════════════════════════════════════
# COMPARAÇÃO
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*50}")
print("COMPARAÇÃO DOS RESULTADOS")
print(f"{'='*50}\n")

print(f"[STUFF]     — 1 chamada à API, modelo vê o texto inteiro:")
print(f"  {result_stuff[:100]}...\n")

print(
    f"[MAP_REDUCE] — {len(chunks)+1} chamadas à API, texto dividido em {len(chunks)} partes:"
)
print(f"  {final_summary[:100]}...")

# MAP_REDUCE não é uma escolha, é uma necessidade quando STUFF não pode ser utilizado. MAP_REDUCE é útil quando o texto é grande demais para caber na janela de contexto do modelo.
