import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# FERRAMENTAS (tools)
# ═══════════════════════════════════════════════════════════════


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a simple mathematical expression and return the result as a string.
    Use this for any mathematical calculation.
    """
    try:
        result = eval(expression)  # cuidado: apenas para exemplo didático
    except Exception as e:
        return f"Error: {e}"
    return str(result)


@tool
def web_search_mock(query: str) -> str:
    """
    Return the capital of a given country if it exists in the mock data.
    Use this when asked about country capitals.
    """
    data = {
        "Brazil": "Brasília",
        "France": "Paris",
        "Germany": "Berlin",
        "Italy": "Rome",
        "Spain": "Madrid",
        "United States": "Washington, D.C.",
    }
    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"The capital of {country} is {capital}."
    return "I don't know the capital of that country."


# ═══════════════════════════════════════════════════════════════
# MODELO (LLM)
# ═══════════════════════════════════════════════════════════════

llm = ChatOpenAI(
    model="deepseek-v4-flash",  ## deepseek-v4-flash ||  deepseek-v4-pro
    api_key=lambda: os.getenv("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com/v1",
    temperature=0.5,
)

# ═══════════════════════════════════════════════════════════════
# AGENTE
# ═══════════════════════════════════════════════════════════════

agent = create_agent(
    model=llm,
    tools=[calculator, web_search_mock],
    system_prompt=(
        "You are a precise assistant. Follow these rules STRICTLY:\n"
        "Return the tool's output EXACTLY as provided, with no additional text.\n"
        "When a tool returns a result, repeat it VERBATIM as your final answer. "
        "Do NOT rephrase, summarize, or add any extra words.\n"
        "Do NOT say 'According to the tool...' or 'The result is...'.\n"
        "Just return the raw output.\n"
        "Only use information from the tools, never guess, even if you think you know the answer. Dont say anything else about the answer\n"
        "If a tool says 'I don't know', tell the user you don't know.\n"
        "Never search the internet. Only use the tools provided.\n"
        "Use calculator for ANY math, even simple arithmetic.\n"
        "Be concise in your final answers. In mathematical calculations, only provide the final result, without any explanation or steps.\n"
    ),
)

# ═══════════════════════════════════════════════════════════════
# EXECUÇÃO
# ═══════════════════════════════════════════════════════════════

sep = "-" * 15

# Primeira pergunta: capital de país
print(f"\n{sep} Teste 1: Capital de país {sep}")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
)
print(f"\n{sep} Pensamento {sep}")
for i, msg in enumerate(result["messages"]):
    print(f"[{i}] {msg.type}: {msg.content[:80]}...")

print(result["messages"][-1].content)
print(f"\n{sep*2}")

# Segunda pergunta: cálculo matemático
print(f"\n{sep} Teste 2: Cálculo matemático {sep}")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "How much is 10 * 10?"}]}
)
print(f"\n{sep} Pensamento {sep}")
for i, msg in enumerate(result["messages"]):
    print(f"[{i}] {msg.type}: {msg.content[:80]}...")

print(result["messages"][-1].content)
