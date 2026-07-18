import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langsmith import Client

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
        result = eval(expression)
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
# PROMPT DO LANGCHAIN HUB (hwchase17/react)
# ═══════════════════════════════════════════════════════════════

# Puxa o prompt público do LangChain Hub
client = Client()
react_prompt = client.pull_prompt(
    "hwchase17/react", dangerously_pull_public_prompt=True
)

print("═" * 60)
print("PROMPT DO LANGCHAIN HUB (hwchase17/react)")
print("═" * 60)
print(react_prompt)
print("═" * 60)
print()

# ═══════════════════════════════════════════════════════════════
# NOTA IMPORTANTE
# ═══════════════════════════════════════════════════════════════
#
# O prompt "hwchase17/react" é do ECOSSISTEMA ANTIGO do LangChain.
# Ele foi projetado para o initialize_agent() com AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# onde a LLM gerava TEXTO no formato Thought:/Action:/Action Input:/Observation:
# e o AgentExecutor parseava esse texto.
#
# No ecossistema MODERNO (create_agent), o agente usa TOOL CALLING nativo
# (JSON estruturado), não parsing de texto. Portanto, o prompt do Hub NÃO
# é usado como system_prompt direto — ele serve como referência histórica
# e didática para entender o padrão ReAct.
#
# O create_agent já implementa o ciclo ReAct internamente via tool calling.
# O system_prompt no agente moderno define APENAS o comportamento,
# não o formato de saída (Thought/Action/Observation).
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# AGENTE (com system_prompt inspirado no ReAct do Hub)
# ═══════════════════════════════════════════════════════════════

agent = create_agent(
    model=llm,
    tools=[calculator, web_search_mock],
    system_prompt=(
        "Answer the following questions as best you can. "
        "You have access to tools — use them when needed.\n\n"
        "RULES:\n"
        "- Think step by step about what you need to do.\n"
        "- Use the tools to get information — never guess.\n"
        "- If a tool returns an answer, use it directly.\n"
        "- If a tool says it doesn't know, tell the user.\n"
        "- When you have enough information, provide the final answer."
        # Nota: NÃO incluímos {tools}, {tool_names}, {input}, {agent_scratchpad}
        # porque create_agent gerencia isso automaticamente via tool calling.
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
print(f"\n{sep} Histórico de mensagens {sep}")
for i, msg in enumerate(result["messages"]):
    print(f"[{i}] {msg.type}: {str(msg.content)[:100]}...")
print(f"\n✅ Resposta final: {result['messages'][-1].content}")
print(f"\n{sep*2}")

# Segunda pergunta: cálculo matemático
print(f"\n{sep} Teste 2: Cálculo matemático {sep}")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "How much is 10 * 10?"}]}
)
print(f"\n{sep} Histórico de mensagens {sep}")
for i, msg in enumerate(result["messages"]):
    print(f"[{i}] {msg.type}: {str(msg.content)[:100]}...")
print(f"\n✅ Resposta final: {result['messages'][-1].content}")
