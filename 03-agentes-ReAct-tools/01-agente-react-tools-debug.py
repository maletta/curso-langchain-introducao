import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
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
# MIDDLEWARE DE DEBUG — revela o ciclo ReAct completo
# ═══════════════════════════════════════════════════════════════


class DebugMiddleware(AgentMiddleware):
    """
    Middleware que expõe o ciclo de vida completo do agente:

    - Quantas iterações (chamadas à LLM) ocorreram
    - O que foi enviado à LLM a cada iteração (prompt + histórico acumulado)
    - O que a LLM decidiu fazer (chamar tool ou responder)
    - Quais tools foram executadas e o que retornaram
    """

    def __init__(self):
        self.llm_call_count = 0
        self.tool_call_count = 0

    # ── ANTES de cada chamada à LLM ──────────────────────────

    def before_model(self, state, runtime):
        """Loga o estado do prompt ANTES de enviar para a LLM."""
        self.llm_call_count += 1
        messages = state.get("messages", [])

        print(f"\n{'='*60}")
        print(f"🔄 ITERAÇÃO {self.llm_call_count} — Enviando prompt para LLM")
        print(f"   Total de mensagens acumuladas: {len(messages)}")
        print(f"   {'─'*50}")

        for i, msg in enumerate(messages):
            msg_type = msg.type
            msg_preview = str(msg.content)[:100].replace("\n", "\\n")
            # Destaca mensagens da tool
            if msg_type == "tool":
                print(f"   [{i}] 📋 {msg_type}: {msg_preview}...")
            elif msg_type == "ai":
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    print(f"   [{i}] 🤔 {msg_type}: [decidiu chamar tool(s)]")
                else:
                    print(f"   [{i}] 💬 {msg_type}: {msg_preview}...")
            else:
                print(f"   [{i}] 👤 {msg_type}: {msg_preview}...")

        print(f"   {'─'*50}")
        print(f"   📤 Enviando {len(messages)} mensagens para LLM...")
        return None

    # ── DEPOIS de cada chamada à LLM ─────────────────────────

    def wrap_model_call(self, request, handler):
        """Intercepta a resposta da LLM e loga o que ela decidiu."""
        result = handler(request)  # ← chama a LLM de fato

        tool_calls = getattr(result, "tool_calls", None)
        content = getattr(result, "content", None)

        if tool_calls:
            for tc in tool_calls:
                print(f"   🔧 LLM decidiu USAR TOOL: {tc['name']}")
                print(f"      Argumentos: {tc['args']}")
        elif content:
            print(f"   ✅ LLM decidiu RESPONDER (FINAL):")
            print(f"      {str(content)[:150]}...")

        return result

    # ── ANTES e DEPOIS de cada tool ──────────────────────────

    def wrap_tool_call(self, request, handler):
        """Loga a execução da tool e seu resultado."""
        self.tool_call_count += 1
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call["args"]

        print(f"   ⚙️  [{self.tool_call_count}] Executando tool '{tool_name}'")
        print(f"      Input: {tool_args}")

        result = handler(request)  # ← executa a tool

        output = getattr(result, "content", str(result))
        print(f"      📋 Output: {str(output)[:150]}...")
        return result


# ═══════════════════════════════════════════════════════════════
# AGENTE (mesmo que o arquivo original, só adiciona o middleware)
# ═══════════════════════════════════════════════════════════════

debug = DebugMiddleware()

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
    middleware=[debug],
)

# ═══════════════════════════════════════════════════════════════
# EXECUÇÃO
# ═══════════════════════════════════════════════════════════════

sep = "-" * 15

# Teste 1: capital de país — 1 iteração esperada
print(f"\n{'='*60}")
print(f"🧪 TESTE 1: Capital de país")
print(f"{'='*60}")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
)
print(f"\n{sep} Resumo pós-execução {sep}")
print(f"Chamadas à LLM: {debug.llm_call_count}")
print(f"Tools executadas: {debug.tool_call_count}")
print(f"Resposta final: {result['messages'][-1].content}")

# Reset contadores para o próximo teste
debug.llm_call_count = 0
debug.tool_call_count = 0

# Teste 2: cálculo matemático — 1 iteração esperada
print(f"\n{'='*60}")
print(f"🧪 TESTE 2: Cálculo matemático")
print(f"{'='*60}")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "How much is 10 * 10?"}]}
)
print(f"\n{sep} Resumo pós-execução {sep}")
print(f"Chamadas à LLM: {debug.llm_call_count}")
print(f"Tools executadas: {debug.tool_call_count}")
print(f"Resposta final: {result['messages'][-1].content}")
