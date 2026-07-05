# LangChain Course — Notes & Code

Curso de LangChain com Python, utilizando DeepSeek como provedor de LLM.

---

## 00 — Introdução ao Ecossistema LangChain

### O que é LangChain?

Framework open-source (Python/JavaScript) criado em 2022 para simplificar a integração com LLMs e serviços auxiliares: carregamento de dados, memória, busca por documentos, agentes, roteadores, etc.

### Ecossistema de Serviços

| Serviço | Propósito |
|---|---|
| **LangSmith** | Observabilidade, debugging e monitoramento de aplicações em produção. Gerencia custos, latência e performance dos modelos. |
| **LangServe** | Plataforma hosted para deploy de aplicações LangChain como APIs REST, com infra gerenciada e escalabilidade. |
| **LangGraph Platform** | Framework para orquestração de agentes com grafos determinísticos. Suporta agentes autônomos de longa duração com memória. |
| **LangGraph Studio** | IDE visual para criação e debugging de grafos do LangGraph. |
| **LangChain Hub** | Catálogo para publicar, versionar, testar e baixar prompts e artefatos. Inclui playground e SDK para push/pull. |

### Separação em Pacotes

- `langchain-core` — Abstrações base (Runnable, prompts, parsers)
- `langchain` — Implementações principais e chains
- `langchain-community` — Integrações da comunidade
- `langchain-openai`, `langchain-deepseek`, `langchain-google-genai` — Pacotes de provedores
- Pacotes de terceiros (Pinecone, Weaviate, FAISS, etc.)

### Principais Recursos

- **Chains**: Fluxos de execução compostos por etapas encadeáveis (Runnables)
- **LCEL (LangChain Expression Language)**: Operador `|` (pipe) para compor Runnables
- **Carregamento de Documentos**: Loaders para CSV, JSON, HTML, Markdown, PDF, sites
- **Embeddings e Bancos Vetoriais**: Transformar conteúdo em vetores para busca semântica (Pinecone, PGVector, Weaviate, FAISS)
- **Agentes e Ferramentas**: LLM decide quais ações/ferramentas executar
- **Memória e Histórico**: Persistência de contexto entre interações
- **Prompt Templates**: Templates com placeholders substituíveis
- **Output Parsing**: Extração de saída estruturada com Pydantic
- **Sumarização**: Estratégia Map-Reduce para lidar com textos longos

---

## 01 — Fundamentos

### Ambiente Python

```bash
# Criar ambiente virtual (usa o Python ativo no momento)
python -m venv .venv

# Ativar (apenas para o terminal atual)
source .venv/bin/activate

# Sair do ambiente
deactivate

# Instalar dependências
pip install langchain langchain-openai langchain-google-genai langchain-deepseek python-dotenv beautifulsoup4 pypdf

# Salvar dependências
pip freeze > requirements.txt
```

> **Convenção:** Use `.venv` (com ponto) — é o padrão moderno, não polui o `ls` e ferramentas como VS Code detectam automaticamente.

### Gerenciamento de Versão Python com pyenv

```bash
# Listar versões disponíveis
pyenv install --list | grep "3.13"

# Instalar uma versão
pyenv install 3.13.14

# Definir versão para o projeto (cria arquivo .python-version)
pyenv local 3.13.14

# Definir versão global (fallback)
pyenv global 3.13.14

# Ver versão ativa
pyenv versions
```

> Após mudar a versão do pyenv, **recrie o `.venv`** — ele é uma "fotografia" do Python ativo no momento da criação.

### `.env` e Variáveis de Ambiente

```bash
# .env
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=sk-...
```

```python
from dotenv import load_dotenv
load_dotenv()  # Carrega .env para os.environ
```

> Use `python-dotenv` no código, não a injeção do VS Code (`python.terminal.useEnvFile`). É portátil e funciona em qualquer terminal.

### Hello World — OpenAI/DeepSeek

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="deepseek-v4-pro",       # ou deepseek-v4-flash
    api_key=lambda: os.getenv("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com/v1",
    temperature=0.5,
)

message = model.invoke("Hello world")
print(message.content)
```

> `lambda` no `api_key` = lazy evaluation. A chave só é lida quando a API é chamada, não no import.

### Usando `langchain-deepseek` (pacote específico)

```python
from langchain_deepseek import ChatDeepSeek

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0.5,
)
# base_url e api_key já são configurados automaticamente!
```

### Listar Modelos Disponíveis

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

for model in client.models.list().data:
    print(f"  - {model.id}")
```

> A DeepSeek não tem SDK oficial próprio — o cliente `openai` funciona com `base_url` apontado para DeepSeek.

### `init_chat_model` (atalho multi-provedor)

```python
from langchain.chat_models import init_chat_model

gemini = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
answer = gemini.invoke("Hello world")
```

### Prompt Templates

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["name"],
    template="Hello i'm {name}. Tell me a joke with my name",
)

text = template.format(name="Maletta")  # Preenche o placeholder
```

> **Nota:** Importe de `langchain_core.prompts`, NÃO de `langchain.prompts` (módulo antigo, removido).

### Chat Prompt Templates (mensagens com papéis)

```python
from langchain_core.prompts import ChatPromptTemplate

# Tupla: (role, content_template)
system = ("system", "you are an assistant that answers questions in a {style} style")
user = ("user", "{question}")

chat_prompt = ChatPromptTemplate([system, user])
messages = chat_prompt.format_messages(style="funny", question="who is alan turing?")

for msg in messages:
    print(f"{msg.type}: {msg.content}")
```

> Tuplas `("role", "template")` são um atalho do LangChain para definir mensagens de chat.

---

## 02 — Chains e Processamento

### LCEL: LangChain Expression Language

O operador `|` (pipe) encadeia componentes Runnable. A saída do componente à esquerda vira a entrada do componente à direita.

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

question_template = PromptTemplate(
    input_variables=["name"],
    template="Hello i'm {name}. Tell me a joke with my name",
)

model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=lambda: os.getenv("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com/v1",
    temperature=0.5,
)

# Chain: prompt → model
chain = question_template | model

result = chain.invoke({"name": "Maletta"})
print(result.content)
```

### Como o pipe funciona internamente

```
chain.invoke({"name": "Maletta"})
        │
        ▼
┌─────────────────────────────────────────┐
│ 1. question_template.invoke({"name"})   │ → "Hello i'm Maletta. Tell me a joke..."
│ 2. model.invoke("Hello i'm Maletta...") │ → AIMessage("Why did Maletta...")
│ 3. Retorna AIMessage                    │
└─────────────────────────────────────────┘
```

- `|` chama `__or__` que cria um `RunnableSequence`
- `RunnableSequence.invoke()` chama `.invoke()` em cada componente em sequência
- A saída de um é passada como entrada do próximo
- **Tipos precisam ser compatíveis:** output do primeiro deve bater com input do segundo

### Contrato Runnable

Todo componente LangChain herda de `Runnable` (classe do `langchain-core`, não é padrão Python). O contrato define:

| Método | Propósito |
|---|---|
| `.invoke(input)` | Execução síncrona |
| `.ainvoke(input)` | Execução assíncrona (async/await) |
| `.stream(input)` | Streaming (iterator) |
| `.batch(inputs)` | Processamento em lote |
| `.__or__(other)` | Operador `\|` (pipe) |

---

## Conceitos Fundamentais de Python Aprendidos

| Conceito | Explicação |
|---|---|
| **Virtual Environment (`.venv`)** | Isola dependências por projeto. Equivalente ao `node_modules` do Node. |
| **`self`** | Equivalente ao `this` do JS, mas explícito como parâmetro. Não é palavra reservada — é convenção. |
| **`lambda`** | Função anônima de uma linha. Equivalente à arrow function `() =>` do JS. |
| **`:=` (walrus operator)** | Atribui e retorna valor na mesma expressão. Permite atribuição dentro de `if`/`while`. |
| **f-string** | Interpolação de string: `f"texto {variavel}"`. Equivalente a `` `texto ${var}` `` no JS. |
| **Tupla `(a, b)`** | Sequência imutável. LangChain usa tuplas `(role, content)` como atalho para mensagens. |
| **Expression vs Statement** | Expression produz valor (pode ir à direita de `=`). Statement executa ação (não pode). |
| **`import` vs `import_module()`** | `import` é statement. `importlib.import_module()` é expression que retorna o módulo. |
| **`openai.OpenAI` sub-recursos** | `client.models` é instância de outra classe (`openai.resources.models.Models`). Padrão API Resource. |

---

## Comandos Úteis

```bash
# Ambiente
python -m venv .venv          # Criar venv
source .venv/bin/activate      # Ativar venv
deactivate                     # Sair do venv
pip install <pkg>              # Instalar pacote
pip freeze > requirements.txt  # Salvar dependências

# pyenv
pyenv versions                 # Listar versões instaladas
pyenv install 3.13.14          # Instalar versão
pyenv local 3.13.14            # Definir versão do projeto
pyenv global 3.13.14           # Definir versão global

# Git
git init                       # Inicializar repositório
```

---

## Extensões VS Code Recomendadas

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.debugpy
code --install-extension ms-python.black-formatter
code --install-extension charliermarsh.ruff
code --install-extension continue.continue  # AI code completion com DeepSeek
```
