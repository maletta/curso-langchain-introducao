# Agentes de IA

## O que é um Agente?

Um agente é um software que usa uma LLM como "cérebro" para **decidir o que fazer** de forma autônoma.

Diferente de uma chain tradicional (onde o caminho é fixo: prompt → modelo → parser), um agente pode:
- Escolher qual ferramenta usar
- Decidir em qual ordem executar as ações
- Avaliar se a resposta está pronta ou se precisa de mais informações
- Repetir o ciclo até chegar a uma resposta satisfatória

## Chain vs Agente

| Chain | Agente |
|---|---|
| Caminho fixo e previsível | Caminho dinâmico, decidido em runtime |
| "Faça isso, depois isso, depois aquilo" | "Descubra o que precisa fazer e faça" |
| Ex.: traduzir → resumir → responder | Ex.: "Quantos usuários ativos temos? Consulte o banco, analise, compare com o mês passado" |

## Do que um Agente precisa?

### 1. Informação (Contexto)
O agente precisa entender o ambiente em que está atuando:
- Qual é o objetivo da tarefa?
- O que o usuário pediu?
- O que já foi feito até agora?
- Quais foram os resultados das ações anteriores?

### 2. Ferramentas (Tools)
APIs, funções ou serviços que o agente pode chamar para adicionar capacidades além da LLM:
- Busca na internet (Tavily, SerpAPI)
- Consulta a banco de dados (SQL)
- Calculadora (para operações matemáticas precisas)
- Leitura de arquivos (PDF, CSV, etc.)
- Chamadas a APIs externas (clima, cotação, etc.)
- Execução de código (Python REPL)

A LLM **não executa** as ferramentas — ela **decide** qual ferramenta chamar e com quais parâmetros. A execução é feita pelo runtime do agente.

### 3. Prompts (Instruções de Comportamento)
O prompt define:
- O papel do agente ("Você é um assistente financeiro...")
- Como ele deve decidir qual ferramenta usar
- Quando ele deve parar e dar a resposta final
- Restrições e regras de comportamento

### 4. Memória (Histórico)
Para manter contexto entre ações, o agente precisa lembrar:
- O que já foi feito (ações executadas)
- O que foi observado (resultados das ferramentas)
- O raciocínio anterior (thoughts)

## O Ciclo ReAct (Reasoning + Acting)

É o padrão mais comum para agentes. A cada passo, o agente:

```
Observation → Thought → Action → Observation → Thought → Action → ... → Final Answer
     │            │         │
     │            │         └─ Executa uma ferramenta
     │            └─ Raciocina sobre o que fazer
     └─ Observa o resultado da ação anterior
```

### Exemplo concreto:

```
User: "Qual a população do Brasil e qual a raiz quadrada desse número?"

Thought: Preciso descobrir a população do Brasil primeiro.
Action: buscar_no_google("população do Brasil 2024")
Observation: "A população do Brasil é aproximadamente 215 milhões."

Thought: Agora preciso calcular a raiz quadrada de 215.000.000.
Action: calcular("sqrt(215000000)")
Observation: "14662.75"

Final Answer: "A população do Brasil é ~215 milhões. A raiz quadrada é ~14.663."
```

## Por que Agentes são importantes?

- **Autonomia**: Resolvem problemas complexos sem intervenção humana a cada passo
- **Flexibilidade**: Mesmo agente pode lidar com cenários diferentes usando ferramentas diferentes
- **Escalabilidade**: Um agente pode orquestrar múltiplas ferramentas e APIs

## Frameworks

- **LangGraph**: Framework LangChain para agentes com grafos determinísticos
- **CrewAI**: Múltiplos agentes colaborando
- **AutoGPT**: Agentes autônomos com loops longos
- **OpenAI Assistants**: Agentes gerenciados pela OpenAI
