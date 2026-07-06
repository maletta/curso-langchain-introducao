## Me permite pegar qualquer coisa e fazer com que vire um runnable
## nem sempre dá pra transformar uma função em um runnable, pois a função pode estar sendo usada em outro contexto do negócio, mas é possível criar um runnable que encapsula a função e a transforma em um runnable

from langchain_core.runnables import RunnableLambda


def parse_number(text: str) -> int:
    return int(text.strip())


parse_runnable = RunnableLambda(parse_number)


print(parse_runnable.invoke("  42  "))  # Saída: 42
