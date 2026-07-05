from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["name"],
    template="Hello i'm {name}. Tell me a joke with my name",
)

text = template.format(name="Maletta")
print(text)
