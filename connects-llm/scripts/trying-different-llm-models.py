from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq


_ = load_dotenv(find_dotenv())

# Los siguientes modelos pueden ir cambiando o deprecando
llamaChatModel = ChatGroq(
    model="llama-3.3-70b-versatile"
)

mistralChatModel = ChatGroq(
    model="qwen/qwen3-32b"
)

messages = [
    ("system", "You are an historian expert in the Kennedy family."),
    ("human", "How many members of the family died tragically?"),
]

print("\n----------\n")

print("How many members of the family died tragically? - LLama3 Response:")

print("\n----------\n")

llamaResponse = llamaChatModel.invoke(messages)

print(llamaResponse.content)

print("\n----------\n")

print("How many members of the family died tragically? - Mistral Response:")

print("\n----------\n")

mistralResponse = mistralChatModel.invoke(messages)

print(mistralResponse.content)
