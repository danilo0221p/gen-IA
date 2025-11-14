import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from pydantic import BaseModel, Field
# from langchain_core.pydantic_v1 import BaseModel, Field --> es reemplazada por from pydantic import BaseModel, Field ya que tenemos librerías mas reciente
# from langchain.output_parsers.json import SimpleJsonOutputParser--> es reemplazada por from langchain_core.output_parsers import JsonOutputParser ya que tenemos librerías mas reciente



_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

llmModel = OpenAI()

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")


json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)

json_parser = JsonOutputParser()

json_chain = json_prompt | llmModel | json_parser

"""
La linea anterior es equivalente a:
def json_chain_func(input_dict):
    # 1) Renderizar el prompt
    prompt_str = json_prompt.format(**input_dict)
    
    # 2) Llamar al LLM
    llm_output = llmModel.invoke(prompt_str)
    
    # 3) Parsear el JSON
    parsed = json_parser.parse(llm_output)
    return parsed
"""

response = json_chain.invoke({"question": "What is the biggest country?"})

print("What is the biggest country?")
print(response)

print("\n----------\n")


# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


# Set up a parser
parser = JsonOutputParser(pydantic_object=Joke)

# Inject parser instructions into the prompt template.
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Create a chain with the prompt and the parser
chain = prompt | chatModel | parser

response = chain.invoke({"query": "Tell me a joke."})

print("Tell me a joke in custom format defined by Pydantic:")
print(response)

print("\n----------\n")
