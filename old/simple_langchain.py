import os
import dotenv as env
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

env.load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=API_KEY, model="gpt-4.1-nano")

num_days = 7
num_childs = 0
activity = "Praia"

template = PromptTemplate.from_template(
    "Crie um roteiro de viagem de {num_days} dias, para uma familia com {num_childs} crianças, que gostam de {activity}"
)

prompt = template.format(
    num_days=num_days, 
    num_childs=num_childs, 
    activity=activity
)

print("User:")
print(f"-> {prompt}")
print()

response = llm.invoke(prompt)

print("System:")
print(response.content)
