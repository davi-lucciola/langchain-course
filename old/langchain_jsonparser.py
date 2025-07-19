import os
import dotenv as env
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain

from langchain.prompts import PromptTemplate
from langchain.globals import set_debug
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser


class Destino(BaseModel):
    city: str = Field(description="Cidade recomendada")
    motivo: str = Field(description="Motivo de recomendação da cidade")


env.load_dotenv()
set_debug(True)

API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=API_KEY, model="gpt-4.1-nano", temperature=0.6)


destino_parser = JsonOutputParser(pydantic_object=Destino)

# Prompt para a sugestão da cidade
city_prompt_template_message = """
Sugira uma cidade dado o meu interesse em: {intress}. 
{output_format}
"""
city_prompt_template = PromptTemplate(
    template=city_prompt_template_message,
    input_variables=["intress"],
    partial_variables={"output_format": destino_parser.get_format_instructions()},
)

# Prompt para os restaurantes
restaurants_prompt_template = PromptTemplate(
    template="Sugira 3 restaurantes populares entre locais em {city}"
)

# Prompt para as atividades culturais
culture_prompt_template = PromptTemplate(
    template="Sugira 3 atividades e locais culturais imperdíveis em {city}"
)

# --- Resto do código permanece o mesmo ---
city_chain = LLMChain(prompt=city_prompt_template, llm=llm)
restaurants_chain = LLMChain(prompt=restaurants_prompt_template, llm=llm)
culture_chain = LLMChain(prompt=culture_prompt_template, llm=llm)


chain = SimpleSequentialChain(
    # chains=[city_chain, restaurants_chain, culture_chain], verbose=True
    chains=[city_chain],
    verbose=True,
)

result = chain.invoke(input="Praias")
print(result)
