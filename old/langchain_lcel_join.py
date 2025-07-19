from operator import itemgetter
import os
import dotenv as env
from langchain_openai import ChatOpenAI

from langchain.prompts import PromptTemplate, ChatPromptTemplate

from langchain.globals import set_debug
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


class Destino(BaseModel):
    city: str = Field(description="Cidade recomendada")
    motivo: str = Field(description="Motivo de recomendação da cidade")


env.load_dotenv()
set_debug(True)

API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=API_KEY, model="gpt-4.1-nano", temperature=0.6)


str_parser = StrOutputParser()
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
city_agent = city_prompt_template | llm | destino_parser


# Prompt para os restaurantes
restaurants_prompt_template = PromptTemplate(
    template="Sugira 3 restaurantes populares entre locais em {city}"
)
restaurant_agent = restaurants_prompt_template | llm | str_parser

# Prompt para as atividades culturais
culture_prompt_template = PromptTemplate(
    template="Sugira 3 atividades e locais culturais imperdíveis em {city}"
)
culture_agent = culture_prompt_template | llm | str_parser

resume_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("ai", "Faça um plano de viagem para a cidade sugerida: {city}"),
        ("ai", "Restaurantes que você não pode perder: {restaurants}"),
        ("ai", "Atividades e locais culturais: {cultural_places}"),
        (
            "system",
            "Combine as informações anteriores em 2 paragrafos coerentes.",
        ),
    ]
)
resume_agent = resume_prompt_template | llm | str_parser


chain = (
    city_agent
    | {
        "city": itemgetter("city"),
        "restaurants": restaurant_agent,
        "cultural_places": culture_agent,
    }
    | resume_agent
)

result = chain.invoke({"intress": "Musica Reggae"})
print(result)
