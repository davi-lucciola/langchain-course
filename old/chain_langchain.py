import os
import dotenv as env
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain

# Importe as classes de mensagem para construir o ChatPromptTemplate de forma mais completa
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.globals import set_debug

env.load_dotenv()
set_debug(True)

API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=API_KEY, model="gpt-4.1-nano", temperature=0.7)

# --- Mensagem de Sistema Central ---
# Esta mensagem define o papel geral do seu assistente de viagens
system_travel_assistant_template_message = """Você é um assistente de viagens prestativo e experiente.
Seu objetivo é ajudar os usuários a planejar suas viagens, sugerindo destinos, atividades e locais interessantes.
Seja criativo e forneça informações úteis, mas mantenha suas respostas concisas e diretas ao ponto, como solicitado.
"""
system_travel_assistant_template = SystemMessagePromptTemplate.from_template(
    system_travel_assistant_template_message
)


# --- Refatorando os Prompts para Incluir a Mensagem de Sistema ---

# Prompt para a sugestão da cidade
city_prompt_template_message = """
Sugira uma cidade dado o meu interesse em: {intress}. 
Sua saída deve ser SOMENTE o nome da cidade. Cidade:
"""
city_prompt_template = ChatPromptTemplate.from_messages(
    [
        system_travel_assistant_template,  # Inclui a mensagem de sistema aqui
        HumanMessagePromptTemplate.from_template(city_prompt_template_message),
    ]
)

# Prompt para os restaurantes
restaurants_prompt_template = ChatPromptTemplate.from_messages(
    [
        system_travel_assistant_template,  # Inclui a mensagem de sistema aqui
        HumanMessagePromptTemplate.from_template(
            "Sugira 3 restaurantes populares entre locais em {city}"
        ),
    ]
)

# Prompt para as atividades culturais
culture_prompt_template = ChatPromptTemplate.from_messages(
    [
        system_travel_assistant_template,  # Inclui a mensagem de sistema aqui
        HumanMessagePromptTemplate.from_template(
            "Sugira 3 atividades e locais culturais imperdíveis em {city}"
        ),
    ]
)

# --- Resto do código permanece o mesmo ---
city_chain = LLMChain(prompt=city_prompt_template, llm=llm)
restaurants_chain = LLMChain(prompt=restaurants_prompt_template, llm=llm)
culture_chain = LLMChain(prompt=culture_prompt_template, llm=llm)


chain = SimpleSequentialChain(
    chains=[city_chain, restaurants_chain, culture_chain], verbose=True
)

result = chain.invoke(input="Praias")
print(result)
