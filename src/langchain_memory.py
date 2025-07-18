import os
import dotenv as env
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.globals import set_debug


env.load_dotenv()
set_debug(True)

API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=API_KEY, model="gpt-4.1-nano", temperature=0.6)


messages = [
    "Quero visitar um lugar famoso no Brasil por suas praias e cultura",
    "Qual é o melhor periodo do ano para visitar em termos de clima?",
    "Quais tipos de atividades ao ar livre estão disponiveis?",
    "Alguma sugestão de acomodação eco-friendly por lá?",
    "Cite outras 20 cidades com caracteristicas semelhantes as que descrevemos até agora. Rankeie por mais interessante, incluindo no meio ai a que você já sugeriu.",
    "Na primeira cidade que você sugeriu lá atrás, quero saber 5 restaurantes para visitar. Responda somente o nome da cidade e o nome dos restaurantes.",
]

chat = ""
for message in messages:
    chat += f"User: {message}\n"

    prompt = PromptTemplate(template=chat, input_variables=[""])
    chain = prompt | llm | StrOutputParser()

    chat += f"AI: {chain.invoke({})}\n"
    print(chat)
