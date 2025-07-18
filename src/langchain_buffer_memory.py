import os
import dotenv as env
from langchain_openai import ChatOpenAI

from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
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


PROMPT_TEMPLATE = """
A seguir, uma conversa amigável entre um humano e uma IA. A IA é falante e fornece muitos detalhes específicos de seu contexto. Se a IA não souber a resposta para uma pergunta, ela diz honestamente que não sabe.

Conversa atual:
{history}
Humano: {input}
IA:
"""
system_template = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["input", "history"]
)

memory_buffer = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, memory=memory_buffer, verbose=True, prompt=system_template
)

for message in messages:
    response = conversation.predict(input=message)
    print(response)
