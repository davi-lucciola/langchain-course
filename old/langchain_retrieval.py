import os
import dotenv as env
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.globals import set_debug
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


env.load_dotenv()
set_debug(True)

API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=API_KEY, model="gpt-4.1-nano", temperature=0.7)

file_path = "./old/GTB_gold_Nov23.pdf"
loader = PyPDFLoader(file_path)

documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000)
texts = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
question = "Como devo proceder caso tenha um item comprado roubado?"

meu_resultado = qa_chain.invoke({"query": question})
print(meu_resultado)
