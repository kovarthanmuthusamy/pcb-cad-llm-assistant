from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain import hub
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
load_dotenv()
#os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGCHAIN_API_KEY"
os.getenv("LANGCHAIN_API_KEY")
os.getenv("USER_AGENT")
#os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"


def get_result(CHROMA_PATH="./chroma_web",model="deepseek-r1:32b",question="What is eCadstar?"):
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=GPT4AllEmbeddings())
    llm = ChatOllama(model=model)
    prompt = hub.pull("rlm/rag-prompt")
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    question = question
    result = qa_chain.invoke({"query": question })
    return result
