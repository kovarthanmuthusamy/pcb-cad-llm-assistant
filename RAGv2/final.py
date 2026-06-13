from w_RAG_web import web_rag
from f_RAG_pdf import pdf_rag
from f_query_data_default import get_result_default
from chooser import get_llm_response
import os
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"


def RAG(question="What is eCadstar?",num_results=2,searcher="duckDuckgo",isLink=False,model="deepseek-r1:32b",CHROMA_PATH = "./chroma"):
    search=get_llm_response(question)
    if search.isdigit():  # Handles positive & negative integers
        search=int(search)
    else:
        search=3  
    print(search)
    if(search==3):
        CHROMA_PATH = "./chroma_web"
       # web_rag(input="delete_chroma",query=question,question=question,CHROMA_PATH = CHROMA_PATH,num_results=num_results,searcher=searcher,isLink=isLink,model=model)
        web_rag(input="store_data",query=question,question=question,CHROMA_PATH = CHROMA_PATH,num_results=num_results,searcher=searcher,isLink=isLink,model=model)
        result=web_rag(input="get_result",query=question,question=question,CHROMA_PATH = CHROMA_PATH,num_results=num_results,searcher=searcher,isLink=isLink,model=model)
        return result
    elif(search==2):
        CHROMA_PATH = "./chroma"
        DATA_PATH = "data_pdf"
        pdf_rag(input="store_data",CHROMA_PATH = CHROMA_PATH,DATA_PATH = DATA_PATH)
        result=pdf_rag(input="get_result",question=question,model=model,CHROMA_PATH = CHROMA_PATH,DATA_PATH =DATA_PATH)
        return result
    elif(search==4):
        pdf_rag(input="delete_chroma",CHROMA_PATH = CHROMA_PATH)
        pdf_rag(input="delete_data",DATA_PATH = DATA_PATH)
    elif(search==1):
        result=get_result_default(query_text=question,model=model,CHROMA_PATH="chroma_def")
        return result
    else:
        return None