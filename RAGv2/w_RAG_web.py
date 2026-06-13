from w_write_db_web import store_data
from w_get_result import get_result
from w_delete_chroma import delete_chroma
import os
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"


def web_rag(input="none",query="What is eCadstar?",question="What is eCadstar?",CHROMA_PATH = "./chroma_web",num_results=2,searcher="duckDuckgo",isLink=False,model="deepseek-r1:32b"):
    if(input=="store_data"):
        result=store_data(query=query,num_results=num_results,searcher=searcher,isLink=isLink,CHROMA_PATH=CHROMA_PATH)
        #print(result)
        return result
    elif(input=="get_result"):
        result=get_result(CHROMA_PATH=CHROMA_PATH,model=model,question=question)
        #print(result)
        return result
    elif(input=="delete_chroma"):
        delete_chroma(CHROMA_PATH = CHROMA_PATH)
    else:
        {}

def hello():
    return hello