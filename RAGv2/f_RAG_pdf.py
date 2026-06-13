
from f_populate_database import write_chroma, clear_database, clear_data
from f_query_data import get_result

def pdf_rag(input="none",question="How to external program in an unlocked state?",model="deepseek-r1:14b",CHROMA_PATH = "chroma",DATA_PATH = "data"):
    if(input=="delete_chroma"):
        clear_database(CHROMA_PATH)
    elif(input=="delete_data"):
        clear_data(DATA_PATH)
    elif(input=="store_data"):
        write_chroma(CHROMA_PATH,DATA_PATH)
    elif(input=="get_result"):
        result=get_result(query_text=question,model=model)
        #print(result)
        return result
    else:
        {}

def hello():
    return "Hi how are you"