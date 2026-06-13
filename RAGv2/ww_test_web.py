from w_RAG_web import web_rag
import os
#from write_db_web import store_data
from w_get_result import get_result
from w_delete_chroma import delete_chroma
import time
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"
#web_rag(input="none",query="What is eCadstar?",question="What is eCadstar?",CHROMA_PATH = "./chroma_db",num_results=2,searcher="google",isLink=False,model="deepseek-r1:32b")
#store_data(query="eCadstar",num_results=2,searcher="google",isLink=False,CHROMA_PATH=CHROMA_PATH)
#get_result(CHROMA_PATH=CHROMA_PATH,model="deepseek-r1:32b",question="What is eCadstar?")
#delete_chroma(CHROMA_PATH = CHROMA_PATH)
input="store_data"
query="what is the World's fastest car"
question="what is the World's fastest car"
CHROMA_PATH = "./chroma_web"
num_results=2
#["google","duckDuckGo","brave","bing","yahoo","startpage"]
searcher="google"
isLink=False
#["mistral:7b", "deepseek-r1:14b", "deepseek-r1:1.5b","deepseek-r1:7b","deepseek-r1:32b","gemma2:latest","llama3.1:latest"] 
model="gemma2:latest"

#web_rag(input="store_data",query=query,question=question,CHROMA_PATH = CHROMA_PATH,num_results=num_results,searcher=searcher,isLink=isLink,model=model)
print(web_rag(input="get_result",query=query,question=question,CHROMA_PATH = CHROMA_PATH,num_results=num_results,searcher=searcher,isLink=isLink,model=model))
#time.sleep(5)
#web_rag(input="delete_chroma",query=query,question=question,CHROMA_PATH = CHROMA_PATH,num_results=num_results,searcher=searcher,isLink=isLink,model=model)


#import time
#
#file_path = 'data_level0.bin'
#for attempt in range(3):
#    try:
#        # Attempt file operation
#        with open(file_path, 'r') as file:
#            data = file.read()
#        break  # Exit loop if successful
#    except PermissionError:
#        time.sleep(1)  # Wait for 1 second before retrying
#else:
#    print(f"Failed to access {file_path} after multiple attempts.")