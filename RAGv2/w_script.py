import sys
import time
from w_RAG_web import web_rag, hello

CHROMA_PATH = "chroma_web"

def process_input(input,query, isLink, question,model,search,num_result):
    start_time = time.time() 
    value=web_rag(input=input,query=query,question=question,CHROMA_PATH = CHROMA_PATH,num_results=num_result,searcher=search,isLink=isLink,model=model)
    #value=hello()
    end_time = time.time()  # End the timer
    execution_time = end_time - start_time
    result = f"{value},Execution Time: {execution_time:.6f} seconds"
    return result

def fun():
    return hello()

if __name__ == "__main__":
    input = sys.argv[1]  # Get text input
    query = sys.argv[2]  # Get dropdown value
    isLink = sys.argv[3]=="true"  # Get dropdown value
    question = sys.argv[4]  # Get file paths
    model = sys.argv[5]  # Get file paths
    search = sys.argv[6]  # Get file paths
    num_result = int(sys.argv[7])  # Get file paths

    output = process_input( input, query,isLink,question,model,search,num_result)
    #output=fun()
    print(output)  # Send output back to Flask
