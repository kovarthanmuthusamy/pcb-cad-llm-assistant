import sys
import time
from f_RAG_pdf import pdf_rag, hello

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def process_input(user_text,selected_input, selected_option, file_paths):
    start_time = time.time() 
    test=pdf_rag(input=selected_input,question=user_text,model=selected_option,CHROMA_PATH = CHROMA_PATH,DATA_PATH =DATA_PATH)
    end_time = time.time()  # End the timer
    execution_time = end_time - start_time
    result = f"{test},Execution Time: {execution_time:.6f} seconds"
    return result

if __name__ == "__main__":
    user_text = sys.argv[1]  # Get text input
    selected_input = sys.argv[2]  # Get dropdown value
    selected_option = sys.argv[3]  # Get dropdown value
    file_paths = sys.argv[4:]  # Get file paths
    output = process_input(user_text,selected_input, selected_option, file_paths)
    print(output)  # Send output back to Flask
