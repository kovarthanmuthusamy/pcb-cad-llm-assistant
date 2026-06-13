import time


start_time = time.time()  # Start the timer

from f_RAG_pdf import pdf_rag

CHROMA_PATH = "chroma"
DATA_PATH = "data"
model="mistral:7b"

#pdf_rag(input="delete_chroma",CHROMA_PATH = CHROMA_PATH)
#pdf_rag(input="delete_data",DATA_PATH = DATA_PATH)
#pdf_rag(input="store_data",CHROMA_PATH = CHROMA_PATH,DATA_PATH = DATA_PATH)
#pdf_rag(input="get_result",question="How to external program in an unlocked state?",model=model,CHROMA_PATH = CHROMA_PATH,DATA_PATH =DATA_PATH)


end_time = time.time()  # End the timer
execution_time = end_time - start_time

print(f"Execution Time: {execution_time:.6f} seconds")