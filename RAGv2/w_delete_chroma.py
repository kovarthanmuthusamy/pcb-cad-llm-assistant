import shutil
import os

def delete_chroma(CHROMA_PATH = "./chroma_db"):
    # Check if the directory exists
    if os.path.exists(CHROMA_PATH):
        # Remove the directory and all its contents
        shutil.rmtree(CHROMA_PATH)
        print(f"Directory '{CHROMA_PATH}' has been deleted.")
    else:
        print(f"Directory '{CHROMA_PATH}' does not exist.")