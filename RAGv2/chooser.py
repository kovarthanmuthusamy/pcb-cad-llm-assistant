import ollama
import os



def check_folder_and_list_files(folder_path):
    """
    Checks if a folder exists and returns a list of files if the folder is present.

    :param folder_path: Path to the folder
    :return: List of files if the folder exists, otherwise None
    """
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        return os.listdir(folder_path)  # Returns list of files and directories inside
    else:
        return None  # Folder does not exist

def get_llm_response(user_input=None, model="deepseek-r1:32b"):
    set_1 = check_folder_and_list_files("data_def")
    set_2 = check_folder_and_list_files("data_pdf")
    prompt = f"""
        You have two sets of document names, each containing various topics. Your task is to determine if a given question can be answered solely based on the document names. Analyze only the document names and do not assume any additional content.

        Here are the document sets:

        Set A:"{set_1}"
        Set B: "{set_2}"

        Instructions for response:

        If Set A likely contains the answer, return 1.
        If Set B likely contains the answer, return 2.
        If neither contains the answer, return 3.
        If both sets contain relevant information, choose the set that is more specific and detailed.
        If you are unsure whether the question can be answered using these document names, return 3
        Now, analyze the following question and determine the appropriate response.

        Question: "{user_input}"

        Final Output: [1/2/3] (Respond only with the number.)"""
    
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    
    return response['message']['content'].strip()

# Example Usage
if __name__ == "__main__":
    user_question = "use of Transformer?"
    answer = get_llm_response(user_question)
    print(answer) # Expected output: "Yes" or "No"
