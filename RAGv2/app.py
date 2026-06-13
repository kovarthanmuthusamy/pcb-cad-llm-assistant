from flask import Flask, render_template, request
import ollama
import os
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    models = [m["model"] for m in ollama.list()["models"]] # Dropdown choices
    inputs= ["get_result","delete_chroma","delete_data","store_data"]

    if request.method == "POST":
        selected_input = request.form.get("input","get_result")  # Get selected dropdown value
        user_text = request.form.get("question","What is eCadstar")  # Get text input
        selected_option = request.form.get("model","mistral:7b")  # Get selected dropdown value
        uploaded_files = request.files.getlist("files")  # Get multiple files

        saved_files = []
        for file in uploaded_files:
            if file.filename:  # Ensure a file is selected
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(file_path)
                saved_files.append(file_path)
        print( user_text, selected_option,selected_input)
        # Call external Python script (pass text, dropdown value, and file paths)
        result = subprocess.run(
            ["python", "f_script.py", user_text,selected_input,selected_option] + saved_files,
            capture_output=True,
            text=True
        )

        return render_template("index.html", output=result.stdout, files=saved_files, models=models, inputs=inputs)

    return render_template("index.html", output=None, files=[], models=models, inputs=inputs)

# New route for Settings page
@app.route("/web", methods=["GET", "POST"])
def web():
    models = [m["model"] for m in ollama.list()["models"]] # Dropdown choices
    inputs= ["get_result","store_data","delete_chroma"]
    searcher=["duckDuckGo","google","brave","bing","yahoo","startpage"]
    if request.method == "POST":
        input = request.form.get("input","get_result")  # Get selected dropdown value
        query = request.form.get("query","What is eCadstar")  # Get text input
        isLink = request.form.get("isLink","false")        # Get text input
        question = request.form.get("question","What is eCadstar")  # Get text input
        model = request.form.get("model","deepseek-r1:32b")  # Get selected dropdown value
        search = request.form.get("search","duckDuckgo")  # Get selected dropdown value
        num_result=request.form.get("num_result",2)

        result = subprocess.run(
            ["python", "w_script.py", input, query,isLink,question,model,search,num_result],
            capture_output=True,
            text=True
        )

        return render_template("web.html", output=result.stdout,models=models, inputs=inputs,searcher=searcher)
    return render_template("web.html", output=None, models=models, inputs=inputs,searcher=searcher)

if __name__ == "__main__":
    app.run(debug=True)
