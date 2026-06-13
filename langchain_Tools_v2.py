import setup_imports
from langchain.tools import StructuredTool
from typing import Optional
from pydantic import BaseModel, Field 
from image_descriptor_tool.granite_vision32_inference import  granite_inference
from PCB.PCB_expert import get_response
from RAGv2.final import RAG
from typing import Any
from simulation_tool.simtool_search import extract_params_and_simulate
import json,time,os,re


# ✅ Flatten any input (dict, list, object) into a readable string format
def flatten_input(input_data):
    """Flattens any input (dict, list, etc.) into a clean, readable string without brackets."""
    
    if isinstance(input_data, dict):
        return ", ".join([f"{k}: {flatten_input(v)}" for k, v in input_data.items()])  # Remove `{}` brackets

    elif isinstance(input_data, list):
        return ", ".join(map(str, input_data))  # Remove `[]` brackets
    
    return str(input_data)  # Convert everything else to string


# Structured input schema for Simulation Tool
class SimulationInput(BaseModel):
    type: str = Field(..., description="query for eCADstar simulations ")

# Simulation Tool using `StructuredTool`
def simulate(*args, **kwargs) -> str:
    
        # 🔹 Extract action_input (which might be passed as a positional or keyword argument)
    if args:
        input_data = args[0]  # Take the first positional argument if available
    else:
        action_input = kwargs.get("action_input", kwargs)
        input_data = action_input if isinstance(action_input, str) else action_input.get("query", action_input)
        
    if isinstance(input_data, dict):
       input_data = str(input_data)

    
    print(input_data)
    print("The simulation is in progress.....")
    img_path = extract_params_and_simulate(input_data)
    result = granite_inference(img_path)
    return f"Simulation completed for {input_data}, Result Image{result}"  


# RAG Tool Input Schema
# ✅ Define Input Schema
class RagInput(BaseModel):
    query: str = Field(..., description="Query to retrieve factual information from databases or the web.")

def rag_tool(*args, **kwargs) -> str:
    """Handles both positional and keyword arguments for RAG tool."""
    
    # 🔹 Extract action_input (which might be passed as a positional or keyword argument)
    if args:
        query_text = args[0]  # Take the first positional argument if available
    else:
        action_input = kwargs.get("action_input", kwargs)
        query_text = action_input if isinstance(action_input, str) else action_input.get("query", None)

    if query_text is None:
        return "❌ Error: No query provided to RAG Tool."
    
    query = flatten_input(query_text)
    
    rag_result = RAG(question=query, num_results=5, searcher="duckDuckgo", isLink=False, model="deepseek-r1:32b")
    
    # 🔹 Step 3: Extract only the relevant text (the "result" field)
    if isinstance(rag_result, dict) and "result" in rag_result:
        response_text = rag_result["result"].strip()
    else:
        response_text = json.dumps(rag_result, indent=2, ensure_ascii=False)  # Fallback if unexpected format

    # 🔹 Step 4: Remove any unwanted <think> tags (if present)
    cleaned_response = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

    del rag_result
    time.sleep(2)  # Ensure cleanup

    return cleaned_response  # ✅ Return only the extracted paragraph text


# PCB Expert Tool Input Schema
# ✅ Allow `query` to accept any input type
class PCBExpertInput(BaseModel):
    query: Any = Field(..., description="ask question")

def pcb_expert_tool(*args, **kwargs) -> str:
    """Handles both positional and keyword arguments for PCB Expert tool."""
    
    # 🔹 Extract action_input (which might be passed as a positional or keyword argument)
    if args:
        input_data = args[0]  # Take the first positional argument if available
    else:
        action_input = kwargs.get("action_input", kwargs)
        input_data = action_input if isinstance(action_input, str) else action_input.get("query", action_input)

    query_text = flatten_input(input_data)
    
    # 🔹 Call the actual expert system (replace with your real function)
    response = get_response(query_text)

    # 🔹 Ensure valid response
    if response and response.strip():
        return response

    print(f"⚠️ No response received from PCB Expert.")
    

    return "❌ Failed to get a response from PCB Expert."



