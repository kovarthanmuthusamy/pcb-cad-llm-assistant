import setup_imports
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOllama
from langchain.tools import StructuredTool
from langchain_Tools_v2 import rag_tool,pcb_expert_tool,RagInput,PCBExpertInput,simulate,SimulationInput
from image_descriptor_tool.granite_vision32_inference import  granite_inference
import time,re,json,os
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate

# 🔹 OpenAI API key — read from the environment (set OPENAI_API_KEY in your shell or a .env file).
# Never hardcode credentials. ChatOpenAI picks up OPENAI_API_KEY automatically.
from dotenv import load_dotenv
load_dotenv()


# 🔹 Initialize GPT-4-turbo with ReAct compatibility
orchestrator_llm = ChatOpenAI(
    model="gpt-4o",  # You can also use "gpt-4" or "gpt-3.5-turbo"
    temperature=0.7,
    verbose=True)
#initiate memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Tools List
tools = [
    
    StructuredTool.from_function(
    simulate,
    name="Simulation",
    description= """
    This action is used to invoke simulations within the eCADstar Electrical Editor simulation tool using the default P2P topology circuit file. The tool supports the following analysis Signal Integrity (SI) Analysis,Frequency Domain Analysis,Time-Domain Reflectometry (TDR) Analysis.
    You can dynamically adjust the following pvalues before simulations or leave as null,because there is default values for this parameters in the simulator - Resistor Value (in ohms),Transmission Line Length (in mm).
    The tool processes run simulation and provide relevant analysis results as image description based on the input configuration.Form questions like below to call the tool
    example tool input -  SI with resistor value 100 ohm and transmission value of 100 mm .
    """,
    args_schema=SimulationInput
    ),
   
    StructuredTool.from_function(
        pcb_expert_tool,
        name="PCB Expert",
        description="""Primary tool for PCB-related queries, including design, manufacturing, 
        noise issues, and electronics-specific topics. If a query partially overlaps with PCB topics,
        prioritize this tool before considering others.Ensure responses are precise, detailed, 
        and relevant using retrieved information.""",
        args_schema=PCBExpertInput
    ), 
    StructuredTool.from_function(
        rag_tool,
        name="RAG",
        description="""Retrieves accurate and relevant information from local databases, 
        user-provided documents, or web sources. Use this tool when other actions do not provide sufficient or reliable results.""",
        args_schema=RagInput
    )
]

# Initialize the Agent (Note: LangChain recommends LangGraph, but keeping agent for now)
agent_executor = initialize_agent(
    llm=orchestrator_llm,
    tools=tools,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
    max_iterations=4
)

# Get existing prompt
old_prompt = agent_executor.agent.llm_chain.prompt

# Get the list of messages
messages = old_prompt.messages.copy()

# Add a new system message at the top (or bottom depending on use case)
extra_instruction = SystemMessagePromptTemplate.from_template("You are chat assistant specializing in PCB design. You have access to expert tools to retrieve accurate technical information, perform simulations, and analyze design principles.provide accurate and broad answers.You may only take ONE action at a time and wait for observation,dont miss captions - Action: , Think etc..")
messages.insert(0, extra_instruction)  # or append() if you want it last

# Create new ChatPromptTemplate
new_prompt = ChatPromptTemplate.from_messages(messages)

# Assign it back to the agent
agent_executor.agent.llm_chain.prompt = new_prompt
#print(agent_executor.agent.llm_chain.prompt)

# Interactive Chatbot Loop
def chatbot_loop():
    print("⚡ PCB AI Chatbot Started. Type 'exit' to quit.\n")

    while True:
        query = input("🔹 Enter your query: ")
        start_time = time.time()
        if query.lower() == "exit":
            print("🔴 Exiting chatbot.")
            break

        ### Step 1: Ask for Image Paths (Manually Entered in Terminal)
        images = []
        for i in range(3):  # Max 3 images
            image_path = input(f"📷 Enter image path {i+1} (or press Enter to skip): ").strip()
            if not image_path:
                break
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}. Try again.")
                continue
            images.append(image_path)

        ### Step 2: Generate Image Descriptions (if images are provided)
        image_descriptions = []
        if images:
            print("🔍 Processing images...")  
            for img in images:
                try:
                    description = granite_inference(img)  # Call image descriptor agent
                    image_descriptions.append(description)
                except Exception as e:
                    print(f"⚠️ Error describing image {img}: {e}")
                    image_descriptions.append("Image description unavailable")

        ### Step 3: Format Image Descriptions for Input
        image_info = "\n".join([f"Image {i+1}: {desc}" for i, desc in enumerate(image_descriptions)])
        image_input = f"Attached images:\n{image_info}\n\n" if image_descriptions else ""

        ### Step 4: Add Chat History
        history_content = [msg.content for msg in memory.chat_memory.messages][-1:]
        formatted_history = "\n".join(history_content)
        formatted_input = {
            "input": f"previous history = {formatted_history}\n{image_input}\nUser: {query}"
        }
        print(formatted_input)

        ### Step 5: Call Agent 
        try:
            response = agent_executor.invoke(formatted_input)
            #print(response["output"])
            cleaned_response = re.split(r'</think>\s*', response["output"], maxsplit=1)[-1]

            # Step 2: Remove any trailing JSON
            json_match = re.search(r'\{\s*"action":\s*".*?"\s*,\s*"action_input":\s*".*?"\s*\}\s*$', cleaned_response, re.DOTALL)
            if json_match:
                cleaned_response = cleaned_response[:json_match.start()].strip()

# Print the cleaned response
            print(cleaned_response) 

        except Exception as e:
            print(f"⚠️ Error: {e}")

    end_time = time.time()
    print(f"⏳ Time taken: {end_time - start_time:.2f} seconds\n")

# Run the chatbot loop
if __name__ == "__main__":
    chatbot_loop()






