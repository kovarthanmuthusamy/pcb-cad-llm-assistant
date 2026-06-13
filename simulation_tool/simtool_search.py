import re
import time  # New import for tracking time
from sentence_transformers import SentenceTransformer, util
from simulation_tool.sim_tool import simulation

# Load S-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define synonyms for simulation types
SIM_TYPES = {
    "FD": ["FD", "Frequency Domain", "Frequency Analysis"],
    "SI": ["SI", "Signal Integrity"],
    "TDR": ["TDR", "Time-Domain Reflectometry"]
}

R_KEYWORDS = ["resistor", "R", "R_value", "resistance", "ohm"]
TL_KEYWORDS = ["transmission line", "TL", "line length", "length", "mm", "millimeter", "millimeters"]

# Flatten SIM_TYPES list
flat_sim_types = []
sim_type_map = {}
for key, synonyms in SIM_TYPES.items():
    for syn in synonyms:
        flat_sim_types.append(syn)
        sim_type_map[syn] = key  # Map synonyms to FD, SI, TDR

# Precompute embeddings
sim_type_embeds = model.encode(flat_sim_types, convert_to_tensor=True)
r_embeds = model.encode(R_KEYWORDS, convert_to_tensor=True)
tl_embeds = model.encode(TL_KEYWORDS, convert_to_tensor=True)
#print(sim_type_embeds,r_embeds,tl_embeds)

def extract_params_and_simulate(text):
    print("search......")
    start_time = time.time()  # Start timing

    # Encode input text
    text_embed = model.encode(text, convert_to_tensor=True)

    # Find best match for simulation type
    sim_scores = util.cos_sim(text_embed, sim_type_embeds)
    best_match = flat_sim_types[sim_scores.argmax().item()]
    sim_type = sim_type_map[best_match]  # Map to FD, SI, TDR

    # Find best match for R/TL values
    r_scores = util.cos_sim(text_embed, r_embeds).max().item()
    tl_scores = util.cos_sim(text_embed, tl_embeds).max().item()

    # Lower threshold to 0.3 for more flexibility
    r_match = r_scores > 0.3
    tl_match = tl_scores > 0.3

    # Extract all numbers with associated units or keywords
    matches = re.findall(r"(\d+\.?\d*)\s*(ohm|Ω|mm|millimeters?)", text, re.IGNORECASE)
    numbers = {unit.lower(): float(value) for value, unit in matches}

    # Assign numbers dynamically
    r_value = numbers.get("ohm") or numbers.get("Ω")
    tl_value = numbers.get("mm") or numbers.get("millimeter") or numbers.get("millimeters")

    end_time = time.time()  # End timing
    time_taken = end_time - start_time  # Calculate time taken

    print(f"type = {sim_type}, R_value = {r_value}, TL_value = {tl_value}, Time Taken = {time_taken:.4f} seconds")
    # Call the simulate function
    img_path = simulation(type=sim_type, R_value=r_value, L_value=tl_value)  # Using L_value for TL
    return img_path

# Test cases

# Test cases without resistor values and transmission line lengths

#extract_params_and_simulate("Perform Frequency Analysis with 28 ohm resistor value and 103 mm transmission line lenght")  
# type = FD, R_value = None, TL_value = None, Time Taken = X seconds

