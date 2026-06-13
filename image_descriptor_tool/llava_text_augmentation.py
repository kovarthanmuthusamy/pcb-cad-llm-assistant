import os
import json
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer, util
from statistics import mean
import copy
import random

# =============== Additional Imports for Augmentation =============== #
import nltk
from nltk.corpus import wordnet
from deep_translator import GoogleTranslator

# Ensure you've downloaded the necessary NLTK data: nltk.download('wordnet')

# =============== Device Setup =============== #
device = "cuda" if torch.cuda.is_available() else "cpu"
print("*************************************************** The device has been set as ", device, " ***************************************************")

# =============== Similarity Model (SBERT) =============== #
similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)

# =============== Paraphrasing Models =============== #
models = {
    "FLAN-T5 Base": "google/flan-t5-base",
    "FLAN-T5 Large": "google/flan-t5-large",
    "BART Large CNN": "facebook/bart-large-cnn",
}

loaded_models = {}
for model_name, model_path in models.items():
    if "t5" in model_path:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    else:
        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
    loaded_models[model_name] = (model, tokenizer)

# =============== Augmentation Functions =============== #
def synonym_replacement(sentence, n=2):
    """
    Replace up to n words in 'sentence' with synonyms (WordNet-based).
    """
    words = sentence.split()
    new_sentence = words[:]
    if not words:
        return sentence  # Edge case: empty string
    
    # Number of replacements cannot exceed the number of words
    n = min(n, len(words))

    # Randomly pick indices to replace
    random_indices = random.sample(range(len(words)), n)

    for i in random_indices:
        synonyms = wordnet.synsets(words[i])
        if synonyms:
            # Take the first lemma name of the first synset
            synonym = synonyms[0].lemmas()[0].name()
            new_sentence[i] = synonym

    return " ".join(new_sentence)

def back_translation(text, lang="de"):
    """
    Translate 'text' to a foreign language (default=Deutsch), then back to English.
    """
    translated = GoogleTranslator(source="auto", target=lang).translate(text)
    return GoogleTranslator(source=lang, target="en").translate(translated)

def generate_paraphrase(text, model, tokenizer):
    """
    Paraphrase the text using T5/BART-based models,
    ensuring length is between 90% and 100% of original.
    """
    input_text = f"paraphrase: {text} </s>"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

    # Ensure the paraphrased text length is between 90% and 100% of original length
    min_length = int(len(text.split()) * 0.9)
    max_length = max(min_length, int(len(text.split()) * 1.0))  # Ensure max_length >= min_length

    outputs = model.generate(
        **inputs,
        min_length=min_length,
        max_length=max_length,
        num_return_sequences=1
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def calculate_similarity(original, paraphrased):
    """
    Calculate cosine similarity between original and paraphrased text using SBERT.
    """
    embeddings = similarity_model.encode([original, paraphrased], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity_score.item()

# =============== Main Script =============== #
orig_dataset_file_name = "json_data_llava_01_02_2025.json"
with open(orig_dataset_file_name, "r", encoding="utf-8") as file:
    dataset = json.load(file)

# Start our final dataset with the original data
final_dataset = list(dataset)

# Determine the highest ID in the original dataset (string or int),
# and convert to int for safe incrementing.
max_id_int = 0
if dataset:
    max_id_int = max(int(entry["id"]) for entry in dataset)

new_id_int = max_id_int + 1

# Track how many new samples each augmentation inserts.
synonym_count = 0
backtrans_count = 0
inserted_counts_paraphrasing = {model_name: 0 for model_name in loaded_models.keys()}

# =====================================================
# 1) Synonym Replacement
# =====================================================
print("\n===== Starting Synonym Replacement Augmentation =====")
for entry in tqdm(dataset, desc="Synonym Replacement...", unit="entry"):
    # We only augment GPT responses in 'conversations'
    for idx, conv in enumerate(entry["conversations"]):
        if conv["from"] == "gpt":
            original_text = conv["value"]
            
            # Generate synonym-replaced text
            replaced_text = synonym_replacement(original_text, n=2)
            
            # Create a new entry
            new_entry = copy.deepcopy(entry)
            new_entry["id"] = str(new_id_int)
            new_id_int += 1

            # Replace GPT message with synonym replaced text
            new_entry["conversations"][idx]["value"] = replaced_text

            final_dataset.append(new_entry)
            synonym_count += 1

print(f"Synonym Replacement: Inserted {synonym_count} new row(s).")

# =====================================================
# 2) Back Translation
# =====================================================
print("\n===== Starting Back Translation Augmentation (English -> Deutsch -> English)=====")
for entry in tqdm(dataset, desc="Back Translation...", unit="entry"):
    for idx, conv in enumerate(entry["conversations"]):
        if conv["from"] == "gpt":
            original_text = conv["value"]

            # Generate back-translated text
            bt_text = back_translation(original_text, lang="de")

            # Create a new entry
            new_entry = copy.deepcopy(entry)
            new_entry["id"] = str(new_id_int)
            new_id_int += 1

            # Replace GPT message with the back-translated text
            new_entry["conversations"][idx]["value"] = bt_text

            final_dataset.append(new_entry)
            backtrans_count += 1

print(f"Back Translation: Inserted {backtrans_count} new row(s).")

# =====================================================
# 3) Paraphrasing (with Similarity Threshold >= 93%)
# =====================================================
print("\n===== Starting Paraphrasing Augmentation =====")
for entry in tqdm(dataset, desc="Paraphrasing...", unit="entry"):
    for idx, conv in enumerate(entry["conversations"]):
        if conv["from"] == "gpt":
            original_text = conv["value"]

            # For each model, generate a paraphrase and check similarity
            for model_name, (model, tokenizer) in loaded_models.items():
                paraphrased_text = generate_paraphrase(original_text, model, tokenizer)
                similarity = calculate_similarity(original_text, paraphrased_text)

                # Only create a new entry if similarity >= 0.93
                if similarity >= 0.93:
                    new_entry = copy.deepcopy(entry)
                    new_entry["id"] = str(new_id_int)
                    new_id_int += 1

                    new_entry["conversations"][idx]["value"] = paraphrased_text
                    final_dataset.append(new_entry)

                    inserted_counts_paraphrasing[model_name] += 1

print("Paraphrasing Insertion Summary:")
for model_name, count in inserted_counts_paraphrasing.items():
    print(f"  {model_name} inserted: {count} new data sample(s)")

# =============== Write Combined Dataset =============== #
new_dataset_file_name =  orig_dataset_file_name + "_text_aug"
with open(new_dataset_file_name, "w", encoding="utf-8") as out_file:
    json.dump(final_dataset, out_file, ensure_ascii=False, indent=2)

print("\n===== Augmentation Completed =====")
print(f"Synonym Replacement total inserted: {synonym_count}")
print(f"Back Translation total inserted: {backtrans_count}")
print("Paraphrasing total inserted by model:")
for model_name, count in inserted_counts_paraphrasing.items():
    print(f"  {model_name}: {count}")
