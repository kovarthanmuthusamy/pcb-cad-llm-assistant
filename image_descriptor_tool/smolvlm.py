#import torch
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
import torch

import torchvision
torchvision.disable_beta_transforms_warning()

def image_format_check(image):
    """
    Ensure the given image is in PIL format. If not, convert it to PIL.
    """
    if isinstance(image, Image.Image):
        return image  # Already a PIL image, return as is
    
    elif isinstance(image, str):  # If it's a file path
        return Image.open(image)
    
    elif isinstance(image, np.ndarray):  # If it's a NumPy array (e.g., OpenCV image)
        return Image.fromarray(image)
    
    else:
        raise TypeError("Unsupported image format. Provide a PIL image, NumPy array, or file path.")
    
def smolvlm_inference(model, processor, image):
    """This function is used for generating a detailed description of the given image
    
    Parameters:
        image: image in PIL image format

    Return:
        gen_text: The output generated text from the model
    """
    image = image_format_check(image)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # if model == "1":
    #     model_id = "HuggingFaceTB/SmolVLM-Instruct"
    # elif model == "2": 
    #     model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    
    
    #Decide which finetuned model version to be loaded
    # if model_no == "1":
    #     adapter_path = "smolvlm-instruct-13-03-2025"
    #     #adapter_path = "smolvlm-500m-instruct-17-02-2025/checkpoint-1028"  #Old model
    # elif model_no == "2":
    #     adapter_path = "smolvlm-500m-instruct-13-03-2025"
        #adapter_path = "smolvlm-500m-instruct-04-02-2025"  #Path to the trained model weights #Old Model
    # elif model == "3":
    #     adapter_path = "smolvlm-instruct-28-02-2025" 
    # else:
    #     print("Give a valid model number (1 or 2)")

    prompt_txt = "You are useful for describing the images related to PCB design domain eg.(circuit images (star or p2p topology),simulation resiults images), give me the detailed description of the image" #Prompt to be passed to the model

    # model = Idefics3ForConditionalGeneration.from_pretrained(
    # model_id,
    # device_map="auto",
    # #torch_dtype=torch.bfloat16,
    # )
    
    # processor = AutoProcessor.from_pretrained(model_id)
    
    # if adapter == True:
    #     model.load_adapter(adapter_path)

    text_template = [{'role': 'user',
                  'content': [{'type': 'image',
                               'image': image},
                              {'type': 'text', 
                               'text': prompt_txt}]
                 }]
    
    text_input = processor.apply_chat_template(
        text_template, 
        add_generation_prompt=True
    )

    model_inputs = processor(text=text_input,
                         images=image,
                         return_tensors="pt",).to("cuda")
    
    compiled_model = torch.compile(model)
    compiled_model.eval()
    with torch.no_grad():
        generated_ids = compiled_model.generate(**model_inputs, max_new_tokens=4000)

    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    gen_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    #return "description: " + " ".join(gen_text)
    return gen_text
