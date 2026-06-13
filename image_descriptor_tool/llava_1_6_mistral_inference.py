from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig
from PIL import Image
import numpy as np
import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_image(img, target_size=(512, 512)):
    """
    Reduces resolution of image.

    Args:
        img (PIL.Image type): Image to be reduced in resolution/size
        target_size (tuple, optional): Specify the target resolution/size. Defaults to (512, 512).

    Returns:
        img.copu(): Copy of the reduced resolution image
    """
    img = img.convert("RGB")  # Ensure compatibility
    img = img.resize(target_size, Image.LANCZOS) #High-Quality Downscaling
    return img.copy()

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


def llava_inference(model, processor, image):
    """This function is used for generating a detailed description of the given image
    
    Parameters:
        image: image in PIL image format
        adapter: flag to determine if to use fine tuned model or not

    Return:
        gen_text: The output generated text from the model
    """
    image = image_format_check(image) #Converts to PIL Image format
    image = load_image(image) #Reduces resolution to accomodate as input for model
    
    prompt_txt = "You are useful for describing the images related to PCB design domain eg.(circuit images (star or p2p topology),simulation resiults images), give me the detailed description of the image" #Prompt to be passed to the model
    
    messages = [
        {
            "role": "user",
            "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {   "type": "text", 
                                "text": prompt_txt
                            },
                        ],
        }
    ]
    
    #model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    #adapter_path = "llava-1.6-mistral-7b-13-03-2025"
    
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    # )
    
    # Load model and tokenizer
    # model = LlavaNextForConditionalGeneration.from_pretrained(model_id,
    #                                                   quantization_config=quantization_config,
    #                                                   torch_dtype=torch.float16)

    # processor = LlavaNextProcessor.from_pretrained(model_id)
    
    # if adapter == True:
    #     model.load_adapter(adapter_path) #Load fine tuned adapter
    
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template( messages, tokenize=False, add_generation_prompt=True )

    # Prepare the inputs for the model
    model_inputs = processor(
        #text=[text_input],
        text=text_input,
        images=image,
        return_tensors="pt",
    ).to("cuda")  # Move inputs to the specified device

    #Compile model to enable faster inference (trial)
    compiled_model = torch.compile(model)
    compiled_model.eval()
    
    # Generate text with the model
    generated_ids = compiled_model.generate(**model_inputs, max_new_tokens=4000)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    gen_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return gen_text  # Return the first decoded output text