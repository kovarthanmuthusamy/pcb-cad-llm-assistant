from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import torch

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


def qwen_inference(model, processor, image):
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
    
    # model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    #adapter_path = "qwen2.5-vl-instruct-7b-04-03-2025"
    #adapter_path = "qwen2.5-vl-instruct-7b-13-03-2025"
    
    # BitsAndBytesConfig int-4 config
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # # default: Load the model on the available device(s)
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     model_id, torch_dtype="auto", device_map="auto"
    # )
    # # default processer
    # processor = AutoProcessor.from_pretrained(model_id)
    
    # if adapter == True:
    #     model.load_adapter(adapter_path) #Load fine tuned adapter
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

    #Compile model to enable faster inference (trial)
    compiled_model = torch.compile(model)
    compiled_model.eval()
    
    # Inference: Generation of the output
    generated_ids = compiled_model.generate(**inputs, max_new_tokens=4000)
    generated_ids_trimmed = [ out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids) ]
    
    gen_text = processor.batch_decode( 
                            generated_ids_trimmed, 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False
                            )
    
    return gen_text