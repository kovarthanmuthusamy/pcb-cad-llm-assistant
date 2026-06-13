from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
import numpy as np
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_image(img, target_size=(512, 512)):
    """
    Reduces resolution of image.
    """
    img = img.convert("RGB")  # Ensure compatibility
    img = img.resize(target_size, Image.LANCZOS)  # High-Quality Downscaling
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

# Initialize model and processor once (globally)
model_id = "ibm-granite/granite-vision-3.2-2b"
adapter_path = "C:/LLM_Agent/image_descriptor_tool/granite-vision-3.2-2b-04-03-2025"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["vision_tower", "lm_head"],
)

# Load model and processor (only once)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)
processor = AutoProcessor.from_pretrained(model_id)

# Load adapter if needed
model.load_adapter(adapter_path)

# Compile model for faster inference (trial)
compiled_model = torch.compile(model)
compiled_model.eval()

def granite_inference(image, model=model, processor=processor, adapter=True):
    """
    Function for generating a detailed description of the given image.
    """
    image = image_format_check(image)  # Converts to PIL Image format
    image = load_image(image)  # Reduces resolution

    prompt_txt = "You are useful for describing the images related to PCB design domain eg.(circuit images (star or p2p topology),simulation results images), give me the detailed description of the image."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_txt}
            ]
        }
    ]

    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=text_input,
        images=image,
        return_tensors="pt",
    ).to("cuda")  # Move inputs to the specified device

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

    return " description = " + " ".join(gen_text)  # Return the first decoded output text
