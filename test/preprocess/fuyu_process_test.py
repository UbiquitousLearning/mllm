import requests
import torch
from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
torch.set_printoptions(profile="full")

model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
text_prompt = "a coco-style image captioning model"
url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
image = Image.open(requests.get(url, stream=True).raw)
# with open("bus.png", "wb") as f:
#     f.write(image.tobytes())
inputs = processor(text=text_prompt,images=[image,image], return_tensors="pt")
print(inputs)
print(inputs['image_patches'][0].shape)
