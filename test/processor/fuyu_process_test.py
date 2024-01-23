import requests
import torch
from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
torch.set_printoptions(profile="full")

model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id,local_files_only=True)
text_prompt = "a coco-style image captioning model"
url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
image = Image.open("./bus.png")
# with open("bus.png", "wb") as f:
#     f.write(image.tobytes())
inputs = processor(text=text_prompt,images=[image], return_tensors="pt")
print(inputs)
for k, v in inputs.items():
    print(k, v[0].shape)
# print(inputs['image_patches'][0].shape)
print(inputs['input_ids'][0])