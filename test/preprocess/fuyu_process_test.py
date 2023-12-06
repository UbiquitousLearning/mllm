import requests
from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image

model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
text_prompt = "Hello world!.s."
url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
image = Image.open(requests.get(url, stream=True).raw)
# with open("bus.tmp", "wb") as f:
#     f.write(image.tobytes())
inputs = processor(text=[text_prompt,text_prompt],images=[image,image], return_tensors="pt")
print(inputs)
