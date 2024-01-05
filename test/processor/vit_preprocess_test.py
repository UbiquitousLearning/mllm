from transformers import CLIPProcessor
from PIL import Image
text=["a photo of a cat", "a photo of a dog"]
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
inputs = processor(text=text, return_tensors="pt", padding=True)
print(inputs)