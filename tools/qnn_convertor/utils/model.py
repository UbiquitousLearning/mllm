import os
import json
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

from .wrapper import VisionMLPWrapper, MLPWrapper
from .quantization_simulation import quantize_qwen2vl_qkvnobias_like
from .get_input_output_scales import get_clip_and_scale

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
TP_REGION_FUSE_THRESHOLD = 10

_SCREENSPOT_SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location."
_SYSTEM_point = "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
_SYSTEM_point_int = "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 1 to 1000."

_SCREENSPOT_USER = "<|image_1|>{system}{element}"


class LLMNPUShowUIProcessor:
    def __init__(self, processor_path):
        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            model_max_length=8192,
        )
        self.messages_template = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": None,
                },
                {"type": "text", "text": None},
            ],
        }
        self.chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    def screenspot_to_openai_qwen(self, element_name, image, xy_int=False):
        transformed_data = []
        user_content = []

        if xy_int:
            system_prompt = _SCREENSPOT_SYSTEM + " " + _SYSTEM_point_int
        else:
            system_prompt = _SCREENSPOT_SYSTEM + " " + _SYSTEM_point

        "{system}<|image_1|>{element}"
        user_content.append({"type": "text", "text": system_prompt})
        user_content.append(image)
        user_content.append({"type": "text", "text": element_name})

        transformed_data.append(
            {
                "role": "user",
                "content": user_content,
            },
        )
        return transformed_data

    def process(self, img: Image, text: str, json_path):

        img_dict = {
            "type": "image",
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
            "image": img,
        }
        source = self.screenspot_to_openai_qwen(text, img_dict)
        prompt = self.processor.tokenizer.apply_chat_template(
            source,
            chat_template=self.chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt],
            images=img,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        return inputs

class LLMNPUShowUIModel:
        
    def __init__(self, tokenizer_name,
                 model_name,
                 args, t01m_clip_threshold=64):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.processor = LLMNPUShowUIProcessor(tokenizer_name)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cuda", return_dict_in_generate=True,
        )
        # print(f"Model loaded: {model}")
        if args.online_rotation:
            import rotate
            if not args.random_rotate:
                print(f"load R from {args.R_path}")
                R_bin = torch.load(args.R_path)
                R = R_bin["R"]
                R_v = R_bin["R_v"]
                R_vit = R_bin["R_vit"]
                R_vs_vit = R_bin["R_vs_vit"]
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # model info
                num_layers = model.config.num_hidden_layers
                dim = model.config.hidden_size
                qo_heads = model.config.num_attention_heads
                head_dim = dim // qo_heads
                
                # get random hadamard rotation matrix
                R = rotate.get_orthogonal_matrix(dim, mode="hadamard", device=device)
                R_v = [rotate.get_orthogonal_matrix(head_dim, mode="hadamard", device=device) for _ in range(num_layers)]
                
                # vision info
                vit_dim = model.config.vision_config.embed_dim
                vit_heads = model.config.vision_config.num_heads
                vit_head_dim = vit_dim // vit_heads
                vit_layers = model.config.vision_config.depth
                R_vit = rotate.get_orthogonal_matrix(vit_dim, mode="hadamard", device=device)
                R_vs_vit = [rotate.get_orthogonal_matrix(vit_head_dim, mode="hadamard", device=device) for _ in range(vit_layers)]
                
                if args.save_rotation:
                    R_bin = {
                        "R": R,
                        "R_v": R_v,
                        "R_vit": R_vit,
                        "R_vs_vit": R_vs_vit
                    }
                    torch.save(R_bin, args.save_rotation)
                    print(f"Rotation matrix saved to {args.save_rotation}")

            from rotate import rotate_model

            rotate_model(model, R, R_v)
            if args.rotate_vit:
                rotate_model(model.visual, R_vit, R_vs_vit)
            print(f"finish online rotation")
            
        if args.vision_mlp_rotate:
            print("rotate vision mlp")
            from rotate import hadmard_matrix
            hadamard = hadmard_matrix(model.visual.config.embed_dim * model.visual.config.mlp_ratio,
                                             "cuda")
            for layer_idx in args.vision_layers_to_rotate:
                mlp = model.visual.blocks[layer_idx].mlp
                model.visual.blocks[layer_idx].mlp = VisionMLPWrapper(mlp, 
                                                                hadamard if args.rot_fc1 else None, 
                                                                hadamard if args.rot_fc2 else None)
                print(f"rotate mlp layer {layer_idx} with {hadamard.shape}")
        
        if args.lm_mlp_rotate:
            print("rotate lm mlp")
            from rotate import hadmard_matrix
            hadamard = hadmard_matrix(model.config.intermediate_size, "cuda")
            for layer_idx in args.lm_layers_to_rotate:
                mlp = model.model.layers[layer_idx].mlp
                model.model.layers[layer_idx].mlp = MLPWrapper(mlp, 
                                                                hadamard if args.rot_up else None, 
                                                                hadamard if args.rot_gate else None,
                                                                hadamard if args.rot_down else None)
                print(f"rotate lm mlp layer {layer_idx} with {hadamard.shape}")
        
        # print(f"model loaded: {model}")
        
        if not args.no_quantize:
            no_clip_input = {
                # "visual.blocks.22.mlp.fc2",
                # "visual.blocks.23.mlp.fc2",
                # "visual.blocks.24.mlp.fc2",
                # "visual.blocks.25.mlp.fc2",
                # "visual.blocks.26.mlp.fc2",
                # "visual.blocks.27.mlp.fc2",
            }
            
            no_clip_output = {
                # "visual.blocks.22.mlp.fc2",
                # "visual.blocks.23.mlp.fc2",
                # "visual.blocks.24.mlp.fc2",
                # "visual.blocks.25.mlp.fc2",
                # "visual.blocks.26.mlp.fc2",
                # "visual.blocks.27.mlp.fc2",
            }
        
            act_dict = json.load(open(args.scale_file.name))

            act_scales, clip_top, return_dict = get_clip_and_scale(act_dict, t01m_clip_threshold, args.clip_all,
                                                                   no_clip_input=no_clip_input, no_clip_output=no_clip_output)

            file_name = os.path.basename(model_name) + "." + os.path.basename(args.scale_file.name) + "." + str(t01m_clip_threshold)  + ".clip.info"
            
            f = None
            if args.save_clip_info:
                f = open(file_name, "a")
            
            print(f"clip input num: {return_dict['clip_input_num']}", file=f)
            print(f"clip output num: {return_dict['clip_output_num']}", file=f)
            print(f"no clip input num: {return_dict['no_clip_input_num']}", file=f)
            for i in return_dict["no_clip_input_name"]:
                print(f"no clip input: {i}", file=f)
            print(f"no clip output num: {return_dict['no_clip_output_num']}", file=f)
            for i in return_dict["no_clip_output_name"]:
                print(f"no clip output: {i}", file=f)
                
            if args.save_clip_info:
                f.close()
                
            print(f"quantize_vit: {args.quantize_vit}")
                    
            nbits_dict = {
                "model.layers.1.mlp.down_proj": 16,
                "model.layers.26.mlp.down_proj": 16,
            }
            
            model = quantize_qwen2vl_qkvnobias_like(model, act_scales, 
                                                    layer_clip=clip_top, 
                                                    quantize_ViT=args.quantize_vit,
                                                    nbits_dict=nbits_dict)

        self.model = model
        

    def infer(self, image: Image, text: str, json_path: str):
        inputs = self.processor.process(image, text, json_path)
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=128,
                                            # do_sample=False,
                                            # temperature=None,
                                            # top_p=None,
                                            # top_k=None
                                            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids["sequences"])
        ]
        output_text = self.processor.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text
    
