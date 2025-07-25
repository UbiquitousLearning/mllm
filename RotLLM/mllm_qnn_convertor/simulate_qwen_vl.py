"""
This file is a simulation of the inference process of a model that has been quantized using the quantization functions in the `quantization_simulation.py` file. 
The model is loaded from the path specified in the `model_name` argument, and the activation scales are loaded from the path specified in the `scale_file` argument. The `t01m_clip_threshold` argument specifies the threshold for clipping the activations. 
The model is quantized using the specified `model_type` argument, which determines the quantization function to be used. The quantized model is then used to generate an example based on the provided prompt.
"""
import json
from PIL import Image

import torch
import json

from args import args, remain_args
import argparse

parser = argparse.ArgumentParser()
_ = parser.parse_args(remain_args) # no additional arguments needed here

def get_photo_info(json_file):
    with open(json_file) as f:
        data = json.load(f)
    photo_info = []
    for photo in data:
        photo_info.append(
            {
                "image": "flick30k-10/" + str(photo["id"]) + ".jpg",
                "label": photo["text"],
            }
        )
    return photo_info


from utils.model import LLMNPUShowUIModel

from PIL import Image, ImageDraw

def draw_point_on_image(input_path, output_path, offset):
    """
    在图片上根据给定的偏移坐标绘制一个点，并保存为新图片。

    :param input_path: 输入图片的路径
    :param output_path: 输出图片的路径
    :param offset: 一个元组，形如 (水平偏移, 垂直偏移)，值在0到1之间
    """
    # 打开图片
    image = Image.open(input_path)
    width, height = image.size

    # 计算实际的绘制坐标
    x = int(offset[0] * width)
    y = int(offset[1] * height)

    # 创建一个可以在图片上绘图的对象
    draw = ImageDraw.Draw(image)

    # 绘制一个红色的点（半径为5）
    point_radius = 5
    draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill='red')

    # 保存图片
    image.save(output_path)



if __name__ == "__main__":
    with torch.no_grad():
        model = LLMNPUShowUIModel(
            args.tokenizer_name,
            args.model_name,
            args=args,
            t01m_clip_threshold=args.t01m_clip_threshold,
        )

        raw_image = Image.open(args.image_path)
        
        actions = [
            "锁屏",
            "通知与状态栏",
            "桌面",
            "显示与亮度",
            "声音与触感",
            "指纹、面部与密码",
            "隐私与安全",
            "应用设置",
            "省电与电池",
            "健康使用手机",
            "更多设置"
        ]
        
        import ast
        import os
        
        for action in actions:
            out_text = model.infer(raw_image, action, None)[0]

            try:
                point = ast.literal_eval(out_text)
            except Exception as e:
                print(f"Error parsing output: {out_text}, Error: {e}")
                continue

            print(f"Action: {action}, Point: {point}")
        
            image_name = os.path.basename(args.image_path)
            image_name = os.path.splitext(image_name)[0]  # Remove file extension
            out_image = f"{image_name}_{action}.png"
            os.makedirs(args.out_path, exist_ok=True)
            saved_figure = os.path.join(args.out_path, out_image)
            draw_point_on_image(args.image_path, saved_figure, point)
        
