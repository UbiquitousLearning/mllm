import argparse
import ast

def parse_list_int(value):
    """将字符串解析为 list[int]"""
    try:
        # 使用 ast.literal_eval 安全地解析字符串
        parsed_list = ast.literal_eval(value)
        # 验证是否为 list[int]
        if isinstance(parsed_list, list) and all(isinstance(x, int) for x in parsed_list):
            return parsed_list
        else:
            raise ValueError("输入的值不是一个有效的 list[int]")
    except Exception as e:
        raise argparse.ArgumentTypeError(f"无效的 list[int] 格式: {value}. 错误: {e}")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="/data/share/datasets_roci/ScreenSpot/")
parser.add_argument("--model_name", type=str, default='/data/share/ShowUI-2B')
parser.add_argument("--tokenizer_name", type=str, default='/data/share/Qwen2-VL-2B-Instruct')
parser.add_argument("--scale_file", type=argparse.FileType("r"), default='convertor/scales/showui_screenqa_nobias.json')
parser.add_argument("--image_path", type=str, default="convertor/image.png")
parser.add_argument("--out_path", type=str, default="convertor/showui_res")
parser.add_argument("--no_quantize", action="store_true", help="Whether to use quantization")
parser.add_argument("--t01m_clip_threshold", type=int, default=100000)
parser.add_argument("--output_file", type=str, default='showui_rota_screenqa_nobias_dis.json')
parser.add_argument('--quantize_vit', action='store_true', help='quantize vit or not')
parser.add_argument('--rotate_vit', action='store_true', help='rotate vit or not')
parser.add_argument('--online_rotate', action='store_true', help='do online rotation')
parser.add_argument("--random_rotate", action="store_true", help="do random rotation or not")
parser.add_argument("--R_path", type=str, default="./R.bin", help="path of rotation matrix")
parser.add_argument("--clip_all", action="store_true", help="clip all layer")
parser.add_argument("--vision_mlp_rotate", action="store_true", help="rotate vision MLP's activation internally")
parser.add_argument("--rot_fc1", action="store_true", help="rotate fc1's output internally")
parser.add_argument("--rot_fc2", action="store_true", help="rotate fc2's input internally")
parser.add_argument(
        "--vision_layers_to_rotate",
        type=parse_list_int,
        default=[22,23,24,25,26,27],
        help="list of int. e.g. [1, 2, 3, 4, 5]"
    )
parser.add_argument("--lm_mlp_rotate", action="store_true", help="rotate lm MLP's activation internally")
parser.add_argument("--rot_down", action="store_true", help="rotate down's input internally")
parser.add_argument("--rot_gate", action="store_true", help="rotate gate's output internally")
parser.add_argument("--rot_up", action="store_true", help="rotate up's output internally")
parser.add_argument(
        "--lm_layers_to_rotate",
        type=parse_list_int,
        default=[1, 26],
        help="list of int. e.g. [1, 2, 3, 4, 5]"
    )

parser.add_argument("--save_clip_info", action="store_true", help="save clip info or not")
# 添加模型类型参数
parser.add_argument('--model_type', type=str, default='qwen-vl', 
                    help='Model type (qwen, qwen-vl)')
args, remain_args = parser.parse_known_args()
