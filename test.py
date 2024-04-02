import torch
import argparse
import sys
import os

from model import MattingNetwork

project_path = os.path.dirname(os.path.realpath(__file__))
print("project_path: %s" %project_path)

model = MattingNetwork('mobilenetv3').eval().cuda()  # 或 "resnet50"
model.load_state_dict(torch.load(os.path.join(project_path, 'rvm_mobilenetv3.pth')))


from inference import convert_video


def parse_args():
    parser = argparse.ArgumentParser(
        description='PP-HumanSeg inference for video')
    parser.add_argument(
        "--input",
        help="input video path",
        type=str,
        required=True)
    parser.add_argument('--output', help='output video path', type=str)
    parser.add_argument('--output_type', help='output video type', default="video",  type=str)
    parser.add_argument('--output-bg-image', type=str)
    parser.add_argument('--output-height', type=int)
    parser.add_argument('--output-width', type=int)
    parser.add_argument('--seq-chunk', type=int, default=12)
    parser.add_argument('--alpha', default=False, action='store_true')

    return parser.parse_args()


def main(args):
    if args.output_type == "video":
        base_dir_name = os.path.dirname(args.output)
        if not os.path.exists(base_dir_name):
            os.makedirs(base_dir_name, exist_ok=True)
        if os.path.isdir(args.output):
            args.output = os.path.join(args.output, os.path.basename(args.input))

    convert_video(
        model,                           # 模型，可以加载到任何设备（cpu 或 cuda）
        input_source=args.input,        # 视频文件，或图片序列文件夹
        output_type=args.output_type,             # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
        output_composition=args.output,    # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
        #output_alpha="pha.mp4",          # [可选项] 输出透明度预测
        #output_foreground="fgr.mp4",     # [可选项] 输出前景预测
        output_video_mbps=4,             # 若导出视频，提供视频码率
        downsample_ratio=0.4,           # 下采样比，可根据具体视频调节，或 None 选择自动
        seq_chunk=args.seq_chunk,                    # 设置多帧并行计算
        output_bg_image=args.output_bg_image,
        output_width=args.output_width,
        output_height=args.output_height,
        alpha=args.alpha,
    )


# python D:/code/RobustVideoMatting/test.py --input "D:/code/data/test1.mov" --output="D:/code/data/test1-output.mov"
# python D:/lch/code/RobustVideoMatting/test.py --input "C:/data/test2.mov" --output="C:/data/test2-2.mov" --alpha
# python D:/lch/code/RobustVideoMatting/test.py --input "C:/data/test2.mov" --output="C:/data/test2-2.mov" --output_type video
# python D:/lch/code/RobustVideoMatting/test.py --input "C:/data/test2.mov" --output="C:/data/test2-2.mov" --output_type video --output-bg-image "C:/data/bg.png"
# python D:/lch/code/RobustVideoMatting/test.py --input "C:/data/test2.mov" --output="C:/data/test2-2.mov" --output_type video --output-bg-image "C:/data/bg.png" --output-height 1920 --output-width 1400
if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))