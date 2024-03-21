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
    parser.add_argument(
        '--output', help='output video path', type=str)
    parser.add_argument(
        '--output_type', help='output video type', default="video",
        type=str)
    parser.add_argument('--output-bg-image', type=str)
    parser.add_argument('--output-height', type=int)
    parser.add_argument('--output-width', type=int)

    return parser.parse_args()


def main(args):
    convert_video(
        model,                           # 模型，可以加载到任何设备（cpu 或 cuda）
        input_source=args.input,        # 视频文件，或图片序列文件夹
        output_type=args.output_type,             # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
        output_composition=args.output,    # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
        output_alpha="pha.mp4",          # [可选项] 输出透明度预测
        output_foreground="fgr.mp4",     # [可选项] 输出前景预测
        output_video_mbps=4,             # 若导出视频，提供视频码率
        downsample_ratio=0.4,           # 下采样比，可根据具体视频调节，或 None 选择自动
        seq_chunk=12,                    # 设置多帧并行计算
        output_bg_image=args.output_bg_image,
        output_width=args.output_width,
        output_height=args.output_height,
    )



if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))