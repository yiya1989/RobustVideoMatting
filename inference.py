"""
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "CHECKPOINT" \
    --device cuda \
    --input-source "input.mp4" \
    --output-type video \
    --output-composition "composition.mp4" \
    --output-alpha "alpha.mp4" \
    --output-foreground "foreground.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 1
"""

import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from tqdm.auto import tqdm
import numpy as np
import cv2

import argparse
from model import MattingNetwork
import torch.nn.functional as F

from inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter


def convert_video(model,
                  input_source: any,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_bg_image: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  output_width: Optional[int] = None,
                  output_height: Optional[int] = None,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None):
    
    """
    Args:
        input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
        output_type: Options: ["video", "png_sequence"].
        output_composition:
            The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        output_bg_image: The background image for output from the model.
        output_width: output width.
        output_height: output height.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
        progress: Show progress bar.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """
    
    assert downsample_ratio is None or (downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha, output_foreground]), 'Must provide at least one output.'
    assert output_type in ['video', 'png', 'jpg'], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'
    
    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    if isinstance(input_source, (np.ndarray, np.generic) ):
        pass
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
    else:
        source = ImageSequenceReader(input_source, transform)
    reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers)
    
    # Initialize writers
    if output_type == 'video':
        frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
        output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
        if output_composition is not None:
            writer_com = VideoWriter(
                path=output_composition,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_alpha is not None:
            writer_pha = VideoWriter(
                path=output_alpha,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_foreground is not None:
            writer_fgr = VideoWriter(
                path=output_foreground,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, output_type)
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, output_type)
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, output_type)

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device

    bg_image_cv = None
    if output_bg_image:
        bg_image_cv = load_background_image(output_bg_image)
    source_width, source_height = source[0].shape[1:]
    print(f"input source shape {source_width}*{source_height}")

    if not output_width and not output_height:
        # if bg_image_cv is not None:
        #     output_width, output_height = bg_image_cv.shape[:2]
        # else:
        #     output_width, output_height =source_width, source_height
        output_width, output_height =source_width, source_height
    
    # if (output_composition is not None) and (output_type == 'video'):
    #     bgr = torch.tensor([140, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
    # else:
    #     bgr = torch.tensor([140, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
    
    bgr = torch.tensor([140, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
    print(f"bgr: {bgr.shape}, {bgr}")

    if not output_bg_image:
        bgr = torch.tensor([140, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
    else:
        # h, w = get_video_size(input_source)
        # h, w = source[0].shape[1:]
        bgr = load_background_image_bgr(bg_image_cv, output_width, output_height, device, dtype)
        print(f"bgr: {bgr.shape}, {output_width}, {output_height}")

    try:
        with torch.no_grad():
            bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
            rec = [None] * 4
            for src in reader:

                if downsample_ratio is None:
                    downsample_ratio = auto_downsample_ratio(*src.shape[2:])

                src = src.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
                fgr, pha, *rec = model(src, *rec, downsample_ratio)
                # print(f"###{fgr.shape}")
                # fgr_height, fgr_width = fgr.shape[3:]
                fgr_width, fgr_height  = fgr.shape[3:]
                print(f"org: fgr: {fgr.shape}, pha: {pha.shape}, bgr: {bgr.shape}, output: {output_height} {output_width}")

                if output_height > fgr_height  or output_width > fgr_width:
                    left = int((output_height - fgr_height)/2)
                    right = int(output_height - fgr_height - left)
                    up = int((output_width - fgr_width)/2)
                    down = int(output_width - fgr_width - up)
                    p2d = (left, right, up, down)
                    fgr = F.pad(fgr, p2d, 'constant', 0)
                    pha = F.pad(pha, p2d, 'constant', 0)

                
                print(f"padded: fgr: {fgr.shape}, pha: {pha.shape}, bgr: {bgr.shape}, pad: {left} {right} {up} {down}")

                if output_foreground is not None:
                    writer_fgr.write(fgr[0])
                if output_alpha is not None:
                    writer_pha.write(pha[0])
                if output_composition is not None:
                    if output_type == 'video':
                        com = fgr * pha + bgr * (1 - pha)
                    elif output_type == 'png':
                        fgr = fgr * pha.gt(0)
                        com = torch.cat([fgr, pha], dim=-3)
                    else:
                        com = fgr * pha + bgr * (1 - pha)
                    writer_com.write(com[0])
                
                bar.update(src.size(1))

    finally:
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()


def get_video_size(input_video_path):
    vid = cv2.VideoCapture(input_video_path)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    return height, width


def load_background_image(background_image_path):
    cv_image = cv2.imread(background_image_path)
    print(f"load_background_image shape {cv_image.shape}, {cv_image[0,0]}")
    return cv_image


def load_background_image_bgr(bg_image_cv, height, width, device, dtype):
    height, width = int(height), int(width)

    img_height, img_width = bg_image_cv.shape[:2]
    img_scale_height, img_scale_width = height, width
    if img_height / img_width > height / width:
        img_scale_height = img_height * img_scale_width / img_width
    else:
        img_scale_width = img_width * img_scale_height / img_height 
    
    img_scale_height, img_scale_width = int(img_scale_height), int(img_scale_width)
    print(f"load_background_image resize from {bg_image_cv.shape} to {img_scale_height}*{img_scale_width}, target: {height}*{width} {bg_image_cv[0,0]}")

    bg_image_cv = cv2.cvtColor(bg_image_cv, cv2.COLOR_BGR2RGB)
    bg_image_cv = cv2.resize(bg_image_cv, (img_scale_width, img_scale_height))
    start_height = int((img_scale_height-height)/2)
    start_width = int((img_scale_width-width)/2)
    bg_image_cv = bg_image_cv[start_height:start_height+height,start_width:start_width+width]
    print(f"load_background_image cut from {img_scale_height}*{img_scale_width} to {bg_image_cv.shape} {bg_image_cv[0,0]}")

    # # show image    
    # cv2.imshow('cv_image', cv_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # # 将图像转换为PyTorch张量
    # tensor_image = torch.from_numpy(rgb_image.transpose((2, 0, 1))).float()
    # tensor = tensor_image.to(device, dtype)

    transform = transforms.Compose([ 
        transforms.ToTensor() 
    ])
    tensor = transform(bg_image_cv).to(device, dtype)   #.permute(0, 1, 4, 2, 3)
    print(f"load_background_image tensor: {tensor.shape}  {tensor[:,0,0]}")
    bgr = tensor.view(1, 1, 3, height, width)
    print(f"load_background_image bgr: {bgr.shape} {bgr[0,0,:,0,0]}")

    return bgr


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


class Converter:
    def __init__(self, variant: str, checkpoint: str, device: str):
        self.model = MattingNetwork(variant).eval().to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)
        self.device = device
    
    def convert(self, *args, **kwargs):
        convert_video(self.model, device=self.device, dtype=torch.float32, *args, **kwargs)
    
if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--input-source', type=str, required=True)
    parser.add_argument('--input-resize', type=int, default=None, nargs=2)
    parser.add_argument('--downsample-ratio', type=float)
    parser.add_argument('--output-composition', type=str)
    parser.add_argument('--output-alpha', type=str)
    parser.add_argument('--output-foreground', type=str)
    parser.add_argument('--output-bg-image', type=str)
    parser.add_argument('--output-type', type=str, required=True, choices=['video', 'png_sequence'])
    parser.add_argument('--output-video-mbps', type=int, default=1)
    parser.add_argument('--output-height', type=int)
    parser.add_argument('--output-width', type=int)
    parser.add_argument('--seq-chunk', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--disable-progress', action='store_true')
    args = parser.parse_args()
    
    converter = Converter(args.variant, args.checkpoint, args.device)
    converter.convert(
        input_source=args.input_source,
        input_resize=args.input_resize,
        downsample_ratio=args.downsample_ratio,
        output_type=args.output_type,
        output_composition=args.output_composition,
        output_alpha=args.output_alpha,
        output_foreground=args.output_foreground,
        output_bg_image=args.output_bg_image,
        output_video_mbps=args.output_video_mbps,
        output_width=args.output_width,
        output_height=args.output_height,
        seq_chunk=args.seq_chunk,
        num_workers=args.num_workers,
        progress=not args.disable_progress
    )
    
    
