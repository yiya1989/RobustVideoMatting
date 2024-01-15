import sys
import pathlib

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")

from os import listdir, path

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse
import os
import shutil
import cv2
import traceback
import subprocess
from tqdm import tqdm
from glob import glob

import face_detection
import torch

import ffmpeg

# sys.path.append("../RobustVideoMatting")
import inference
import inference_utils

import numpy as np
import glob

import video_utils


mode_choice = ["all", "convert", "origin_image", "image", "lm", "export", "combine"]

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=4, type=int)
parser.add_argument("--input", help="input video file path", required=False)
parser.add_argument("--convert-output", help="convert video file path", default="output.mp4", required=False)
parser.add_argument("--export-output", help="output video file path", default="output.mp4", required=False)
parser.add_argument("--combine-input-audio", help="combine input video file path", default="", required=False)
parser.add_argument("--combine-input", help="combine input video file path", default="", required=False)
parser.add_argument("--combine-output", help="combine output video file path", default="", required=False)
parser.add_argument("--origin_image", help="Folder of the video origin image dir path")
parser.add_argument("--head_image", help="Folder of the preprocessed head image dir path", required=False)
parser.add_argument("--head_image_lm", type=str, help="video change to green screen image")
parser.add_argument("--mode", type=str, choices=mode_choice, nargs='+', help="export org_imgs or compose video", required=True)
parser.add_argument("--fps", default=25, type=int, help="expect video fps")
parser.add_argument("--size", type=int, help="video or image size")
parser.add_argument("--trim_start", default=None, type=int, help="video start time")
parser.add_argument("--trim_end", default=None, type=int, help="video trim end time")
parser.add_argument("--trim_duration", default=None, type=int, help="video trim duration")
parser.add_argument("--sample", default=25, type=int, help="video detect sample rate")
parser.add_argument("--debug", action='store_true', help="video trim duration")
parser.add_argument("--overwrite", action='store_true', help="processed video overwrite")
parser.add_argument("--with-bg", action='store_true', help="export video with background")

args = parser.parse_args()

root_path = os.path.dirname(os.path.abspath(__file__))
if not path.isfile(os.path.join(root_path, 'face_detection/detection/sfd/s3fd.pth')):
    raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
                            before running this script!')

if torch.cuda.is_available():
    # print("cuda is available")
    cuda_count = torch.cuda.device_count()
    current_cuda = torch.cuda.current_device()
    cuda_name = torch.cuda.get_device_name(current_cuda)
    # print("cuda info, total: {}, current: {}, cuda name: {}".format(cuda_count, current_cuda, cuda_name))
else:
    print("cuda is not available")

if torch.cuda.is_available():
    fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
                                       device='cuda')]
    # device='cuda:{}'.format(id)) for id in range(args.ngpu)]
else:
    args.ngpu = 0
    fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
                                       device='cpu')]


def clear_dir(path):
    if not os.path.isdir(path):
        raise Exception(f"input path is not dir, path: {path}")
    files = glob.glob(os.path.join(path, "*"))
    for f in files:
        if os.path.isfile(f):
            os.remove(f)


def get_video_info(video_path):
    video = video_utils.VideoInfo(video_path)
    return video


def green_screen_video(args, video_info):
    if not args.head_image_lm:
        return video_info

    print('===>Change images to green screen images...')
    os.makedirs(args.head_image_lm, exist_ok=True)
    clear_dir(args.head_image_lm)

    args.variant = 'mobilenetv3'
    args.checkpoint = os.path.join(root_path, 'rvm_mobilenetv3.pth')
    args.device = 'cuda'

    args.input_source = args.head_image
    args.input_resize = None
    args.downsample_ratio = 1
    args.output_type = "jpg"
    args.output_composition = args.head_image_lm
    args.output_alpha = None
    args.output_foreground = None
    args.output_video_mbps = 1
    args.seq_chunk = 8
    args.num_workers = 8
    args.disable_progress = False

    print(f'[INFO] ===== change images to green_screen =====')
    converter = inference.Converter(args.variant, args.checkpoint, args.device)
    converter.convert(
        input_source=args.input_source,
        input_resize=args.input_resize,
        downsample_ratio=args.downsample_ratio,
        output_type=args.output_type,
        output_composition=args.output_composition,
        output_alpha=args.output_alpha,
        output_foreground=args.output_foreground,
        output_video_mbps=args.output_video_mbps,
        seq_chunk=args.seq_chunk,
        num_workers=args.num_workers,
        progress=not args.disable_progress
    )
    print(f'[INFO] ===== change images to green_screen done =====')

    return video_info


def convert_video(args, video_info):
    need_convert = False
    base_path_splits = video_info.Path.rsplit(".")
    base_path = base_path_splits[:len(base_path_splits)-1]
    if not args.convert_output:
        output_video_path = ".".join(base_path) + "-processed.mp4"
    else:
        output_video_path = args.convert_output

    if not args.overwrite:
        if os.path.isfile(output_video_path):
            video_info = get_video_info(output_video_path)
            print("new video info: {}".format(video_info))
            return video_info

    print(f"convert video from {video_info.Path} to {output_video_path}")
    stream = ffmpeg.input(video_info.Path)

    # -ss 00:00:04 -to 00:05:00
    out_kwargs = {}
    trim_kwargs = {}
    if args.trim_start:
        need_convert = True
        out_kwargs["ss"] = args.trim_start
    if args.trim_end:
        need_convert = True
        out_kwargs["to"] = args.trim_end
    if args.trim_duration:
        need_convert = True
        out_kwargs["t"] = args.trim_duration
    if len(trim_kwargs) > 0:
        stream = stream.trim(**trim_kwargs)

    if video_info.FPS != args.fps:
        need_convert = True
        # stream = stream.filter('fps', fps=args.fps, round='up')
    out_kwargs["r"] = str(args.fps)

    loglevel = "warning"
    loglevel = "info"
    if need_convert:
        # .global_args('-report')
        # print(f"output_video_path: {output_video_path}")
        stream = stream.output(
            output_video_path, loglevel=loglevel,
            acodec="copy", 
            # vcodec="copy",
            audio_bitrate=video_info.AudioBitrate, 
            video_bitrate=video_info.Bitrate,
            crf=18, # 压缩比，一般18~28。-qscale is ignored, -crf is recommended
            ac=1,
            ar='16k',
            # ar='44100',
            **out_kwargs)

        if args.overwrite:
            stream = stream.overwrite_output()
        stream.run()
    else:
        shutil.copyfile(video_info.Path, output_video_path)

    video_info = get_video_info(output_video_path)
    print("new video info: {}".format(video_info))

    return video_info


def export_origin_image(args, video_info):
    stream = ffmpeg.input(video_info.Path).filter('fps', fps=args.fps)
    # thumb_stream = stream.filter('scale', target_witdh, -1)
    # thumb_stream.output('thumbs/test-%d.jpg', start_number=0).run(quiet=True)

    # head_image_dir = args.head_image
    # if head_image_dir:
    #     os.makedirs(head_image_dir, exist_ok=True)
    #     clear_dir(head_image_dir)

    origin_image_dir = args.origin_image
    if len(origin_image_dir) > 0:
        out_kwargs = {}
        out_kwargs["qscale:v"] = str(1)
        os.makedirs(origin_image_dir, exist_ok=True)
        clear_dir(origin_image_dir)
        stream.output(f'{origin_image_dir}/%d.jpg', start_number=0, crf=18, **out_kwargs).run()

    return 0


def process_video_file(args, video_info):
    
    head_detect_image_dir = os.path.join(args.head_image, "detect")
    os.makedirs(head_detect_image_dir, exist_ok=True)
    clear_dir(head_detect_image_dir)

    head_image_dir = args.head_image
    os.makedirs(head_image_dir, exist_ok=True)
    clear_dir(head_image_dir)

    _, image_path_list = inference_utils.get_file_list(args.origin_image, "jpg", True)

    max_frame = len(image_path_list)
    # max_frame = 10

    preds_cache = np.empty((max_frame, 4))
    preds_location = np.empty((max_frame, 6))

    idx = 0
    ix1, iy1, ix2, iy2 = 0, 0, int(video_info.Width), int(video_info.Height)
    bar = tqdm(total=max_frame, disable=False, dynamic_ncols=True, desc='DetectHead')
    while idx < max_frame:
        start_num = idx
        batch = []
        for b_idx in range(args.batch_size):
            # print(f"batch_size: {idx}")
            name = image_path_list[idx]
            img = cv2.imread(name, cv2.IMREAD_COLOR)
            batch.append(img[iy1:iy2+1, ix1:ix2+1])
            idx += args.sample
            if idx >= max_frame:
                break
        bar.update(idx-start_num)

        preds = fa[0].get_detections_for_batch(np.asarray(batch))
        pred_idx = start_num
        for j, f in enumerate(preds):
            if f is None:
                continue
            x1, y1, x2, y2 = f
            # print("process_video_file detected: {}, {}".format(pred_idx, f))
            for t_idx in range(args.sample):
                # print(f"pred_idx: {pred_idx}")
                preds_cache[pred_idx] = np.array([x1, y1, x2, y2])
                cv2.imwrite(path.join(head_detect_image_dir, '{}.jpg'.format(pred_idx)),  batch[j][y1:y2+1, x1:x2+1])
                pred_idx += 1
                if pred_idx >= max_frame:
                    break

        # width1 = x2 - x1 + 1
        # hight1 = y2 - y1 + 1
        # diffx = (width1)//2 * 0.5
        # diffy = (hight1)//2 * 0.5

        # iix1 = int(x1 - diffx)
        # iix2 = int(x2 + diffx)
        # iiy1 = int(y1 - diffy)
        # iiy2 = int(y2 + diffy)

        # ix1 = int(max(iix1, 0))
        # iy1 = int(max(iiy1, 0))
        # ix2 = int(min(iix2, video_info.Width))
        # iy2 = int(min(iiy2, video_info.Height))
        
    bar.refresh()
    bar.close()
    # print(f"preds_cache: {preds_cache} {preds_cache.shape}")

    # method1: get range
    x1 = preds_cache[:,0].min()
    x2 = preds_cache[:,2].max()
    y1 = preds_cache[:,1].min()
    y2 = preds_cache[:,3].max()

    width = x2 - x1 + 1
    hight = y2 - y1 + 1 
    square = int(max(width, hight) * 1.8)
    print(f"Head image location input: [{x1}, {y1}, {x2}, {y2}] {width}x{hight} -> {square}x{square}")
    
    diffx = (square - width)//2
    diffy = (square - hight)//3

    iix1 = int(x1 - diffx)
    iix2 = iix1 + square - 1
    iiy1 = int(y1 - diffy*2)
    iiy2 = iiy1 + square - 1
    iiwidth = iix2 - iix1 + 1
    iihight = iiy2 - iiy1 + 1
    print(f"Head image location output: [{iix1}, {iiy1}, {iix2}, {iiy2}] {iiwidth}x{iihight}")

    bar = tqdm(total=max_frame, disable=False, dynamic_ncols=True, desc='ExportHead')
    for idx in range(max_frame):
        name = image_path_list[idx]
        img = cv2.imread(name, cv2.IMREAD_COLOR)

        x1, y1, x2, y2 = preds_cache[idx]
        width2 = x2 - x1 + 1
        hight2 = y2 - y1 + 1 
        if args.debug:
            print("process_video_file dealed: {}, {} -> [{} {} {} {}], {}x{} -> {}x{}".format(
                idx, preds_cache[idx], x1, y1, x2, y2, width, hight, width2, hight2))
        preds_location[idx] = np.array([iix1, iiy1, iix2, iiy2, iiwidth, iihight])
        head_img = img[iiy1:iiy2+1, iix1:iix2+1]
        if square > 512:
            img_size=(512, 512)
            head_img = cv2.resize(head_img, img_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(path.join(head_image_dir, '{}.jpg'.format(idx)), head_img)
        bar.update(1)

    bar.refresh()
    bar.close()

    
    # # method2: get range
    # width = preds_cache[:, 2] - preds_cache[:, 0] + 1
    # hight = preds_cache[:, 3] - preds_cache[:, 1] + 1

    # square = int(max(width.max(), hight.max()) * 1.6)
    # print("target talking head square: {}".format(square))

    # bar = tqdm(total=max_frame, disable=False, dynamic_ncols=True, desc='ExportHead')
    # for idx in range(max_frame):
    #     name = image_path_list[idx]
    #     img = cv2.imread(name, cv2.IMREAD_COLOR)

    #     x1, y1, x2, y2 = preds_cache[idx]
    #     if args.debug:
    #         print("process_video_file detected: {}, {}, {}, {}, {}".format(
    #             idx, x1, y1, x2, y2))

    #     width1 = width[idx]
    #     hight1 = hight[idx]
    #     diffx = (square - width1)//2
    #     diffy = (square - hight1)//3

    #     x1 = int(x1 - diffx)
    #     x2 = int(x2 + diffx)
    #     y1 = int(y1 - diffy*1)
    #     y2 = int(y2 + diffy*2)

    #     width2 = x2 - x1 + 1
    #     hight2 = y2 - y1 + 1

    #     if hight2 < square:
    #         y2 += square-hight2
    #     if width2 < square:
    #         x2 += square-width2

    #     # we may need to check square in images
    #     x1 = max(x1, 0)
    #     y1 = max(y1, 0)
    #     x2 = min(x2, video_info.Width)
    #     y2 = min(y2, video_info.Height)

    #     width3 = x2 - x1 + 1
    #     hight3 = y2 - y1 + 1

    #     if args.debug:
    #         print("process_video_file dealed: {}, {} -> [{} {} {} {}], {}x{} -> {}x{}".format(
    #             idx, preds_cache[idx], x1, y1, x2, y2, width1, hight1, width3, hight3))
    #     preds_location[idx] = np.array([x1, y1, x2, y2, width3, hight3])
    #     cv2.imwrite(path.join(head_image_dir, '{}.jpg'.format(idx)),
    #                 img[y1:y2+1, x1:x2+1])
    #     bar.update(1)

    # bar.refresh()
    # bar.close()

    py_file = os.path.join(head_image_dir, "head_location.npy")
    with open(py_file, 'wb') as f:
        np.save(f, preds_location)


def process_video_file2(args, video_info):

    head_image_dir = args.head_image
    os.makedirs(head_image_dir, exist_ok=True)
    clear_dir(head_image_dir)

    vfile = video_info.Path
    video_stream = cv2.VideoCapture(vfile)

    max_frame = int(video_info.NumFrames)
    # max_frame = 30

    print("Loading video...")
    bar = tqdm(total=max_frame, disable=False, dynamic_ncols=True)
    frames = []
    while 1:
        still_reading, frame = video_stream.read()
        bar.update(1)
        if not still_reading or (max_frame > 0 and len(frames) > max_frame):
            video_stream.release()
            break
        frames.append(frame)

    frames = frames[:max_frame]
    batches = [frames[i:i + args.batch_size]
               for i in range(0, len(frames), args.batch_size)]
    print(f"video frames: {len(frames)}, batches: {len(batches)}")
    # print(f"video frames: {frames[3][0]}, batches: {batches[3][0]}")
    # print(frames[0])

    i = -1
    preds_cache = np.empty((len(frames), 4))
    preds_location = np.empty((len(frames), 6))

    ix1, iy1, ix2, iy2 = 0, 0, int(video_info.Width), int(video_info.Height)
    print("Detect video...")
    bar = tqdm(total=len(frames), disable=False, dynamic_ncols=True)
    for fb in batches:
        # print("input range:", ix1, iy1, ix2, iy2)
        ifb = [f[iy1:iy2+1, ix1:ix2+1] for f in fb]
        preds = fa[0].get_detections_for_batch(np.asarray(ifb))
        # preds = fa[0].get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            i += 1
            bar.update(1)

            if f is None:
                continue

            x1, y1, x2, y2 = f
            x1 += ix1
            x2 += ix1
            y1 += iy1
            y2 += iy1
            # print("process_video_file detected: {}, {}".format(i, f))
            preds_cache[i] = np.array([x1, y1, x2, y2])

        width1 = x2 - x1 + 1
        hight1 = y2 - y1 + 1
        diffx = (width1)//2 * 1.5
        diffy = (hight1)//2 * 1.5

        iix1 = int(x1 - diffx)
        iix2 = int(x2 + diffx)
        iiy1 = int(y1 - diffy)
        iiy2 = int(y2 + diffy)

        ix1 = int(max(iix1, 0))
        iy1 = int(max(iiy1, 0))
        ix2 = int(min(iix2, video_info.Width))
        iy2 = int(min(iiy2, video_info.Height))

    width = preds_cache[:, 2] - preds_cache[:, 0] + 1
    # print("process_video_file width:\n{}".format(width))
    hight = preds_cache[:, 3] - preds_cache[:, 1] + 1
    # print("process_video_file hight:\n{}".format(hight))

    square = int(max(width.max(), hight.max()) * 1.6)
    print("target talking head square: {}".format(square))

    for idx in range(len(frames)):
        x1, y1, x2, y2 = preds_cache[idx]
        if args.debug:
            print("process_video_file detected: {}, {}, {}, {}, {}".format(
                idx, x1, y1, x2, y2))

        width1 = width[idx]
        hight1 = hight[idx]
        diffx = (square - width1)//2
        diffy = (square - hight1)//2

        x1 = int(x1 - diffx)
        x2 = int(x2 + diffx)
        y1 = int(y1 - diffy)
        y2 = int(y2 + diffy)

        width2 = x2 - x1 + 1
        hight2 = y2 - y1 + 1

        if hight2 < square:
            y2 += square-hight2
        if width2 < square:
            x2 += square-width2

        # we may need to check square in images
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, video_info.Width)
        y2 = min(y2, video_info.Height)

        width3 = x2 - x1 + 1
        hight3 = y2 - y1 + 1

        if args.debug:
            print("process_video_file dealed: {}, {} -> [{} {} {} {}], {}x{} -> {}x{}".format(
                idx, preds_cache[idx], x1, y1, x2, y2, width1, hight1, width3, hight3))
        preds_location[idx] = np.array([x1, y1, x2, y2, width3, hight3])
        cv2.imwrite(path.join(head_image_dir, '{}.jpg'.format(idx)),
                    frames[idx][y1:y2+1, x1:x2+1])

    py_file = os.path.join(head_image_dir, "head_location.npy")
    with open(py_file, 'wb') as f:
        np.save(f, preds_location)


def export_process_video(args):

    if not os.path.isfile(args.input):
        print("{} is not exist or is not a file".format(args.input))
        return 1
    print("input video: {}".format(args.input))

    video_info = get_video_info(args.input)
    print("video info: {}".format(video_info))

    print('===>Convert videos...')
    video_info = convert_video(args, video_info)

    return 0, video_info


def export_image(args, video_info):
    print("video info: {}".format(video_info))

    # video_info = convert_video(args, video_info)

    # ret = export_origin_image(args, video_info)
    # if ret != 0:
    #     return ret

    print('===>Detect head...')
    process_video_file(args, video_info)


def export_lm(args, video_info):
    print("Video info: {}".format(video_info))

    video_info = green_screen_video(args, video_info)


def run_cmd(cmd):
    if isinstance(cmd, slice):
        cmd = " ".join(cmd)
    return_code = subprocess.call(cmd, shell=False)
    return return_code


def combine_or_export_video(audio_path, video_path, video_fps, image_format, video_input_image_path, head_image_dir=None, location_path=None, head_input_video=None, input_video_path=None):
    # au video tmp path
    base_path_splits = video_path.rsplit(".")
    base_path = base_path_splits[:len(base_path_splits)-1]
    output_tmp = ".".join(base_path) + "-au.mp4"
    
    # use_cli_create_au_video = None
    use_cli_create_au_video = False
    preds_location = np.zeros(0)
    if location_path:
        print("Head location path: {}".format(location_path))
        py_file = os.path.join(location_path, "head_location.npy")
        with open(py_file, 'rb') as f:
            preds_location = np.load(f)
        # print("preds_location: {}".format(preds_location))
        x1 = preds_location[:,0]
        x2 = preds_location[:,2]
        y1 = preds_location[:,1]
        y2 = preds_location[:,3]

        if use_cli_create_au_video is None and not head_image_dir and \
            x1.max() == x1.min() and x2.max() == x2.min() and \
            y1.max() == y1.min() and y2.max() == y2.min():
                use_cli_create_au_video = True
                x = int(x1.max())
                y = int(y1.max())
                width = int(preds_location[:,4].max())
                hight = int(preds_location[:,4].max())
                
                cmd = f'''ffmpeg -y -i {head_input_video} -i {input_video_path} '''\
                f'''-filter_complex "[0:v]chromakey=0x00FF00:0.4:0[p];[p]scale={width}:{hight}[pip];[1][pip]overlay=x={x}:y={y}" '''\
                f'''-shortest -r {args.fps}  {output_tmp}'''
                ret = run_cmd(cmd)
                if ret != 0:
                    raise Exception(f"run cmd failed, cmd: {cmd}")

    if not use_cli_create_au_video:
        print(f"target video path: {video_path}, video input image path: {video_input_image_path}, head input image path: {head_image_dir}")
        if head_image_dir:
            image_path = head_image_dir
        else:
            image_path = video_input_image_path
        image_format = "jpg"
        image_names, _ = inference_utils.get_file_list(image_path, image_format, True)
        # image_names = image_names[:500]
        print(f"input origin image frame num: {len(image_names)}, over image frame num: {len(preds_location)}")

        def get_video_writer(numpy_data):
            size = (numpy_data.shape[1], numpy_data.shape[0])
            print(f"write video to {output_tmp}, size: {size}, numpy_data shape: {numpy_data.shape}")

            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            return cv2.VideoWriter(output_tmp, fourcc, float(video_fps), size)

        video_writer = None
        bar = tqdm(total=len(image_names), disable=False, dynamic_ncols=True, desc='CV2Video')
        for idx in range(len(image_names)):
            name = image_names[idx]
            numpy_data = None
            numpy_over_data = None

            # 1. video main input image
            origin_image_path = os.path.join(video_input_image_path, f"{name}.{image_format}")
            img = cv2.imread(origin_image_path)
            # data = np.random.randint(0,255, (height, width, channel), dtype = np.uint8)
            numpy_data = np.asarray(img)
            # print(f"numpydata {numpy_data[:10]}")
            # print(f"numpydata {numpy_data.shape}")

            # 2. video over image
            if head_image_dir:
                # x1, y1, x2, y2, width3, hight3
                idx_info = preds_location[idx]
                if args.debug:
                    print(f"preds_location: {idx}, {idx_info}")
                [x1, y1, x2, y2, width, hight] = idx_info
                
                origin_image_over_path = os.path.join(head_image_dir, f"{name}.{image_format}")
                img_over = cv2.imread(origin_image_over_path)
                if img_over.shape[0] != width or img_over.shape[1] != hight:
                    img_size=(int(width),  int(hight))
                    img_over = cv2.resize(img_over, img_size, interpolation=cv2.INTER_NEAREST)
            
                numpy_over_data = np.asarray(img_over)
                # print(f"numpy_over_data {numpy_over_data[:10]}")
                # print(f"numpy_over_data {numpy_over_data.shape}")
                
                # lower_green = np.array([0, 250, 0])
                # upper_green = np.array([0, 255, 0])
                # lower_green = np.array([45, 100, 50])
                # upper_green = np.array([75, 255, 255])
                lower_green = np.array([35, 43, 46])
                upper_green = np.array([77, 255, 255])
                hsv = cv2.cvtColor(img_over, cv2.COLOR_BGR2HSV)
                mask_lm = cv2.inRange(hsv, lower_green, upper_green)
                mask = cv2.bitwise_not(mask_lm)       
                        
                # index_mask = np.where(mask)
                # index_min_x = index_mask[0].min()
                # # print(mask[:,index_min_y:index_min_y+10])
                # mask[index_min_x:index_min_x+100,:] = 0

                result = cv2.bitwise_and(img_over, img_over, mask=mask)
                numpy_over_data = np.asarray(result)
                # numpy_over_data.resize()
                
                # cv2.imshow("cc", mask)
                # import time
                # time.sleep(3)
                # print(f"mask: {mask.shape} {mask[0]}")
                index = np.where(numpy_over_data != np.array([0, 0, 0]))
                # print(f"lm index: \n{numpy_over_data.shape} {index[0]}\n{index[1]}\n{index[2]} ")
                # print(f"numpy_over_data[10][10] {numpy_over_data[10][10]}")


                # print(f"idx_info: {idx_info}, numpy_over_data: {len(numpy_over_data)}*{len(numpy_over_data[0])}")
                # numpy_data[int(y1):int(y2)+1, int(x1):int(x2)+1] = numpy_over_data
                # print(type(index[0]), type(y1))
                new_idx = (index[0]+int(y1), index[1]+int(x1), index[2])
                numpy_data[new_idx] = numpy_over_data[index]

                # numpy_data[mask[0]+int(y1)][mask[1]+int(x1)] = numpy_over_data[mask[0]][mask[1]]

            if video_writer is None:
                video_writer = get_video_writer(numpy_data)
            video_writer.write(numpy_data)
            bar.update(1)
        if video_writer is not None:
            video_writer.release()

    # return
    video = ffmpeg.input(output_tmp).video
    audio = ffmpeg.input(audio_path).audio
    # ffmpeg.concat(video, audio, v=1, a=1).output(
    #     args.export_output,
    #     audio_bitrate=video_info.AudioBitrate, 
    #     video_bitrate=video_info.Bitrate, 
    #     crf=18, # 压缩比，一般18~28。-qscale is ignored, -crf is recommended
    #     ac=1,
    # ).overwrite_output().run()
    kwargs = {
        "loglevel": "warning",
        "vcodec": "copy",
        # acodec:"copy",
        # audio_bitrate="copy", video_bitrate="copy", 
        # audio_bitrate=video_info.AudioBitrate, video_bitrate=video_info.Bitrate, 
        "crf": 18, # 压缩比，一般18~28。-qscale is ignored, -crf is recommended
        # ac: 1,
    }
    if not audio_path.endswith(".wav"):
        kwargs["acodec"] = "copy"
    ffmpeg.output(
        video, audio, video_path, **kwargs
    ).overwrite_output().run()


def export_video(args, video_info):
    print('===>Exporting videos...')
    
    input_image_path = None
    if args.head_image_lm:
        input_image_path = args.head_image_lm
    elif args.head_image:
        input_image_path = args.head_image

    image_format = "jpg"
    combine_or_export_video(video_info.Path, args.export_output, args.fps, image_format, input_image_path, None)


def combine_video(args, video_info):
    print('===>Combine videos...')
    
    # export video image to images
    head_input_video = video_utils.get_file_name(args.combine_input)
    print("Input head video path: {}".format(head_input_video))
    stream = ffmpeg.input(head_input_video)
    # stream = stream.filter('fps', fps='1/%d' % args.fps)
    # head_output_dir = os.path.join(os.path.dirname(head_input_video), "head_images_new")
    head_output_dir = os.path.join(os.path.dirname(head_input_video), "head_images")
    os.makedirs(head_output_dir, exist_ok=True)
    clear_dir(head_output_dir)
    head_output_format = os.path.join(head_output_dir, '%d.jpg')
    stream.output(head_output_format, start_number=0).overwrite_output().run(quiet=True)

    image_format = "jpg"
    combine_or_export_video(args.combine_input_audio, args.combine_output, args.fps, image_format, args.origin_image, head_output_dir, args.head_image, head_input_video, video_info.Path)


def main(args):
    ret = 0
    if args.mode == ["all"]:
        args.mode = mode_choice[1:]

    print(f"current args: {args}")

    if not os.path.isfile(args.input):
        print("Input video {} is not exist or is not a file".format(args.input))
        return 1
    print("Input video: {}".format(args.input))

    video_info = get_video_info(args.input)
    print("video info: {}".format(video_info))

    if "convert" in args.mode:
        args.overwrite = True

    ret, video_info = export_process_video(args)
    if ret and ret != 0:
        sys.exit(ret)

    if "origin_image" in args.mode:
        ret = export_origin_image(args, video_info)
        if ret and ret != 0:
            sys.exit(ret)
    if "image" in args.mode:
        ret = export_image(args, video_info)
        if ret and ret != 0:
            sys.exit(ret)
    if "lm" in args.mode:
        ret = export_lm(args, video_info)
        if ret and ret != 0:
            sys.exit(ret)
    if "export" in args.mode:
        ret = export_video(args, video_info)
        if ret and ret != 0:
            sys.exit(ret)
    if "combine" in args.mode:
        ret = combine_video(args, video_info)
        if ret and ret != 0:
            sys.exit(ret)
    sys.exit(ret)


if __name__ == '__main__':
    main(args)
