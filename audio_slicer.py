import os
import pysrt
import argparse
import datetime
import glob

from pydub import AudioSegment
from pydub.utils import make_chunks

import inference_utils

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--srt", help="srt file or dir path", default="", required=True)
    parser.add_argument("--audio-format", help="audio format", default="mp3", required=False)
    parser.add_argument("--out-format", help="output audio format", default="wav", required=False)
    parser.add_argument("--out-dir", help="output dir path", default="output", required=False)
    parser.add_argument("--min", help="min audio second", default=3, type=int, required=False)
    parser.add_argument("--max", help="max audio second", default=10, type=int, required=False)
    parser.add_argument("--role-path", help="role path", default="./data/ada", type=str, required=True)

    args = parser.parse_args()

    return args


def clear_and_pare_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    else:
        files = glob.glob(os.path.join(path, "*"))
        for f in files:
            if os.path.isfile(f):
                os.remove(f)


def show_one_srt(content):
    characters_per_second = content.characters_per_second
    duration = content.duration
    end = content.end
    # from_lines = content.from_lines
    # from_string = content.from_string
    index = content.index
    position = content.position
    shift = content.shift
    split_timestamps = content.split_timestamps
    start = content.start
    text = content.text
    text_without_tags = content.text_without_tags

    print(f"characters_per_second: {characters_per_second}")
    print(f"duration: {duration} {type(duration.to_time().second)}")
    print(f"start: {start} {type(start)}")
    print(f"end: {end} {type(end)}")
    print(f"text: {text}")
    print(f"text_without_tags: {text_without_tags}")
    # print(f"from_lines: {from_lines}")
    # print(f"from_string:{from_string}")
    print(f"index: {index}")
    print(f"position: {position}")
    print(f"shift: {shift}")
    print(f"split_timestamps: {split_timestamps}")


def write_audio(audio, output_file, start, end, format='wav'):
    chunk = audio[int(start*1000):int(end*1000)+1]
    chunk.export(output_file, format=format)


def write_esd(chunk_idx, role_path, audio_text):
    #./data/ada/wavs/ada_42.wav|ada|ZH|在德林哈一夜的結尾處
    esf_file = os.path.join(role_path, "esd.list")
    role = os.path.basename(role_path.rstrip("/\\"))
    line = f"./data/{role}/wavs/{role}_{chunk_idx}.wav|{role}|ZH|{audio_text}\n"
    if chunk_idx == 0:
        with open(esf_file, "w+") as fp:
            fp.write(line)
    else:
        with open(esf_file, "a+") as fp:
            fp.write(line)
    

def get_second(duration):
    duration_delta = datetime.timedelta(
        hours=duration.hour, minutes=duration.minute, seconds=duration.second, microseconds=duration.microsecond)
    duration_second = duration_delta.total_seconds()
    return duration_second


def parse_audio(args, srt_file, chunk_idx):
    srt = pysrt.open(srt_file)

    # show_one_srt(srt.data[0])

    
    audio_name = os.path.basename(srt_file).split(".")[0]
    audio_format = args.audio_format
    audio_file_path = os.path.join(os.path.dirname(srt_file), f"{audio_name}.{audio_format}")
    if args.role_path:
        target_audio_name = os.path.basename(args.role_path.rstrip("/\\"))
    else: 
        target_audio_name = audio_name
    print(f"==> dealing srt file: {srt_file}, audio file: {audio_file_path}...")
    audio = AudioSegment.from_file(audio_file_path, format=audio_format)
    
    clear_and_pare_dir(args.out_dir)
    output_format = args.out_format
    
    length = 0
    audio_start = -1
    audio_text = ""
    for item in srt.data:
        content = item

        index = content.index
        start_time = content.start.to_time().isoformat("milliseconds")
        end_time = content.end.to_time().isoformat("milliseconds")
        duration = content.duration.to_time()
        duration_second = get_second(duration)
        text = content.text
        
        chunk_file = os.path.join(args.out_dir, f"{target_audio_name}_{chunk_idx}.{output_format}")
        
        # print("index: {:4d}, time: {} -> {}, duration: {:3.3f}, text: {}".format(index, start_time, end_time, duration_second, text))
        
        if audio_start == -1:
            audio_start = get_second(content.start.to_time())
        end = get_second(content.end.to_time())
        audio_text = audio_text + text + ", "
        if end - audio_start < args.min:
            continue
        else:
            print("chunk_idx: {:4d}, chunk_file: {},\t duration: {:3.3f}, audio_text: {}".format(chunk_idx, chunk_file, end - audio_start, audio_text))
            write_audio(audio, chunk_file, audio_start, end, format=output_format)
            write_esd(chunk_idx, args.role_path, audio_text)
            audio_start = -1
            chunk_idx += 1
            audio_text = ""
   
    return chunk_idx

   
def main(args):
    
    chunk_idx = 0
    
    if os.path.isdir(args.srt):
        _, image_path_list = inference_utils.get_file_list(args.srt, "srt", True)
    else:
        image_path_list = [args.srt]
    
    for srt_file in image_path_list:
        chunk_idx = parse_audio(args, srt_file,  chunk_idx)

    
    
if __name__ == '__main__':
    # (nerf) PS D:\lch\code> python .\slicer.py --audio "D:/lch/audio/1-10/1.mp3" --srt "D:/lch/audio/1.srt" 
    args = parse_args()
    print(f"Input args: {args}")
    # args.srt = "D:/lch/audio/1-10/1.srt"
    # args.srt = "srt_file.srt"
    main(args)
