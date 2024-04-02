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

    parser.add_argument("--audio", help="audio file or dir path", default="", required=True)
    parser.add_argument("--audio-format", help="audio format", default="mp3", required=False)
    parser.add_argument("--srt", help="srt file or dir path", default="", required=False)
    parser.add_argument("--out-format", help="output audio format", default="wav", required=False)
    parser.add_argument("--out-audio", help="output dir path", default="output", required=True)
    parser.add_argument("--out-asr", help="output asr file path", default="", required=True)
    parser.add_argument("--out-role", help="output role and role dir name", default="", required=True)
    parser.add_argument("--min", help="min audio second", default=3, type=int, required=False)
    parser.add_argument("--max", help="max audio second", default=10, type=int, required=False)

    args = parser.parse_args()

    return args


def pare_dir(path, clear=True):
    if os.path.isfile(path):
        os.remove(path)
        os.makedirs(path, exist_ok=True)
    elif not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    elif clear:
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


def write_asr(chunk_idx, esf_file, role, audio_text):
    #./data/ada/wavs/ada_42.wav|ada|ZH|在德林哈一夜的結尾處
    line = f"./output/{role}/slicer_opt/{role}_{chunk_idx}.wav|{role}|ZH|{audio_text}\n"
    dir_name = os.path.dirname(esf_file)
    pare_dir(dir_name, clear=False)
    # if not os.path.exists(dir_name):
    #     os.makedirs(dir_name, exist_ok=True)
    if chunk_idx == 0:
        with open(esf_file, "w+", encoding="utf-8") as fp:
            fp.write(line)
    else:
        with open(esf_file, "a+", encoding="utf-8") as fp:
            fp.write(line)
    

def get_second(duration):
    duration_delta = datetime.timedelta(
        hours=duration.hour, minutes=duration.minute, seconds=duration.second, microseconds=duration.microsecond)
    duration_second = duration_delta.total_seconds()
    return duration_second


def parse_audio(args, srt_file, chunk_idx, target_audio_output_dir):
    srt = pysrt.open(srt_file)

    # show_one_srt(srt.data[0])

    audio_name = os.path.basename(srt_file).split(".")[0]
    audio_format = args.audio_format
    audio_file_path = os.path.join(os.path.dirname(srt_file), f"{audio_name}.{audio_format}")
    if args.out_role:
        target_audio_role = args.out_role
    else: 
        target_audio_role = audio_name
    print(f"==> dealing srt file: {srt_file}, audio file: {audio_file_path}...")
    audio = AudioSegment.from_file(audio_file_path, format=audio_format)
    
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
        
        chunk_file = os.path.join(target_audio_output_dir, f"{target_audio_role}_{chunk_idx}.{output_format}")
        
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
            write_asr(chunk_idx, args.out_asr, target_audio_role, audio_text)
            audio_start = -1
            chunk_idx += 1
            audio_text = ""
   
    return chunk_idx

   
def main(args):
    
    chunk_idx = 0

    target_audio_output_dir = args.out_audio
    pare_dir(target_audio_output_dir)
    
    if os.path.isdir(args.audio):
        args.srt = args.audio
    else:
        print("===>", args.audio.split(".")[:-1])
        args.srt = ".".join(args.audio.split(".")[:-1]) + ".srt"
        args.audio_format = args.audio.split(".")[-1]
        
    
    if os.path.isdir(args.srt):
        _, image_path_list = inference_utils.get_file_list(args.srt, "srt", True)
    else:
        image_path_list = [args.srt]

    print(f"current args: {args}")
    for srt_file in image_path_list:
        chunk_idx = parse_audio(args, srt_file,  chunk_idx, target_audio_output_dir)

    
    
if __name__ == '__main__':
    # PS D:\lch\code\RobustVideoMatting> python.exe .\audio_slicer.py --audio "D:\360安全浏览器下载\小宝克隆\小宝克隆.WAV" --out-audio D:\lch\code\GPT-SoVITS-beta\GPT-SoVITS-beta0217\output\xiaobao\slicer_opt --out-asr D:\lch\code\GPT-SoVITS-beta\GPT-SoVITS-beta0217\output\xiaobao\asr_opt\slicer_opt.list --out-role xiaobao
    args = parse_args()
    print(f"Input args: {args}")
    # args.srt = "D:/lch/audio/1-10/1.srt"
    # args.srt = "srt_file.srt"
    main(args)
