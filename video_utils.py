
import ffmpeg
import json
import sys
import glob


def get_file_name(input):
    file_list = glob.glob(input)
    file_list.sort()
    file_list = [n for n in file_list if "_depth" not in n]
    
    return file_list[-1]
    

class VideoInfo:
    Path = ""
    Width = 0
    Height = 0
    FPS = 0
    NumFrames = 0
    DurationInSeconds = 0
    Bitrate = 0
    Codec = ""
    AudioBitrate = 0
    AudioChannels = 0
    AudioSampleRate = 0
    AudioSampleRate = 0
    AudioCodec = ""

    def __init__(self, path):
        self.Path = path

        self.load()

    def __str__(self) -> str:
        return f'Video path: {self.Path}, width:{self.Width}, height:{self.Height}, fps:{self.FPS}, '\
            f'num_frames:{self.NumFrames}, duration_in_seconds:{self.DurationInSeconds}, '\
            f'bitrate:{self.Bitrate}, codec:{self.Codec}, '\
            f'audio_bitrate:{self.AudioBitrate}, audio_channels:{self.AudioChannels}, '\
            f'audio_sample_rate:{self.AudioSampleRate}, audio_codec:{self.AudioCodec}'

    def load(self):
        probe = ffmpeg.probe(self.Path)

        # print(json.dumps(probe))
        video_info = {}
        audio_info = {}
        for s in probe['streams']:
            if s['codec_type'] == 'video':
                video_info = s
            if s['codec_type'] == 'audio':
                audio_info = s

        # print(f"video_info: {video_info}, audio_info: {audio_info}")

        self.Width = int(video_info['width'])
        self.Height = int(video_info['height'])
        self.FPS = int(video_info['r_frame_rate'].split('/')[0])
        self.NumFrames = int(video_info['nb_frames'])
        self.DurationInSeconds = float(video_info['duration'])
        self.Bitrate = int(video_info['bit_rate'])
        self.Codec = video_info['codec_name']
        self.AudioBitrate = int(audio_info.get('bit_rate', '0'))
        self.AudioChannels = int(audio_info.get('channels', '0'))
        self.AudioSampleRate = int(audio_info.get('sample_rate', '0'))
        self.AudioCodec = audio_info.get('codec_name', "")


if __name__ == "__main__":
    filename = sys.argv[1]
    print(f"filename: {filename}")

    video_info = VideoInfo(filename)
    print(f"video_info: {video_info}")
