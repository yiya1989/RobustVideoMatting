import urllib.request
from urllib.parse import quote
import os
import re
import random
import librosa
import shutil
import imgkit
from html2image import Html2Image
import subprocess 
concat ="./concat.txt"
if os.path.exists(concat):
       os.remove(concat) 

def count_lines_in_file(filename):  
    """  
    计算文件中的行数。  
  
    :param filename: 要读取的文件名  
    :return: 文件中的行数  
    """  
    try:  
        with open(filename, 'r', encoding='utf-8') as file:  
            lines = file.readlines()  
            return len(lines)  
    except FileNotFoundError:  
        print(f"文件 {filename} 未找到。")  
        return 0  
  
def random_line_from_file(filename, min_lines=1):  
    """  
    从文件中随机读取一行，并确保文件至少有min_lines行。  
  
    :param filename: 要读取的文件名  
    :param min_lines: 文件应包含的最少行数  
    :return: 随机选择的一行文本  
    """  
    try:  
        with open(filename, 'r', encoding='utf-8') as file:  
            lines = file.readlines()  
    except FileNotFoundError:  
        print(f"文件 {filename} 未找到。")  
        return None  
  
    if len(lines) < min_lines:  
        print(f"文件 {filename} 中的行数不足 {min_lines} 行。")  
        return None  
  
    # 从文件中随机选择一行  
    random_line = random.choice(lines)  
  
    # 去除行尾的换行符并返回结果  
    return random_line.strip()  
def get_duration(filename):  
    """获取媒体文件的时长（秒）"""  
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', filename],  
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)  
    duration = float(result.stdout)  
    return duration 
    

with open('text.txt', 'r', encoding='gbk') as f:  
    # 按行读取文件  
    c=0
    for line in f:  
        # 去除每行末尾的换行符并打印  
        a=line.strip()

        c=c+1
        
        # 使用函数读取文件中的随机行  
        filename = 'E:/GPT-SoVITS-0310-liuyue/output/xbf/asr_opt/'+'slicer_opt.list'  
        min_required_lines =count_lines_in_file(filename)  
        #min_required_lines=100
        random_line = random_line_from_file(filename, min_required_lines)  
        print(random_line)     
        parts = random_line.split("|")
        cktxt = parts[-1] 
        ckwav = parts[0]   
        ckwav = ckwav.replace("./","E:/GPT-SoVITS-0310-liuyue/")
        video_duration = get_duration(ckwav) 
        if video_duration < 3 or video_duration > 10:  
       # 如果初始的ckwav对应的音频时长不满足条件，则进入循环  
            while True:  
                random_line = random_line_from_file(filename, min_required_lines)  
                print(random_line)  
                parts = random_line.split("|")  
                cktxt = parts[-1]  
                ckwav = parts[0].replace("./", "E:/GPT-SoVITS-0310-liuyue/")  
                video_duration = get_duration(ckwav)  # 更新音频时长  
                if 3 <= video_duration <= 10:  # 检查新的音频时长是否满足条件  
                    break  # 如果满足条件，则退出循环     

        
    
        url = 'http://localhost:9880?text=' + quote(a) + '&text_lang=' + quote("中文") + '&ref_audio_path='+ckwav+'&prompt_text=' + quote(cktxt) + '&prompt_lang=' + quote("中文") + '&text_split_method=' + quote("按中文句号。切")+'&sweight=SoVITS_weights/bbb_e8_s576.pth&gweight=GPT_weights/bbb-e50.ckpt&speed_factor=1'
        
        #url = 'http://localhost:9880?text=' + quote(zimu) + '&text_lang=' + quote("中文") + '&ref_audio_path='+ckwav+'&prompt_text=' + quote(cktxt) + '&prompt_lang=' + quote("中文") + '&text_split_method=' + quote("按中文句号。切")+'&sweight=SoVITS_weights/yq_e8_s152.pth&gweight=GPT_weights/yq-e50.ckpt&speed_factor=1'  
        #print(url)
        
        #url = 'http://localhost:9880?text=' + quote(a) + '&text_lang=' + quote("中文") + '&ref_audio_path=E:/GPT-SoVITS-0310-liuyue/output/cankao.wav&prompt_text=' + quote("首先,感谢您在百忙之中来听小宝说书,真是太给面子了") + '&prompt_lang=' + quote("中文") + '&text_split_method=' + quote("按中文句号。切")+'&sweight=SoVITS_weights/bbb_e8_s576.pth&gweight=GPT_weights/bbb-e50.ckpt&speed_factor=1.1'
        print(url)
        save_path ='./'             #文件下载目录
        filename = save_path +str(c)+'.wav'  #文件名
        if os.path.exists(filename):
           os.remove(filename)   
        urllib.request.urlretrieve(url, filename)
        
        concatTextFile = open(concat, "a+", encoding='utf-8')
        concatTextFile.write("file "+str(c)+".wav" + "\n")
        concatTextFile.close() 
out="output.wav"        
if os.path.exists(out):
  os.remove(out)         

cmd = "ffmpeg -f concat -safe 0 -i {} -c copy {}".format(concat, out)
os.system(cmd)