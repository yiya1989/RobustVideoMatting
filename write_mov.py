import av
import numpy as np

# 创建一个输出容器
container = av.open('C:/data/test2-m.mov', 'w')

# 创建一个视频流
video_stream = container.add_stream('prores_ks', rate=30)
# video_stream = container.add_stream('png', rate=30)

# 设置视频流的分辨率和比特率
video_stream.width = 600
video_stream.height = 800
video_stream.bit_rate = 2000000

# video_stream.pix_fmt = 'rgba'
# 设置像素格式为yuva444p10le
video_stream.pix_fmt = 'yuva444p10le'

# 设置编码器配置参数
video_stream.options = {'profile': '4444', 'vendor': 'ap10'}

# 打开输入的视频文件
input_video = av.open("C:/data/test2.mov")

# 遍历输入视频的帧并写入输出视频
for frame in input_video.decode(video=0):
    # 编码帧
    frame = frame.reformat(width=video_stream.width, height=video_stream.height)
    
    # 将帧转换为RGBA格式
    frame = frame.reformat(format='rgba')
    rgba_frame = np.array(frame.to_image())
    
    # 将透明通道添加到每个像素值中
    alpha_channel = np.full((video_stream.height, video_stream.width, 1), 240, dtype=np.uint8)
    print(f"rgba_frame: {rgba_frame.shape}, alpha_channel: {alpha_channel.shape}")
    rgba_frame = np.concatenate(
        (rgba_frame, alpha_channel), 
        axis=2
    )
    # new_frame.planes[3] = av.VideoFrame.from_ndarray(alpha_channel, format='yuv444p10le')
    print(f"rgba_frame new: {rgba_frame.shape}")

    
    # 创建一个新的帧并复制像素值
    new_frame = av.VideoFrame.from_ndarray(rgba_frame, format='rgba')
    
    # 编码新的帧
    new_frame = video_stream.encode(new_frame)
    
    # 写入输出视频
    container.mux(new_frame)

# 关闭输入视频和输出容器
input_video.close()
container.close()