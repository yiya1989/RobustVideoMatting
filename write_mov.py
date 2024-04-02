import av
import numpy as np


def generate_video(input_video_path, format, video_encode, pix_fmt, data_fmt, options:dict={}):
    # 创建一个输出容器
    output_video_base = str(input_video_path).rsplit(".", 2)[0]
    output_video_path = f"{output_video_base}-{video_encode}-{pix_fmt}.{format}"
    container = av.open(output_video_path, 'w')
    # container.format = "matroska"

    # 创建一个视频流
    video_stream = container.add_stream(video_encode, rate=30)

    # 设置视频流的分辨率和比特率
    video_stream.width = 1080
    video_stream.height = 1920
    video_stream.bit_rate = 1000000

    # 设置像素格式为yuva444p10le
    video_stream.pix_fmt = pix_fmt

    # 设置编码器配置参数
    video_stream.options.update(options)

    # 打开输入的视频文件
    input_video = av.open(input_video_path)

    # 遍历输入视频的帧并写入输出视频
    for frame in input_video.decode(video=0):
        # 编码帧
        frame = frame.reformat(width=video_stream.width, height=video_stream.height)
        
        # 将帧转换为RGBA格式
        frame = frame.reformat(format='rgb24')
        rgba_frame = np.array(frame.to_image())
        # print(f"frame: {rgba_frame.shape}, {rgba_frame[0,0]}")
        
        # 将透明通道添加到每个像素值中
        alpha_channel = np.full((video_stream.height, video_stream.width, 1), 128, dtype=np.uint8)
        # print(f"rgba_frame: {rgba_frame.shape}, alpha_channel: {alpha_channel.shape}")
        rgba_frame = np.concatenate((rgba_frame, alpha_channel), axis=2)
        # new_frame.planes[3] = av.VideoFrame.from_ndarray(alpha_channel, format='yuv444p10le')
        # print(f"rgba_frame new: {rgba_frame.shape}")

        # print(f"rgba_frame: {rgba_frame.shape}, {rgba_frame[0,0]}")
        # 创建一个新的帧并复制像素值
        new_frame = av.VideoFrame.from_ndarray(rgba_frame, format=data_fmt)
        
        # 编码新的帧
        new_frame = video_stream.encode(new_frame)
        
        # 写入输出视频
        container.mux(new_frame)

    # 关闭输入视频和输出容器
    input_video.close()
    container.close()


def main():
    data = [
        ["mov", "png", "rgba", "rgba", {"pred": "4"}],
        # ["mov", "dnxhr", "yuva422p", "yuva422p", {}],
        # ["mov", "prores_ks", "yuva444p10le", "rgba", 
        #     {
        #         "vendor": "apl0", 
        #         "profile": "4444",
        #         "mbs_per_slice": "8",
        #         "alpha_bits": "16",
        #         "quant_mat": "proxy", 
        #         "bits_per_mb": "1000"
        #     }
        # ],
        # ["mov", "qtrle", "argb", "rgba", {}],
        # ["webm", "libvpx-vp9", "yuva420p", "rgba", {}],
    ]

    for item in data:
        generate_video(
            "D:/code/data/test1.mov",
            item[0],
            item[1],
            item[2],
            item[3], 
            item[4],
        )
        print(f"generate {item[0]} {item[1]} {item[2]} done")



if __name__ == "__main__":
    main()