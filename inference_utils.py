import av
import os
import pims
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from PIL import Image


class VideoReader(Dataset):
    def __init__(self, path, transform=None):
        self.video = pims.PyAVVideoReader(path)
        self.rate = self.video.frame_rate
        self.transform = transform
        
    @property
    def frame_rate(self):
        return self.rate
        
    def __len__(self):
        return len(self.video)
        
    def __getitem__(self, idx):
        frame = self.video[idx]
        frame = Image.fromarray(np.asarray(frame))
        if self.transform is not None:
            frame = self.transform(frame)
        return frame


class VideoWriter:
    def __init__(self, path, frame_rate, bit_rate=1000000):
        self.container = av.open(path, mode='w')
        self.stream = self.container.add_stream('h264', rate=f'{frame_rate:.4f}')
        self.stream.pix_fmt = 'yuv420p'
        self.stream.bit_rate = bit_rate
    
    def write(self, frames):
        # frames: [T, C, H, W]
        self.stream.width = frames.size(3)
        self.stream.height = frames.size(2)
        if frames.size(1) == 1:
            frames = frames.repeat(1, 3, 1, 1) # convert grayscale to RGB
        frames = frames.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
        for t in range(frames.shape[0]):
            frame = frames[t]
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            self.container.mux(self.stream.encode(frame))
                
    def close(self):
        self.container.mux(self.stream.encode())
        self.container.close()

class ImageSequenceReader(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        _, image_list = get_image_list(path, "jpg")
        self.files = image_list
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        with Image.open(os.path.join(self.path, self.files[idx])) as img:
            img.load()
        if self.transform is not None:
            return self.transform(img)
        return img


class ImageSequenceWriter:
    def __init__(self, path, extension='jpg'):
        self.path = path
        self.extension = extension
        self.counter = 0
        os.makedirs(path, exist_ok=True)
    
    def write(self, frames):
        # frames: [T, C, H, W]
        for t in range(frames.shape[0]):
            to_pil_image(frames[t]).save(os.path.join(
                self.path, str(self.counter) + '.' + self.extension))
                # self.path, str(self.counter).zfill(4) + '.' + self.extension))
            self.counter += 1
            
    def close(self):
        pass
        

class ImageNPArrayReader(Dataset):
    def __init__(self, input_source, transform=None):
        self.input_source = input_source
        self.transform = transform

    def __len__(self):
        return len(self.input_source)

    def __getitem__(self, idx):
        with Image.open(os.path.join(self.path, self.files[idx])) as img:
            img.load()
        if self.transform is not None:
            return self.transform(img)
        return img


def get_image_list(path, image_format="jpg", with_path=False):
    if not os.path.isdir(path):
        raise Exception(f"input path is not dir, path: {path}")
    image_name_list = []
    image_list = []
    filelist = os.listdir(path)
    image_names= [int(name[:-(len(image_format)+1)]) for name in filelist if name.endswith(image_format)]
    image_names.sort()
    # print(f"image list in path {path}/*.{image_format}, image_names: {image_names}")
    for name in image_names:
        if with_path:
            image_path = os.path.join(path, f"{name}.{image_format}")
        else:
            image_path = f"{name}.{image_format}"
        image_name_list.append(name)
        image_list.append(image_path)
    return image_name_list, image_list

