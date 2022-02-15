import argparse
import os
import shutil
import threading
from pickletools import uint8
from time import sleep
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from sympy import im
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

import create
from inference_utils import (ImageSequenceReader, ImageSequenceWriter,
                             VideoReader, VideoWriter)
from model import MattingNetwork


def convert_video(model,
                  input_source: str,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = "png_sequence",
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None):
    
    assert downsample_ratio is None or (downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha, output_foreground]), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
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
            writer_com = ImageSequenceWriter(output_composition, 'png')
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, 'png')
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, 'png')

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
    
    if (output_composition is not None) and (output_type == 'video'):
        bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
    
    try:
        with torch.no_grad():
            bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
            rec = [None] * 4
            for src in reader:

                if downsample_ratio is None:
                    downsample_ratio = auto_downsample_ratio(*src.shape[2:])

                src = src.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
                fgr, pha, *rec = model(src, *rec, downsample_ratio)

                if output_foreground is not None:
                    writer_fgr.write(fgr[0])
                if output_alpha is not None:
                    writer_pha.write(pha[0])
                if output_composition is not None:
                    if output_type == 'video':
                        com = fgr * pha + bgr * (1 - pha)
                    else:
                        fgr = fgr * pha.gt(0)
                        com = torch.cat([fgr, pha], dim=-3)
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
    

def img_float32(img):
    return img.copy() if img.dtype != 'uint8' else (img/255.).astype('float32')

# 新的图片合成算法
def over(bgimg, fgimg):
    if bgimg[0][0].size==3:
        bgimg=cv2.cvtColor(bgimg,cv2.COLOR_BGR2BGRA)
    if fgimg[0][0].size==3:
        fgimg=cv2.cvtColor(fgimg,cv2.COLOR_BGR2BGRA)
    fgimg, bgimg = img_float32(fgimg),img_float32(bgimg)
    (fb,fg,fr,fa),(bb,bg,br,ba) = cv2.split(fgimg),cv2.split(bgimg)
    color_fg, color_bg = cv2.merge((fb,fg,fr)), cv2.merge((bb,bg,br))
    alpha_fg, alpha_bg = np.expand_dims(fa, axis=-1), np.expand_dims(ba, axis=-1)
    
    color_fg[fa==0]=[0,0,0]
    color_bg[ba==0]=[0,0,0]
    
    a = fa + ba * (1-fa)
    a[a==0]=np.NaN
    color_over = (color_fg * alpha_fg + color_bg * alpha_bg * (1-alpha_fg)) / np.expand_dims(a, axis=-1)
    color_over = np.clip(color_over,0,1)
    color_over[a==0] = [0,0,0]
    
    result_float32 = np.append(color_over, np.expand_dims(a, axis=-1), axis = -1)
    return (result_float32*255).astype('uint8')

# 寻找无效关键点
def find(kp,img_a,x_b,x_e,y_b,y_e):
    new_kp=[]
    for i in kp:
        p=i.pt
        x=int(p[0])
        y=int(p[1])
        s = sum(img_a[y][x])
        if s == 0:
            if not(x_b[0]<x<x_e[0] and y_b[0]<y<y_e[0]):
                if not(x_b[1]<x<x_e[1] and y_b[1]<y<y_e[1]):
                    new_kp.append(i)
        
    return new_kp

#鼠标回调函数
def draw_rectangle(event,x,y,flags,param):
    global n,ix,iy,a,x_b,x_e,y_b,y_e
    if event==cv2.EVENT_LBUTTONDOWN :
        if n == 0:#首次按下保存坐标值
            n+=1
            ix,iy = x,y
            a+=1
            cv2.circle(img,(x,y),2,(255,255,255),-1)#第一次打点
        else:#第二次按下显示矩形
            n = 0
            cv2.rectangle(img,(ix,iy),(x,y),(255,255,255),2)#第二次画矩形
            x_b[a-1]=int(min(ix,x))
            x_e[a-1]=int(max(ix,x))
            y_b[a-1]=int(min(iy,y))
            y_e[a-1]=int(max(iy,y))

# main
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, required=True,choices=['cuda','cpu'])
parser.add_argument('--input-source', type=str, required=True)
parser.add_argument('--capture-time',type=float,required=True,nargs=2)
parser.add_argument('--times',type=int,default=5)
parser.add_argument('--area-num',type=int,default=0,choices=[0,1,2])
args = parser.parse_args()

class myThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        if os.path.isdir("output_a"):
            shutil.rmtree("output_a")
        else:
            os.makedirs("output_a")
        converter = Converter("mobilenetv3", "rvm_mobilenetv3.pth", args.device)
        converter.convert(
            input_source=args.input_source,
            input_resize=None,
            downsample_ratio=None,
            output_type="png_sequence",
            output_composition="output_a",
            output_alpha=None,
            output_foreground=None,
            output_video_mbps=None,
            seq_chunk=1,
            num_workers=0,
            progress=True
        )

thread1 = myThread()
thread1.start()
thread1.join()

class allcapture (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        cap = cv2.VideoCapture(args.input_source)# 利用VideoCapture捕获视频，这里使用本地视频
        # 创建文件用来保存视频帧
        if os.path.isdir("output"):
            shutil.rmtree("output")
            os.makedirs("output")
        else:
            os.makedirs("output")
        imgPath = ""# 截图的图片命名
        f_sum = 0
        while True:
            ret, frame = cap.read()
            # 读取视频帧
            if ret == False:
                # 判断是否读取成功
                break
            imgPath = "output/%s.jpg" % str(f_sum).zfill(4)
            f_sum += 1
            # 将提取的视频帧存储进imgPath
            cv2.imwrite(imgPath, frame)

        cap = cv2.VideoCapture(args.input_source)

        # 创建文件用来保存停留帧
        if os.path.isdir("tupian"):
            shutil.rmtree("tupian")
            os.makedirs("tupian")
        else:
            os.makedirs("tupian")
        imgPath = ""# 截图的图片命名
        f_sum = cap.get(7)# 视频帧总数
        rate = cap.get(5)# 视频帧速率
        begin=int(args.capture_time[0]*rate)
        end=int(args.capture_time[1]*rate)
        if begin>f_sum or end>f_sum:
            print("error, capture-time out of range.")
        elif (end-begin+1)<args.times:
            print("error, not enough frames in capture-time.")
        else:
            in_sum=end-begin+1
            interval=in_sum//args.times
            for i in range(in_sum-1):
                if i % interval == 0:
                    shutil.copyfile("output/%s.jpg" % str(begin+i).zfill(4),"tupian/%s.jpg" % str(begin+i).zfill(4))

thread2=allcapture()
thread2.start()
thread2.join()

do_path=os.listdir("tupian")
all_path=os.listdir("output")
all_a_path=os.listdir("output_a")

# 处理台标
n = 0 #定义鼠标按下的次数
ix = 0 # x,y 坐标的临时存储
iy = 0
a=0
m=args.area_num+1
x_b=[0,0]
x_e=[0,0]
y_b=[0,0]
y_e=[0,0]
# 创建图像与窗口并将窗口与回调函数绑定
img=cv2.imread("output/"+all_path[0])
y_f = img.shape[0]
x_f = img.shape[1]
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_rectangle)
#显示并延时
while(1):
    cv2.imshow('image',img)
    if (cv2.waitKey(20) & 0xFF==13) or a==m:
        break
#销毁所有窗口
cv2.destroyAllWindows()

# 开始合成视频帧
if os.path.isdir("final"):
    shutil.rmtree("final")
    os.makedirs("final")
else:
    os.makedirs("final")
s=0
mid_img=np.array([[[0]*4]*x_f]*y_f,dtype=np.uint8)

for i in range(len(all_path)-1):
    img1_path = "output/"+all_path[i]
    img2_path = "output/"+all_path[i+1] 
    img1_a_path = "output_a/"+all_a_path[i]
    img2_a_path = "output_a/"+all_a_path[i+1]

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    img1_a = cv2.imread(img1_a_path,-1)
    img2_a = cv2.imread(img2_a_path,-1)


    detector = cv2.ORB_create()
    matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")

    # detect keypoints
    kp1 = detector.detect(img1)
    kp2 = detector.detect(img2)
    kp1 = find(kp1,img1_a,x_b,x_e,y_b,y_e)
    kp2 = find(kp2,img1_a,x_b,x_e,y_b,y_e)
    print('#keypoints in image1: %d, image2: %d' % (len(kp1), len(kp2)))

    # descriptors
    k1, d1 = detector.compute(img1, kp1)
    k2, d2 = detector.compute(img2, kp2)

    print('#keypoints in image1: %d, image2: %d' % (len(d1), len(d2)))

    # match the keypoints
    matches = matcher.match(d1, d2)

    # visualize the matches
    print('#matches:', len(matches))
    dist = [m.distance for m in matches]

    print('distance: min: %.3f' % min(dist))
    print('distance: mean: %.3f' % (sum(dist) / len(dist)))
    print('distance: max: %.3f' % max(dist))

    # threshold: half the mean
    thres_dist = (sum(dist) / len(dist)) * 0.5

    # keep only the reasonable matches
    sel_matches = [m for m in matches if m.distance < thres_dist]
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if len(sel_matches)==0:
        mid_img=np.array([[[0]*4]*x_f]*y_f,dtype=np.uint8)
        cv2.imwrite('final/'+all_path[i+1],img2)
        continue

    pts_src = []
    pts_dst = []
    for m in sel_matches:
        # print m.queryIdx, m.trainIdx, m.distance
        pts_src.append([int(k1[m.queryIdx].pt[0]), int(k1[m.queryIdx].pt[1])])
        pts_dst.append([int(k2[m.trainIdx].pt[0]), int(k2[m.trainIdx].pt[1])])

    pts_src = np.array(pts_src)
    width_dst = 1920
    height_dst = 1080
    pts_dst = np.array(pts_dst)
    x_a=0
    y_a=0
    for j in range(len(pts_dst)):
        x_a+=pts_dst[j][0]-pts_src[j][0]
        y_a+=pts_dst[j][1]-pts_src[j][1]
    x_a=x_a/len(pts_dst)
    y_a=y_a/len(pts_dst)
    mat_translation=np.float32([[1,0,x_a],[0,1,y_a]])

    # 计算单应性矩阵 这个是重点
    # h1, status = cv2.findHomography(pts_src, pts_dst)

    if all_path[i] == do_path[s]:
        mid = img1_a
        if s<len(do_path)-1:
            s+=1
        mid_img=cv2.warpAffine(mid_img,mat_translation,(width_dst, height_dst))[0:img1.shape[0], 0:img1.shape[1], :]
        # mid_img = cv2.warpPerspective(mid_img, h1, (width_dst, height_dst))[0:img1.shape[0], 0:img1.shape[1], :]
        mid_img = over(mid_img, mid)
        img2=over(img2,mid_img)
        cv2.imwrite('final/'+all_path[i+1],img2)
    else:
        mid_img=cv2.warpAffine(mid_img,mat_translation,(width_dst, height_dst))[0:img1.shape[0], 0:img1.shape[1], :]
        # mid_img = cv2.warpPerspective(mid_img, h1, (width_dst, height_dst))[0:img1.shape[0], 0:img1.shape[1], :]
        img2=over(img2,mid_img)
        cv2.imwrite('final/'+all_path[i+1],img2)      
else:
    # 合成视频
    create.out(args.input_source)

print("finish!")
