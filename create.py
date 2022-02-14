import cv2
import os
def out(name):
    images = os.listdir('final')
    img=cv2.imread('final/'+images[0])
    y_f = img.shape[0]
    x_f = img.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    t=cv2.VideoWriter("final"+name,fourcc,10,(x_f,y_f),True)
    for i in images:
        image=cv2.imread("final/"+i)
        t.write(image)