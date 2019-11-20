#coding=utf-8

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import KFDataset
from models import KFSGNet
from train import config,get_peak_points
import cv2
import time


def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    #img = torch.from_numpy(img)
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def test():
    # 加载模型
    net = KFSGNet()
    net.float().cuda()
    net.eval()
    if (config['checkout'] != ''):
        net.load_state_dict(torch.load(config['checkout']))

    all_result = []

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("camera is not ready !!!!!")
        exit(0)
    
    while True:
        ret,frame = camera.read()
        if ret is None:
            break
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        height=len(image)
        width=len(image[0])
        
        t0  = time.time()

        image = image[0:height,0:height]

        image_resized = cv2.resize(image,(256,256))
    
        image1 = Variable(toTensor(image_resized)).cuda()  
        pred_heatmaps = net(image1)
        #print(pred_heatmaps.shape)

        #cv2.imshow("heatmap",pred_heatmaps.cpu().data.numpy()[0][0])

        
        pred_points = get_peak_points(pred_heatmaps.cpu().data.numpy()) #(N,4,2)
        pred_points = pred_points.reshape((pred_points.shape[0],-1)) #(N,8)

        print(pred_points)

        image_resized = cv2.cvtColor(image_resized,cv2.COLOR_RGB2BGR)

        cv2.imshow("result",image_resized)
        cv2.waitKey(1)
 

if __name__ == '__main__':
    test()