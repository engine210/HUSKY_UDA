import argparse
import cv2
# from skimage import feature
import numpy as np
# np.set_printoptions(threshold=np.inf)
import sys
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from model import deeplabv2_end
from model import deeplabv3
from PIL import Image
import timeit
import glob

class Rainbow():
    def __init__(self):
        self.mean = np.array([96.2056, 101.4815, 100.8839])
        self.label_size = [720, 1280]
        # mdoel config
        self.ckpt_path = '/home/elsalab/Desktop/uda22/engine/engine_husky_catkin_ws/src/cv_model/src/scripts/rainbow/35000_model.pth'
        self.num_classes = 9
        
        self.model = deeplabv3.Deeplabv3(backbone='mobilenet', num_classes=self.num_classes)
        checkpoint = torch.load(self.ckpt_path)
        self.model.load_state_dict(checkpoint)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        print('device:', self.device)
        self.model = self.model.cuda()
        self.model.eval()

        colors = [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            # [0, 0, 0],
            [220, 20, 60],
            [0, 0, 142],
            [119, 11, 32],
            [255, 255, 0]
        ]

        class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "terrain",
            "sky",
            "person/rider",
            "car/truck/bus",
            "motorcycle/bicycle",
            "obstacle",
        ]

        self.label_colours = dict(zip(range(10), colors))

    def inference(self, img):
        img = np.asarray(img, np.float32)
        img = img[:, :, ::-1]  # change to BGR
        img -= self.mean
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img.copy()).float()
        interp = nn.Upsample(size=(720,1280), mode='bilinear', align_corners=True)

        
        with torch.no_grad():
            outputs = self.model(Variable(img).cuda())
            output = interp(outputs).squeeze()
            output = output.cpu().data.numpy()
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
            output = self.decode_segmap(output)
        
        return output

    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, 10):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb

if __name__ == "__main__":
    model = Rainbow()

    # DIR = '/media/980099FC0099E214/zed_video/HD720_SN27035985_14-53-14_imgs'
    # DIR = '/media/980099FC0099E214/zed_video/HD720_SN27035985_15-25-50_imgs'
    # DIR = '/media/980099FC0099E214/zed_video/HD720_SN27035985_15-30-31_imgs'

    # DIR = '/media/980099FC0099E214/zed_video/video20211121_162150633542'
    DIR = '/media/980099FC0099E214/zed_video/video20211121_160652885617'
    for img_path in glob.glob(os.path.join(DIR, '*.jpg')):
        if '_' in img_path.split('/')[-1]: continue
        print(img_path)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)
        mask = model.inference(img)
        save_name = img_path.replace('.jpg', '_pred_rainbow_mobilenet.jpg')
        print(save_name)
        cv2.imwrite(save_name, mask[:, :, ::-1]) #cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
