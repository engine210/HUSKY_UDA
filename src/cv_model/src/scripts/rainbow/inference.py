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
            # [70, 130, 180],
            [0, 0, 0],
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
    img_path = '/media/980099FC0099E214/cv/ProDA/test_20211017_145830102729.png'
    img_path = 'img_20211107_144510649657_left.png'
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    
    for i in range(10):
        start = timeit.default_timer()
        mask = model.inference(img)
        end = timeit.default_timer()
        print('Total time: ' + str(end-start) + 'seconds')
    # end = timeit.default_timer()
    # print('Total time: ' + str(end-start) + 'seconds')
    # print(img[0][0])
    cv2.imwrite('./test2.png', mask)
