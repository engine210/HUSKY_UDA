import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
# import torchvision.models as models
import torch.nn.functional as F
# from torch.utils import data, model_zoo

from model.deeplabv2 import Res_Deeplab
from model.deeplabv2_end import Deeplabv2
# from data import get_data_path, get_loader
# import torchvision.transforms as transform

from PIL import Image
import scipy.misc
# from utils.loss import CrossEntropy2d
# from utils.helpers import colorize_mask
import time
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
# np.set_printoptions(threshold=np.inf)


MODEL = 'deeplabv2' # deeeplabv2, deeplabv3p

class DACS():
    def __init__(self):
        
        model_path = '/home/elsalab/Desktop/uda22/engine/engine_husky_catkin_ws/src/cv_model/src/scripts/dacs/checkpoint/checkpoint-iter250000.pth'
        self.num_classes = 9
        # self.model = Res_Deeplab(num_classes=self.num_classes)
        self.model = Deeplabv2(backbone='mobilenet', num_classes=self.num_classes)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])
        self.model.cuda()
        self.model.eval()
        self.NTHU_HUSKY_palette =  [
            128, 64, 128,
            244, 35, 232,
            70, 70, 70,
            107, 142, 35,
            152, 251, 152,
            70, 130, 180,
            220, 20, 60,
            0, 0, 142,
            119, 11, 32,
            255, 255, 0
        ]
        colors = [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [107, 142, 35],
            #[244, 35, 232],
            [152, 251, 152],
            #[244, 35, 232],
            #[70, 130, 180],
            [0, 0, 0],
            [220, 20, 60],
            [0, 0, 142],
            [119, 11, 32],
        ]

        self.label_colors = dict(zip(range(9), colors))
        
    def colorize_mask(self, mask, palette):
        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(palette)
        return new_mask
    
    def decode_segmap(self, pred):
        r = pred.copy()
        g = pred.copy()
        b = pred.copy()
        for l in range(0, self.num_classes):
            r[pred == l] = self.label_colors[l][0]
            g[pred == l] = self.label_colors[l][1]
            b[pred == l] = self.label_colors[l][2]

        rgb = np.zeros((pred.shape[0], pred.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb

    def inference(self, img):
        img_size = img.shape[:2] # (w, h)
        img = np.asarray(img, np.float32)
        img = img[:, :, ::-1]  # change to BGR
        img -= IMG_MEAN
        image = img.transpose((2, 0, 1))# C H W
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image.copy())
        interp = nn.Upsample(size=img_size, mode='bilinear', align_corners=True)
        with torch.no_grad():
            output = self.model(Variable(image).cuda())
            output = interp(output)
            output = output.cpu().data[0].numpy()
            output = np.asarray(np.argmax(output, axis=0), dtype=np.int)
            # colorized_mask = self.colorize_mask(output, self.NTHU_HUSKY_palette)
            colorized_mask = self.decode_segmap(output)

        return colorized_mask


if __name__ == '__main__':
    dacs_model = DACS()
    img_path = '/media/980099FC0099E214/cv/ProDA/test_20211017_145830102729.png'
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    img_pred = dacs_model.inference(img)
    print(type(img_pred))
    print(img_pred.shape)
    Image.fromarray(img_pred.astype(np.uint8)).save('test_pred.png')
