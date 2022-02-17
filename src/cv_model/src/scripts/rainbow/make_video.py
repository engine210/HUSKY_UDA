import cv2
import numpy as np
import sys
import os
from PIL import Image
import timeit
import glob

if __name__ == "__main__":
    DIR = '/media/980099FC0099E214/zed_video/HD720_SN27035985_14-53-14_imgs'
    DIR = '/media/980099FC0099E214/zed_video/HD720_SN27035985_15-25-50_imgs'
    DIR = '/media/980099FC0099E214/zed_video/HD720_SN27035985_15-30-31_imgs'
    for pred_path in glob.glob(os.path.join(DIR, '*_pred_resnet.png')):
        pred = cv2.imread(pred_path)
        img_path = pred_path.replace('_pred_resnet.png', '.png')
        img = cv2.imread(img_path)
        dst = cv2.addWeighted(img, 1, pred, 0.5, 0)
        save_name = pred_path.replace('_pred_resnet.png', '_blend_resnet.png')
        print(save_name)
        cv2.imwrite(save_name, dst)