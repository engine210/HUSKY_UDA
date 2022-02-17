import os
import time
import random
import numpy as np
from PIL import Image
import glob
import cv2

if __name__ == "__main__":
    DIR = '/media/980099FC0099E214/zed_video/HD720_SN27035985_14-53-14_imgs'
    DIR = '/media/980099FC0099E214/zed_video/HD720_SN27035985_15-25-50_imgs'
    DIR = '/media/980099FC0099E214/zed_video/HD720_SN27035985_15-30-31_imgs'

    start = time.time()
    for img_path in glob.glob(os.path.join(DIR, '*_pred_resnet.png')):
        print(img_path)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)

        lo=np.array([0,0,0])
        hi=np.array([0,0,0])
        mask=cv2.inRange(img, lo, hi)
        img[mask>0]=(70,130,180)
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print('time spend', time.time() - start)