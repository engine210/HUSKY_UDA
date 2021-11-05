import os
import time
import random
import numpy as np
from PIL import Image

from models.deeplabv2_inference import Deeplab
from models.sync_batchnorm import SynchronizedBatchNorm2d

import torch
import torch.nn.functional as F

class ProDA():
    def __init__(self):
        self.seed = 24
        self.mean = np.array([96.2056, 101.4815, 100.8839])
        # mdoel config
        self.resume_path = '/home/elsalab/Desktop/uda22/cv/ProDA/checkpoint/from_gta5_to_nthu_on_deeplabv2_current_model.pkl'
        self.bn = SynchronizedBatchNorm2d
        self.num_classes = 9
        self.bn_clr = True

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.model = Deeplab(self.bn, num_classes=self.num_classes, bn_clr=self.bn_clr)
        checkpoint = torch.load(self.resume_path)['ResNet101']["model_state"]
        self.model.load_state_dict(checkpoint)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        print('device:', self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        colors = [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [0, 0, 142],
            [119, 11, 32],
        ]

        self.label_colors = dict(zip(range(9), colors))

    def inference(self, img):
        # process image
        img = img.astype(np.float64)
        img -= self.mean
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, :]
        img = torch.from_numpy(img).float()
        img = img.to(self.device)
        with torch.no_grad():
            # inference with model
            outs = self.model(img)
        outs = F.interpolate(outs, size=img.size()[2:], mode='bilinear', align_corners=True)
        pred = outs.data.max(1)[1].cpu().numpy()
        # convert to colorized mask
        colorized_mask = self.decode_segmap(pred[0])
        return colorized_mask

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

if __name__ == "__main__":
    model = ProDA()
    img_path = '/home/elsalab/Desktop/uda22/cv/ProDA/test_20211017_145830102729.png'
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # mask = model.inference(img)
    # save_img = Image.fromarray(np.uint8(mask), mode='RGB')
    # save_path = '/home/elsalab/Desktop/uda22/cv/ProDA/logs/'
    # save_img.save(os.path.join(save_path, 'seg.png'))
    for i in range(50):
        time_start = time.time()
        mask = model.inference(img)
        time_end = time.time()
        print(time_end-time_start)
