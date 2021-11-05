import os
from PIL import Image
import numpy as np
import time

import torch
import torchvision.transforms as tr

from modeling.deeplab import DeepLab
from dataloaders.utils import decode_segmap

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "deeplab-mobilenet.pth"))

model = DeepLab(num_classes=21,
                # backbone='resnet',
                backbone='mobilenet',
                output_stride=16,
                sync_bn=True,
                freeze_bn=False)

model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)

def transform(image):
    return tr.Compose([
        tr.Resize(513),
        tr.CenterCrop(513),
        tr.ToTensor(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])(image)

torch.set_grad_enabled(False)

timestamp = time.perf_counter()

N = 10
for i in range(N):
    image = Image.open(os.path.join(os.path.dirname(__file__), "2008_000012.jpg"))
    inputs = transform(image).to(device)
    output = model(inputs.unsqueeze(0)).squeeze().cpu().numpy()
    pred = np.argmax(output, axis=0)

timeelapsed = time.perf_counter() - timestamp
print("Time Elapsed: ", timeelapsed / N, "(sec/image)")

# Then visualize it:
decode_segmap(pred, dataset="pascal", plot=True)
