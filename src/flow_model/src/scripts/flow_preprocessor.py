#!/usr/bin/env python
import sys
sys.path.append('core')
import os
import time, glob
import numpy as np
import torch
import cv2
from PIL import Image
import argparse
import rospy
from sensor_msgs.msg import CompressedImage

libs_path = os.path.join(os.path.dirname(__file__), "RAFT/")
sys.path.append(libs_path)
libs_path = os.path.join(os.path.dirname(__file__), "RAFT/core")
sys.path.append(libs_path)

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'
prev_img = None

def callback(ros_data):
    global pub, model, count, ts_start, prev_img
    start_time = time.time()
    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (368, 176))

    image2 = torch.from_numpy(img).permute(2, 0, 1).float()
    image2 = image2[None].to(DEVICE)
    if prev_img == None:
        prev_img = image2
        return

    padder = InputPadder(prev_img.shape)
    image1, image2 = padder.pad(prev_img, image2)
    prev_img = image2
    flow_low, flow_up = model(image1, image2, iters=5, test_mode=True)
    flo = flow_up[0].permute(1,2,0).cpu().detach().numpy()
    flo = flow_viz.flow_to_image(flo)

    # print('inference time:', time.time() - start_time)
    # print('header stamp:', ros_data.header.stamp)
    # print('my delay:', round((rospy.Time.now() - ros_data.header.stamp).to_sec(), 2))

    msg = CompressedImage()
    msg.header.stamp = ros_data.header.stamp
    msg.format = "jpeg"
    flo = cv2.imencode('.jpg', flo)[1]
    img = cv2.imencode('.jpg', img)[1]
    data_to_send = np.concatenate((img, flo), 0)
    s = str(img.shape[0]).rjust(16, '0')
    b = bytes(s, 'ascii')
    msg.data = data_to_send.tostring() + b
    # Publish new image
    pub.publish(msg)

    count += 1
    delta = time.perf_counter() - ts_start
    # Log
    print("Sent", count, "images in",
        round(delta), "seconds with",
        round(count / delta, 2), "FPS and Latency",
        round((rospy.Time.now() - ros_data.header.stamp).to_sec(), 2), "seconds")

def main(args):
    global pub, model, count, ts_start
    print("Torch version", torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load("./RAFT/models/raft-things.pth"))
    model = model.module
    model.to(DEVICE)
    model.eval()

    # Init ROS node
    rospy.init_node('flow_preprocessor', anonymous=True)
    pub = rospy.Publisher("/processed/img_flow/compressed",
        CompressedImage, queue_size=1)
    sub = rospy.Subscriber("/camera/image_raw/compressed",
        CompressedImage, callback, queue_size = 1, buff_size=2*800*800*3*8*100)
    count = 0
    ts_start = time.perf_counter()
    # Begin main loop
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Zed Segmentation module")

if __name__ == '__main__':
    main(sys.argv)
