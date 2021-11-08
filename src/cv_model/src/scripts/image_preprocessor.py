#!/usr/bin/env python
import os, sys, time
import numpy as np
import cv2
from PIL import Image
import rospy
from sensor_msgs.msg import CompressedImage

import torch
import torchvision
import torchvision.transforms as tr

libs_path = os.path.join(os.path.dirname(__file__), "proda")
sys.path.append(libs_path)
libs_path = os.path.join(os.path.dirname(__file__), "proda/models")
sys.path.append(libs_path)

from proda.inference import ProDA

def callback(ros_data):
    global pub, model, count, ts_start
    start_time = time.time()
    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    color_segmentation = model.inference(img)[:, :, ::-1]
    # print('inference time:', time.time() - start_time)
    # print('header stamp:', ros_data.header.stamp)
    # print('my delay:', round((rospy.Time.now() - ros_data.header.stamp).to_sec(), 2))

    msg = CompressedImage()
    msg.header.stamp = ros_data.header.stamp
    msg.format = "jpeg"
    seg = cv2.imencode('.jpg', color_segmentation)[1]
    img = cv2.imencode('.jpg', img)[1]
    data_to_send = np.concatenate((img, seg), 0)
    # data_to_send = np.append(data_to_send, int(img.shape[0]))
    # msg.data = np.array(cv2.imencode('.jpg', color_segmentation)[1]).tostring()
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
    print("TorchVision version:", torchvision.__version__)

    model = ProDA()

    # Init ROS node
    rospy.init_node('image_preprocessor', anonymous=True)
    pub = rospy.Publisher("/processed/img_seg/compressed",
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
