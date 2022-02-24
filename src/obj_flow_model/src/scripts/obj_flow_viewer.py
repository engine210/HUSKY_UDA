#!/usr/bin/env python
import sys, time
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage
import os

libs_path = os.path.join(os.path.dirname(__file__), "RAFT/")
sys.path.append(libs_path)
libs_path = os.path.join(os.path.dirname(__file__), "RAFT/core")
sys.path.append(libs_path)
from utils import flow_viz

recording = False
save_dir = ''

def callback(ros_data):
    global count, ts_start, recording, save_dir
    data = ros_data.data

    # img_len = int(data[-16:].decode('ascii'))
    # data = data[:-16]
    flow_all_raw = np.fromstring(data, np.float32).reshape((672, 376, 2))
    flow_all_color = flow_viz.flow_to_image(flow_all_raw)
    # np_arr = np_arr[img_len:-1]
    # image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imshow('flow_preprocessor_viewer', flow_all_color)
    # print(image_np.shape)
    key = cv2.waitKey(5)
    if key == ord('q'):
        rospy.signal_shutdown("User exit.")
        return
    count += 1
    delta = time.perf_counter() - ts_start
    # Log
    print("Received", count, "images in",
        round(delta), "seconds with",
        round(count / delta, 2), "FPS and Latency",
        round((rospy.Time.now() - ros_data.header.stamp).to_sec(), 2), "seconds")

def main(args):
    global count, ts_start
    # Init ROS node
    rospy.init_node('flow_preprocessor_viewer', anonymous=True)
    sub = rospy.Subscriber("/processed/flow_all_raw/compressed",
        CompressedImage, callback, queue_size = 1, buff_size=2*800*800*3*8*100)
    count = 0
    ts_start = time.perf_counter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Zed Segmentation Viewer module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
