#!/usr/bin/env python
import sys
sys.path.append('core')
import os
import time, glob
import numpy as np
import torch
import cv2
import rospy
from sensor_msgs.msg import CompressedImage

libs_path = os.path.join(os.path.dirname(__file__), "RAFT/")
sys.path.append(libs_path)
libs_path = os.path.join(os.path.dirname(__file__), "RAFT/core")
sys.path.append(libs_path)
from utils import flow_viz

ego_flow_raw = None
ego_flow_header = None

def callback_ego_flow(ros_data):
    global count_ego_flow, ts_start, ego_flow_raw, ego_flow_header
    data = ros_data.data
    ego_flow_header = ros_data.header.stamp
    ego_flow = np.fromstring(data, np.float32).reshape((376, 672, 2))
    ego_flow_raw = cv2.resize(ego_flow, (360, 168), interpolation = cv2.INTER_LINEAR)
    count_ego_flow += 1
    delta = time.perf_counter() - ts_start
    # Log
    print("Received", count_ego_flow, "ego_flow_raw in",
        round(delta), "seconds with",
        round(count_ego_flow / delta, 2), "FPS and Latency",
        round((rospy.Time.now() - ros_data.header.stamp).to_sec(), 2), "seconds")
    
def callback_flow_all(ros_data):
    global pub, count_flow_all, ts_start, ego_flow_raw
    data = ros_data.data
    flow_all_raw = np.fromstring(data, np.float32).reshape((168, 360, 2))
    
    # print("ego_flow_raw:", ros_data.header.stamp)
    # print("flow_all_raw:", ros_data.header.stamp)

    # if count_ego_flow % 10 == 0:
    #     np.save("flow_sample_data/ego_flow_raw"+str(count_ego_flow), ego_flow_raw)
    #     np.save("flow_sample_data/flow_all_raw"+str(count_ego_flow), flow_all_raw)

    flow_all_raw[:,:,0] *= -1.867
    flow_all_raw[:,:,1] *= 2.238

    # print("ego_flow_raw:", ego_flow_raw)
    # print("flow_all_raw:", flow_all_raw)

    obj_flow_raw = flow_all_raw - ego_flow_raw
    obj_flow_color = flow_viz.flow_to_image(obj_flow_raw)
    ego_flow_color = flow_viz.flow_to_image(ego_flow_raw)
    flow_all_color = flow_viz.flow_to_image(flow_all_raw)

    cv2.imshow('obj_flow_viewer', obj_flow_color)
    cv2.imshow('ego_flow_viewer', ego_flow_color)
    cv2.imshow('flow_all_viewer', flow_all_color)
    key = cv2.waitKey(5)
    if key == ord('q'):
        rospy.signal_shutdown("User exit.")
        return

    count_flow_all += 1
    delta = time.perf_counter() - ts_start
    # Log
    print("Received", count_flow_all, "flow_all_raw in",
        round(delta), "seconds with",
        round(count_flow_all / delta, 2), "FPS and Latency",
        round((rospy.Time.now() - ros_data.header.stamp).to_sec(), 2), "seconds")

def main(args):
    global pub, sub_ego_flow, sub_flow_all, count_ego_flow, count_flow_all, ts_start

    # Init ROS node
    rospy.init_node('flow_preprocessor', anonymous=True)
    pub = rospy.Publisher("/processed/ego_flow_raw/compressed",
        CompressedImage, queue_size=1)
    sub_ego_flow = rospy.Subscriber("/ego_flow/ego_flow_raw/compressed",
        CompressedImage, callback_ego_flow, queue_size = 1, buff_size=2*800*800*3*8*100)
    sub_flow_all = rospy.Subscriber("/processed/flow_all_raw/compressed",
        CompressedImage, callback_flow_all, queue_size = 1, buff_size=2*800*800*3*8*100)
    count_ego_flow = 0
    count_flow_all = 0
    ts_start = time.perf_counter()
    # Begin main loop
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Zed Segmentation module")

if __name__ == '__main__':
    main(sys.argv)
