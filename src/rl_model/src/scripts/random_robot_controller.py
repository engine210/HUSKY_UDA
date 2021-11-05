#!/usr/bin/env python
import os, sys, time
# https://github.com/numpy/numpy/issues/18131
import numpy as np
import cv2
from PIL import Image
import rospy
import random
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from RLModel import RLModel

def callback(ros_data):
    global pub, count, ts_start, agent
    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # Publish new image
    action = agent.predict()
    action_str = str(action)
    msg = action_str
    pub.publish(msg)
    count += 1
    delta = time.perf_counter() - ts_start
    # Log
    print("Sent", count, "actions in",
        round(delta), "seconds with",
        round(count / delta, 2), "FPS and Latency",
        round((rospy.Time.now() - ros_data.header.stamp).to_sec(), 2), "seconds")

def main(args):
    global pub, count, ts_start, agent

    agent = RLModel()

    # Init ROS node
    rospy.init_node('robot_controller', anonymous=True)
    pub = rospy.Publisher("/processed/action_id_raw/string",
        String, queue_size=1)
    sub = rospy.Subscriber("/camera/image_raw/compressed",
        CompressedImage, callback, queue_size = 1)
    count = 0
    ts_start = time.perf_counter()
    # Begin main loop
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Robot Controller module")

if __name__ == '__main__':
    main(sys.argv)
