#!/usr/bin/env python
import sys
import time
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
import datetime
import os

recording = False
save_dir = ''

def callback(ros_data):
    global count, ts_start, recording, save_dir
    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # print(image_np.shape)
    cv2.imshow('zed_raw_viewer', image_np)
    key = cv2.waitKey(5)
    if key == ord('q'):
        rospy.signal_shutdown("User exit.")
        return
    if key == ord('r') and not recording:
        count = 0
        ts_start = time.perf_counter()
        recording = True
        save_dir = 'video' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
        os.mkdir('/media/980099FC0099E214/zed_video/' + save_dir)
    if key == ord('s'):
        recording = False
    if recording:
        cv2.imwrite('/media/980099FC0099E214/zed_video/' + save_dir + '/' + str(count).zfill(6) + '.jpg', image_np)
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
    rospy.init_node('zed_raw_viewer', anonymous=True)
    sub = rospy.Subscriber("/camera/image_raw/compressed",
        CompressedImage, callback, queue_size=1, buff_size=2*800*800*3*8*100)
    count = 0
    ts_start = time.perf_counter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Zed Raw Viewer module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
