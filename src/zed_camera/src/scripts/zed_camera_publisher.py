#!/usr/bin/env python
import os
import sys
import time
import cv2
import numpy as np
import pyzed.sl as sl
import rospy
from sensor_msgs.msg import CompressedImage

libs_path = os.path.join(os.path.dirname(__file__), "libs")
sys.path.append(libs_path)
from zed_camera import ZedCamera

def main(args):
    FPS = 60

    # Init ROS node
    rospy.init_node('zed_camera_publisher', anonymous=True)
    pub = rospy.Publisher("/camera/image_raw/compressed",
        CompressedImage, queue_size=1)

    # Init camera
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.camera_fps = FPS
    init.camera_resolution = sl.RESOLUTION.VGA
    # init.camera_resolution = sl.RESOLUTION.HD720
    camera = ZedCamera(init)
    camera.runtime.enable_depth = False

    rate = rospy.Rate(FPS)
    print("Begin Publishing")
    count = 0
    ts_start = time.perf_counter()
    while not rospy.is_shutdown():
        image_np = camera.get_image()
        # cv2.imwrite('/media/980099FC0099E214/zed_video/tmp/' + str(count).zfill(6) + '.jpg', image_np)
        # Create CompressedImage
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # Publish new image
        pub.publish(msg)
        count += 1
        delta = time.perf_counter() - ts_start
        # Log
        print("Sent", count, "images in",
            round(delta), "seconds with",
            round(count / delta, 2), "FPS")
        rate.sleep()

if __name__ == '__main__':
    main(sys.argv)
