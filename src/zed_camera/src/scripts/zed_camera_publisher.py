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
    pub_img = rospy.Publisher("/camera/image_raw/compressed", CompressedImage, queue_size=1)
    pub_depth = rospy.Publisher("/camera/depth_raw/compressed", CompressedImage, queue_size=1)

    # Init camera
    init = sl.InitParameters()
    # init.depth_mode = sl.DEPTH_MODE.NONE
    init.depth_mode=sl.DEPTH_MODE.ULTRA
    init.coordinate_units=sl.UNIT.METER
    init.coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.camera_fps = FPS
    init.camera_resolution = sl.RESOLUTION.VGA
    # init.camera_resolution = sl.RESOLUTION.HD720
    camera = ZedCamera(init)
    # camera.runtime.enable_depth = False
    camera.runtime.sensing_mode = sl.SENSING_MODE.FILL

    rate = rospy.Rate(FPS)
    print("Begin Publishing")
    count = 0
    ts_start = time.perf_counter()
    while not rospy.is_shutdown():
        # image_np = camera.get_image()
        image_np, depth_np = camera.get_image_and_depth()
        # cv2.imwrite('/media/980099FC0099E214/zed_video/tmp/' + str(count).zfill(6) + '.jpg', image_np)
        # Create CompressedImage
        msg_img = CompressedImage()
        msg_depth = CompressedImage()
        msg_img.header.stamp = rospy.Time.now()
        msg_depth.header.stamp = rospy.Time.now()
        msg_img.format = "jpeg"
        msg_depth.format = "jpeg"
        msg_img.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        msg_depth.data = depth_np.tostring()
        # Publish new image
        pub_img.publish(msg_img)
        pub_depth.publish(msg_depth)
        count += 1
        delta = time.perf_counter() - ts_start
        # Log
        print("Sent", count, "images in",
            round(delta), "seconds with",
            round(count / delta, 2), "FPS")
        rate.sleep()

if __name__ == '__main__':
    main(sys.argv)
