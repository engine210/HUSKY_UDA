#!/usr/bin/env python
import sys, time
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage

recording = False
save_dir = ''

def callback(ros_data):
    global count, ts_start, recording, save_dir
    data = ros_data.data
    img_len = int(data[-16:].decode('ascii'))
    data = data[:-16]
    np_arr = np.fromstring(data, np.uint8)
    np_arr = np_arr[img_len:-1]
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imshow('image_preprocessor_viewer', image_np)
    # print(image_np.shape)
    key = cv2.waitKey(5)
    if key == ord('q'):
        rospy.signal_shutdown("User exit.")
        return
    # if key == ord('r'):
    #     recording = True
    #     save_dir = 'video' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    #     os.mkdir('/media/980099FC0099E214/zed_video/' + save_dir)
    # if key == ord('s'):
    #     recording = False
    # if recording:
    #     cv2.imwrite('/media/980099FC0099E214/zed_video/' + save_dir + '/' + str(count).zfill(6) + '.jpg', image_np)
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
    rospy.init_node('image_preprocessor_viewer', anonymous=True)
    sub = rospy.Subscriber("/processed/img_seg/compressed",
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
