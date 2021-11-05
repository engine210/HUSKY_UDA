#!/usr/bin/env python
import sys, time
import numpy as np
import cv2
import rospy
from std_msgs.msg import String

def callback(ros_data):
    global count, ts_start
    action_str = ros_data.data
    # Construct visualization of action
    image_np = np.zeros((376, 672, 3))
    if action_str == '1':
        # Left white
        image_np[:,:224,:] = 255
    elif action_str == '0':
        # Middle white
        image_np[:,224:448,:] = 255
    elif action_str == '2':
        # Right white
        image_np[:,448:,:] = 255
    else:
        print("Unrecognized action str:", action_str)
    cv2.imshow('robot_controller_viewer', image_np)
    key = cv2.waitKey(5)
    if key == ord('q'):
        rospy.signal_shutdown("User exit.")
        return
    count += 1
    delta = time.perf_counter() - ts_start
    # Log
    print("Received action:", ros_data)
    print("Received", count, "actions in",
        round(delta), "seconds with",
        round(count / delta, 2), "FPS and Latency",
        "(unknown)", "seconds")

def main(args):
    global count, ts_start
    # Init ROS node
    rospy.init_node('robot_controller_viewer', anonymous=True)
    sub = rospy.Subscriber("/processed/action_id_raw/string",
        String, callback, queue_size = 1)
    count = 0
    ts_start = time.perf_counter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Robot Controller Viewer module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
