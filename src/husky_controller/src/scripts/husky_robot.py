#!/usr/bin/env python

from __future__ import print_function

import threading

import roslib; roslib.load_manifest('teleop_twist_keyboard')
import rospy

from geometry_msgs.msg import Twist

import sys, select, termios, tty

import os, sys, time
import numpy as np
import cv2
from PIL import Image
import rospy
import random
from std_msgs.msg import String

msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

For Holonomic mode (strafing), hold down the shift key:
---------------------------
   U    I    O
   J    K    L
   M    <    >

t : up (+z)
b : down (-z)

anything else : stop

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%

CTRL-C to quit
"""

moveBindings = {
        'i':(1,0,0,0), # go straight
        'o':(1,0,0,-0.5), # turn right
        'j':(0,0,0,1),
        'l':(0,0,0,-1),
        'u':(1,0,0,0.5), # turn left
        ',':(-1,0,0,0),
        '.':(-1,0,0,1),
        'm':(-1,0,0,-1),
        'O':(1,-1,0,0),
        'I':(1,0,0,0),
        'J':(0,1,0,0),
        'L':(0,-1,0,0),
        'U':(1,1,0,0),
        '<':(-1,0,0,0),
        '>':(-1,-1,0,0),
        'M':(-1,1,0,0),
        't':(0,0,1,0),
        'b':(0,0,-1,0),
    }

speedBindings={
        'q':(1.1,1.1),
        'z':(.9,.9),
        'w':(1.1,1),
        'x':(.9,1),
        'e':(1,1.1),
        'c':(1,.9),
    }

def callback(ros_data):
    global a_key
    global pub, count, ts_start
    action_str = ros_data.data
    a_key = action_str
    print(action_str)

    if action_str == "1":
        # L
        a_key = 'u'
        a_key = '1'
    elif action_str == "0":
        # M
        a_key = 'i'
        a_key = '0'
    elif action_str == "2":
        # R
        a_key = 'o'
        a_key = '2'

    # msg = action_str
    # pub.publish(msg)
    count += 1
    delta = time.perf_counter() - ts_start
    # Log
    print("Sent", count, "actions in",
        round(delta), "seconds with",
        round(count / delta, 2), "FPS and Latency",
        "(unknown)", "seconds")

class PublishThread(threading.Thread):
    def __init__(self, rate):
        super(PublishThread, self).__init__()
        self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size = 1)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.th = 0.0
        self.speed = 0.0
        self.turn = 0.0
        self.condition = threading.Condition()
        self.done = False

        # Set timeout to None if rate is 0 (causes new_message to wait forever
        # for new data to publish)
        if rate != 0.0:
            self.timeout = 1.0 / rate
        else:
            self.timeout = None

        self.start()

    def wait_for_subscribers(self):
        i = 0
        while not rospy.is_shutdown() and self.publisher.get_num_connections() == 0:
            if i == 4:
                print("Waiting for subscriber to connect to {}".format(self.publisher.name))
            rospy.sleep(0.5)
            i += 1
            i = i % 5
        if rospy.is_shutdown():
            raise Exception("Got shutdown request before subscribers connected")

    def update(self, x, y, z, th, speed, turn):
        self.condition.acquire()
        self.x = x
        self.y = y
        self.z = z
        self.th = th
        self.speed = speed
        self.turn = turn
        # Notify publish thread that we have a new message.
        self.condition.notify()
        self.condition.release()

    def stop(self):
        self.done = True
        self.update(0, 0, 0, 0, 0, 0)
        self.join()

    def run(self):
        twist = Twist()
        while not self.done:
            self.condition.acquire()
            # Wait for a new message or timeout.
            self.condition.wait(self.timeout)

            # Copy state into twist message.
            twist.linear.x = self.x * self.speed
            twist.linear.y = self.y * self.speed
            twist.linear.z = self.z * self.speed
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = self.th * self.turn

            self.condition.release()

            # Publish.
            self.publisher.publish(twist)

        # Publish stop message when thread exits.
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        self.publisher.publish(twist)


def getKey(key_timeout):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], key_timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def vels(speed, turn):
    return "currently:\tspeed %s\tturn %s " % (speed,turn)

if __name__=="__main__":
    global a_key
    global pub, count, ts_start
    # Init ROS node
    rospy.init_node('husky_robot')
    sub = rospy.Subscriber("/processed/action_id_raw/string",
        String, callback, queue_size = 1, buff_size=1024)
    count = 0
    ts_start = time.perf_counter()

    settings = termios.tcgetattr(sys.stdin)

    speed = rospy.get_param("~speed", 0.5)
    turn = rospy.get_param("~turn", 1.0)
    repeat = rospy.get_param("~repeat_rate", 0.0)
    key_timeout = rospy.get_param("~key_timeout", 0.0)
    if key_timeout == 0.0:
        key_timeout = None

    pub_thread = PublishThread(repeat)

    x = 0
    y = 0
    z = 0
    th = 0
    status = 0

    a_key = ''
    key = ''
    current_action = ''
    cv_key = ''
    control_state = 0
    try:
        pub_thread.wait_for_subscribers()
        pub_thread.update(x, y, z, th, speed, turn)

        # print(msg)
        print(vels(speed,turn))
        while (1):
            # Construct visualization of action
            image_np = np.zeros((376, 672, 3))
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            fontColor = (255,0,255)
            lineType = 2

            cv_key = cv2.waitKey(5)
            if (cv_key == ord('q')):
                rospy.signal_shutdown("User exit.")
                break
            elif (cv_key == ord('0')):
                key = ''
                current_action = ''
                control_state = 0
            elif (cv_key == ord('1')):
                key = ''
                current_action = ''
                control_state = 1
            elif (cv_key == ord('2')):
                key = ''
                current_action = ''
                control_state = 2

            # Selected control
            if control_state == 0:
                # Joy Stick & Keyboard
                # Up
                image_np[:124,:,0] = 0
                image_np[:124,:,1] = 255
                image_np[:124,:,2] = 0
                if cv_key in range(0x110000):
                    key = chr(cv_key)
            elif control_state == 1:
                # Sim2Real
                # Mid
                image_np[124:248,:,0] = 0
                image_np[124:248,:,1] = 255
                image_np[124:248,:,2] = 0
                if cv_key == ord('u'):
                    current_action = '1'
                elif cv_key == ord('i'):
                    current_action = '0'
                elif cv_key == ord('o'):
                    current_action = '2'
            elif control_state == 2:
                # RL Policy
                # Down
                image_np[248:,:,0] = 0
                image_np[248:,:,1] = 255
                image_np[248:,:,2] = 0
                current_action = a_key

            if control_state != 0:
                # Sim2Real
                # action_id to real action
                if current_action == '1':
                    key = 'u'
                elif current_action == '0':
                    key = 'i'
                elif current_action == '2':
                    key = 'o'
                    
            if control_state == 1:
                if current_action == '1':
                    image_np[124:248,336:448,:] = 255
                elif current_action == '0':
                    image_np[124:248,448:560,:] = 255
                elif current_action == '2':
                    image_np[124:248,560:,:] = 255

            # Received action
            if a_key == '1':
                # L
                # 112
                image_np[248:,336:448,:] = 255
            elif a_key == '0':
                # M
                image_np[248:,448:560,:] = 255
            elif a_key == '2':
                # R
                image_np[248:,560:,:] = 255

            # Draw text
            cv2.putText(image_np, '[0] Joy Stick & Keyboard Control', (100, 124 - 60), font, fontScale, fontColor, lineType)
            cv2.putText(image_np, '[1] Keyboard Sim2Real Test (uio)', (100, 248 - 60), font, fontScale, fontColor, lineType)
            cv2.putText(image_np, '[2] RL Policy', (100, 376 - 60), font, fontScale, fontColor, lineType)
            cv2.imshow('husky_robot', image_np)

            # Control robot
            # key = getKey(key_timeout)
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                y = moveBindings[key][1]
                z = moveBindings[key][2]
                th = moveBindings[key][3]
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]
                turn = turn * speedBindings[key][1]

                print(vels(speed,turn))
                if (status == 14):
                    print(msg)
                status = (status + 1) % 15
            else:
                # Skip updating cmd_vel if key timeout and robot already
                # stopped.
                if key == '' and x == 0 and y == 0 and z == 0 and th == 0:
                    continue
                x = 0
                y = 0
                z = 0
                th = 0
                if (key == '\x03'):
                    break

            pub_thread.update(x, y, z, th, speed, turn)

    except Exception as e:
        print(e)

    finally:
        pub_thread.stop()

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
