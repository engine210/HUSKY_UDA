#!/usr/bin/env python
import sys
import time
import cv2
import numpy as np
import rospy
import torch
import math
from sensor_msgs.msg import CompressedImage
import dungeon_maps as dmap

DEVICE = 'cuda'
last_pose = [0, 0, 0]

WIDTH, HEIGHT = 672, 376
HFOV = math.radians(107)
CAM_PITCH = math.radians(0)
CAM_HEIGHT = 0.6 # meter
MIN_DEPTH = 0.1 # meter
MAX_DEPTH = 10.0 # meter

def denormalize(depth_map):
    """Denormalize depth map, from [0, 1] to [MIN_DEPTH, MAX_DEPTH]"""
    return depth_map * (MAX_DEPTH - MIN_DEPTH) + MIN_DEPTH

def subtract_pose(p1, p2):
    """Caulate delta pose from p1 -> p2"""
    x1, y1, o1 = p1[0], p1[1], p1[2]
    x2, y2, o2 = p2[0], p2[1], p2[2]

    r = ((x1-x2)**2.0 + (y1-y2)**2.0)**0.5 # distance
    p = np.arctan2(y2-y1, x2-x1) - o1 #

    do = o2 - o1
    do = np.arctan2(np.sin(do), np.cos(do)) # [-pi/2, pi/2]
    dx = r * np.cos(p)
    dy = r * np.sin(p)
    return np.stack([dx, dy, do], axis=-1) # (batch, 3)

def compute_ego_flow(proj, depth, trans_pose):
    # Compute egocentric motion flow
    # depth_map = np.transpose(denormalize(depth), (2, 0, 1)) # (1, h, w)
    depth_map = np.transpose(depth, (2, 0, 1)) # (1, h, w)
    depth_map = torch.tensor(depth_map, device='cuda')
    grid = proj.camera_affine_grid(depth_map, -trans_pose)
    x, y = dmap.utils.generate_image_coords(
        depth_map.shape,
        dtype = torch.float32,
        device = 'cuda'
    )
    coords = torch.stack((x, y), dim=-1)
    flow = coords - grid
    flow[..., 0] /= grid.shape[1]
    flow[..., 1] /= grid.shape[0]
    flow[..., 1] = -flow[..., 1] # flip y
    return flow[0, 0] # (h, w, 2)

proj = dmap.MapProjector(
    width = WIDTH, # pixel
    height = HEIGHT, # pixel
    hfov = HFOV,
    vfov = None,
    cam_pose = [0., 0., 0.],
    width_offset = 0.,
    height_offset = 0.,
    cam_pitch = CAM_PITCH,
    cam_height = CAM_HEIGHT, # meter for the trailer
    map_res = None,
    map_width = None,
    map_height = None,
    trunc_depth_min = None,
    trunc_depth_max = None,
    clip_border = None,
    to_global = True
)

def callback(ros_data):
    global proj, pub, last_pose, count, ts_start
    rt = ros_data.data[-48:] # last 48 byte is rt
    rt = np.frombuffer(rt) # [pitch, yaw, row, x, y, -z]
    # print("rt:", rt)
    depth_map = ros_data.data[:-48] # depth_raw
    depth_map = np.fromstring(depth_map, np.float32).reshape((376, 672))
    depth_map = np.expand_dims(depth_map, -1)
    
    depth_map = torch.from_numpy(depth_map)
    cam_pose = [rt[3], -rt[5], rt[1]] # [x, z, yaw]
    trans_pose = subtract_pose(last_pose, cam_pose)
    last_pose = cam_pose
    # print("trans_pose:", trans_pose)
    flow = compute_ego_flow(proj, depth_map, trans_pose)
    flow_np = flow.cpu().numpy()
    msg_ego_flow_raw = CompressedImage()
    msg_ego_flow_raw.header.stamp = rospy.Time.now()
    msg_ego_flow_raw.format = "jpeg"
    msg_ego_flow_raw.data = flow_np.tostring()
    # Publish new image
    pub.publish(msg_ego_flow_raw)

    count += 1
    delta = time.perf_counter() - ts_start
    # Log
    print("Received", count, "images in",
        round(delta), "seconds with",
        round(count / delta, 2), "FPS and Latency",
        round((rospy.Time.now() - ros_data.header.stamp).to_sec(), 2), "seconds")

def main(args):
    global pub, count, ts_start
    # Init ROS node
    rospy.init_node('ego_flow_preprocessor', anonymous=True)
    pub = rospy.Publisher("/ego_flow/ego_flow_raw/compressed",
        CompressedImage, queue_size=1)
    sub = rospy.Subscriber("/camera/depth_rt_raw/compressed",
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
