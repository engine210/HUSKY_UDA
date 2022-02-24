import torch
import math
import numpy as np
import cv2
import dungeon_maps as dmap
from dungeon_maps import camera_affine_grid
from dungeon_maps.demos.ego_flow import vis

'''
zed RT
left hand y up
thumb(x): right
index finger(y): up
middle finger(z): forward
yaw (left turn): x turn to z
row: y turn to -x
pitch: y turn to -z

zed camera return:
    translation: [x, y, -z]
    rotation:    [pitch, yaw, row]

zed camera height: 60cm
'''

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

depth_map = np.expand_dims(np.load("depth_raw_0.npy"), -1)
# depth_map = depth_map / np.max(depth_map)
depth_map = torch.from_numpy(depth_map)
print(depth_map)
print(depth_map.shape)

'''
t rotation = [0.01786496, 0.00034328, -0.02934285]
t translation = [0.0, 0.0, 0.0]
t+1 rotation = [0.0190051, -0.01963669, -0.0288396]
t+1 translation = [0.01, 0.0, 0.0]
'''
last_pose = [0, 0, 0] # [x, z, yaw]
cam_pose = [-0.01, 0, 0]
trans_pose = subtract_pose(last_pose, cam_pose)
print("trans_pose:", trans_pose)

flow = compute_ego_flow(proj, depth_map, trans_pose)
print(flow)
print(type(flow))
print(flow.shape)

# flow_xy = flow.cpu().numpy()
# flow_xy = flow_xy / np.max(flow_xy)
# cv2.imshow('flow_x', flow_xy[:, :, 1])
# cv2.waitKey()
print("flow shape", flow.shape)
flow_bgr = vis.draw_flow(flow)
print(flow_bgr)
print(flow_bgr.shape)
cv2.imshow('ego_flow_viewer', flow_bgr)
cv2.waitKey()