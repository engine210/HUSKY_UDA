from cmath import isnan
import numpy as np
import glob
import math

s = 0
c = 0
for f in glob.glob("flow_sample_data/ego_flow_raw*.npy"):
    ego_flow_raw = np.load(f)
    flow_all_raw = np.load(f.replace("ego_flow_raw", "flow_all_raw"))
    flow_all_raw[:,:,0] *= -1.867
    flow_all_raw[:,:,1] *= 2.238
    # sum_ego_x = np.sum(ego_flow_raw[:,:,0])
    # sum_all_x = np.sum(flow_all_raw[:,:,0])
    # print(sum_ego_x, sum_all_x, sum_ego_x / sum_all_x)
    # print(sum_ego_x / sum_all_x)
    obj_flow_raw = flow_all_raw - ego_flow_raw
    mean_err_x = np.mean(abs(obj_flow_raw[:,:,0]))
    print(mean_err_x)
    if not math.isnan(mean_err_x):
        c += 1
        s += mean_err_x
    # sum_ego_y = np.sum(ego_flow_raw[:,:,1])
    # sum_all_y = np.sum(flow_all_raw[:,:,1])
    # print(sum_ego_y, sum_all_y, sum_ego_y / sum_all_y)
    # print(sum_ego_y / sum_all_y)
    # s += sum_ego_y / sum_all_y

print("avg:", s/c)

# ego_flow_raw = np.load("ego_flow_raw.npy")
# flow_all_raw = np.load("flow_all_raw.npy")

# print(ego_flow_raw)
# print(flow_all_raw)

# sum_ego_x = np.sum(ego_flow_raw[:,:,0])
# sum_all_x = np.sum(flow_all_raw[:,:,0])
# print(sum_ego_x)
# print(sum_all_x)
# print(sum_ego_x / sum_all_x)

# sum_ego_y = np.sum(ego_flow_raw[:,:,1])
# sum_all_y = np.sum(flow_all_raw[:,:,1])
# print(sum_ego_y)
# print(sum_all_y)
# print(sum_ego_y / sum_all_y)