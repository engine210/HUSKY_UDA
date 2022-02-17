import numpy as np
import onnxruntime as ort
import cv2
import time

class RLModel():
    def __init__(self, img_shape=(3, 84, 180)):
    #def __init__(self, img_shape=(3, 90, 160)):
        self.img_shape = img_shape
        #self.ort_session = ort.InferenceSession("/home/elsalab/Desktop/uda22/engine/engine_husky_catkin_ws/src/rl_model/src/scripts/ID_NTHU_LIBRARY/DriverAgent-599958.onnx")
        #self.ort_session = ort.InferenceSession("/home/elsalab/Desktop/uda22/engine/engine_husky_catkin_ws/src/rl_model/src/scripts/ID_NTHU_LIBRARY_1108/DriverAgent-149966.onnx")
        self.ort_session = ort.InferenceSession("/home/elsalab/Desktop/uda22/engine/engine_husky_catkin_ws/src/rl_model/src/scripts/ID_NTHU_v1.2/DriverAgent-388311.onnx")
        #self.ort_session = ort.InferenceSession("/home/elsalab/Desktop/uda22/engine/engine_husky_catkin_ws/src/rl_model/src/scripts/ID_NTHU_v2.1/DriverAgent-1500705.onnx")
        self.last_img = np.zeros(self.img_shape).astype(np.float32)
        self.last_seg = np.zeros(self.img_shape).astype(np.float32)

    def predict(self, img, seg):
        # img shape: (h, w, c)
        img = cv2.resize(img, (self.img_shape[2], self.img_shape[1]))
        img = np.transpose(img, (2, 0, 1))
        seg = cv2.resize(seg, (self.img_shape[2], self.img_shape[1]))
        seg = np.transpose(seg, (2, 0, 1))

        print('img shape', img.shape)
        print('seg shape', seg.shape)
        img_stack = np.vstack((self.last_img, img))
        seg_stack = np.vstack((self.last_seg, seg))

        outputs = self.ort_session.run(None, {
            "obs_0": [img_stack],
            #"obs_0": [seg_stack],
            "obs_1": [seg_stack],
            "action_masks": [[True, True, True]]})
            #"action_masks": [[False, False, False]]})
        action = outputs[2][0][0]
        self.last_img = img
        self.last_seg = seg
        print("action send:", action)
        return int(action)

if __name__ == "__main__":
    model = RLModel()
    raw_img2 = cv2.imread("/home/elsalab/Desktop/uda22/rl/sample_inputs/episode_001_frame_001_image.png")
    seg2 = cv2.imread("/home/elsalab/Desktop/uda22/rl/sample_inputs/episode_001_frame_001_seg.png")
    action = model.predict(raw_img2, seg2)
    print(action, type(action))
