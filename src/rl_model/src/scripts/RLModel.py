import numpy as np
import onnxruntime as ort
import cv2
import time

class RLModel():
    def __init__(self, img_size=(3, 84, 180)):
        self.ort_session = ort.InferenceSession("ID1_seg2_raw2_complex_sac_randomSpeed_for_car/DriverAgent.onnx")
        self.img_size = img_size
        self.last_img = np.zeros(self.img_size).astype(np.float32)
        self.last_seg = np.zeros(self.img_size).astype(np.float32)

    def predict(self, img, seg):
        img_stack = np.vstack((self.last_img, img))
        seg_stack = np.vstack((self.last_seg, seg))

        outputs = self.ort_session.run(None, {
            "obs_0": [img_stack],
            "obs_1": [seg_stack],
            "action_masks": [[False, False, False]]})
        action = outputs[2][0][0]
        self.last_img = img
        self.last_seg = seg
        return int(action)

if __name__ == "__main__":
    model = RLModel()
    raw_img2 = np.moveaxis(cv2.imread("/home/elsalab/Desktop/uda22/rl/sample_inputs/episode_001_frame_001_image.png"), -1, 0)
    seg2 = np.moveaxis(cv2.imread("/home/elsalab/Desktop/uda22/rl/sample_inputs/episode_001_frame_001_seg.png"), -1, 0)
    action = model.predict(raw_img2, seg2)
    print(action, type(action))