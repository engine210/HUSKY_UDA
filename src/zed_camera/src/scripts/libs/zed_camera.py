import time
import cv2
import pyzed.sl as sl

class ZedCamera():
    def __init__(self, init=None, log=True):
        if init is None:
            init = ZedCamera.create_init_params()
        self.init = init
        self.cam = None
        self.runtime = sl.RuntimeParameters()
        self.mat = sl.Mat()
        self.res = sl.get_resolution(sl.RESOLUTION.VGA)
        self.res.width = sl.get_resolution(sl.RESOLUTION.VGA).width
        self.res.height = sl.get_resolution(sl.RESOLUTION.VGA).height
        self.depth_raw = sl.Mat(self.res.width, self.res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        self.py_translation = sl.Translation()
        self.camera_pose = sl.Pose()
        self.log = log

    def __del__(self):
        if self.cam is not None:
            self.cam.close()
            if self.log: print("ZED1 : Camera handle closed")

    def reset_cam(self):
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, -1)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, -1)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.HUE, -1)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, -1)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, -1)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1) #4600)

    def get_image(self, color_channel_order='bgra', termination_predicate=None):
        if self.cam is None:
            self.cam = ZedCamera.create_camera_handle_until_success(self.init, termination_predicate)
            self.reset_cam()
            self.print_camera_information(self.cam)
        if self.cam is None:
            return None
        consecutive_failures = 0
        while termination_predicate is None or not termination_predicate():
            err = self.cam.grab(self.runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                self.cam.retrieve_image(self.mat, sl.VIEW.LEFT)
                img = self.mat.get_data() # color channel order is bgra
                return {
                    'bgra': img,
                    'rgba': cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA),
                    'bgr': cv2.cvtColor(img, cv2.COLOR_BGRA2BGR),
                    'rgb': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                }[color_channel_order]
            else:
                consecutive_failures += 1
                if self.log: print("ZED1 : Failed to retrieve image")
                if consecutive_failures >= 3:
                    if self.log: print("Re-opening ZED Camera...")
                    self.cam = ZedCamera.create_camera_handle_until_success(self.init, termination_predicate)
                    if self.cam is None:
                        break
        return None
    
    def get_image_depth_position(self, color_channel_order='bgra', termination_predicate=None):
        if self.cam is None:
            self.cam = ZedCamera.create_camera_handle_until_success(self.init, termination_predicate)
            self.reset_cam()
            self.print_camera_information(self.cam)
        if self.cam is None:
            return None
        consecutive_failures = 0
        while termination_predicate is None or not termination_predicate():
            err = self.cam.grab(self.runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                self.cam.retrieve_image(self.mat, sl.VIEW.LEFT)
                self.cam.retrieve_measure(self.depth_raw, sl.MEASURE.DEPTH)
                img = self.mat.get_data() # color channel order is bgra
                depth = self.depth_raw.get_data()

                self.cam.get_position(self.camera_pose)
                rotation = self.camera_pose.get_rotation_vector()
                translation = self.camera_pose.get_translation(self.py_translation)
                translation_val = [round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)]

                return {
                    'bgra': img,
                    'rgba': cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA),
                    'bgr': cv2.cvtColor(img, cv2.COLOR_BGRA2BGR),
                    'rgb': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                }[color_channel_order], depth, rotation, translation_val
            else:
                consecutive_failures += 1
                if self.log: print("ZED1 : Failed to retrieve image")
                if consecutive_failures >= 3:
                    if self.log: print("Re-opening ZED Camera...")
                    self.cam = ZedCamera.create_camera_handle_until_success(self.init, termination_predicate)
                    if self.cam is None:
                        break
        return None

    @staticmethod
    def create_termination_predicate_timeout(timeout=1):
        # TODO: test this
        def termination_predicate():
            if termination_predicate.t0 is None:
                t0 = time.perf_counter()
                termination_predicate.t0 = t0
            t1 = time.perf_counter()
            timespan = t1 - t0
            if timespan < termination_predicate.timeout:
                return False
            return True
        termination_predicate.t0 = None
        termination_predicate.timeout = timeout
        return termination_predicate

    @staticmethod
    def create_init_params():
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.VGA
        return init

    @staticmethod
    def create_camera_handle(init, log=True):
        cam = sl.Camera()
        if not cam.is_opened():
            if log: print("Opening ZED Camera...")
        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            if log: print(repr(status))
            return None
        tracking_params = sl.PositionalTrackingParameters()
        cam.enable_positional_tracking(tracking_params)
        return cam

    @staticmethod
    def create_camera_handle_until_success(init, termination_predicate=None, log=True):
        cam = ZedCamera.create_camera_handle(init)
        while cam is None:
            if log: print("ZED1 : device does not (yet) available, is the usb connected?.")
            time.sleep(1)
            cam = ZedCamera.create_camera_handle(init, False)
            if termination_predicate is not None:
                terminate = termination_predicate()
                if terminate:
                    return None
        ZedCamera.print_camera_information(cam)
        return cam

    @staticmethod
    def print_camera_information(cam):
        ZedCamera.print_camera_information(cam)

    @staticmethod
    def print_camera_information(cam):
        print("Resolution: {0}, {1}.".format(round(cam.get_camera_information().camera_resolution.width, 2),
                                         cam.get_camera_information().camera_resolution.height))
        print("Camera FPS: {0}.".format(cam.get_camera_information().camera_fps))
        print("Firmware: {0}.".format(cam.get_camera_information().camera_firmware_version))
        print("Serial number: {0}.\n".format(cam.get_camera_information().serial_number))
        print("BRIGHTNESS", cam.get_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS))
        print("CONTRAST", cam.get_camera_settings(sl.VIDEO_SETTINGS.CONTRAST))
        print("HUE", cam.get_camera_settings(sl.VIDEO_SETTINGS.HUE))
        print("SATURATION", cam.get_camera_settings(sl.VIDEO_SETTINGS.SATURATION))
        print("SHARPNESS", cam.get_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS))
        print("GAIN", cam.get_camera_settings(sl.VIDEO_SETTINGS.GAIN))
        print("EXPOSURE", cam.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE))
        print("WHITEBALANCE_TEMPERATURE", cam.get_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE))
        print("")

if __name__ == "__main__":
    def print_help():
        print("Help for camera setting controls")
        print("  Save current image:                 a")
        print("  Quit:                               q\n")
    camera = ZedCamera()
    print("Running...")
    print_help()
    key = ''
    while key != ord('q'):
        def termination_predicate():
            # TODO: Test unplug & predicate
            key = cv2.waitKey(1)
            if key == ord('q'):
                return True
            return False
        img = camera.get_image('bgr', termination_predicate)
        if img is None:
            break
        cv2.imshow("ZED", img)
        key = cv2.waitKey(1)
    cv2.destroyAllWindows()
    del camera
    print("\nFINISH")