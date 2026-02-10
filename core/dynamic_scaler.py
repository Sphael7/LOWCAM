import cv2
import numpy as np

class DynamicScaler:
    def __init__(self):
        self.modes = {
            "high": (640, 480),
            "med": (480, 360),
            "low": (320, 240)
        }

    def get_scaled_resolution(self, perf_score):
        if perf_score > 0.8: return self.modes["high"]
        if perf_score > 0.5: return self.modes["med"]
        return self.modes["low"]

    def apply_drs(self, frame, perf_score):
        target_size = self.get_scaled_resolution(perf_score)
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)