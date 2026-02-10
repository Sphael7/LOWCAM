import numpy as np

class LowCamQuantizer:
    def __init__(self):
        self.scale = 1.0
        self.zero_point = 0

    def quantize_to_int8(self, tensor):
        """
        Mengompres 32-bit float ke 8-bit integer.
        Mempercepat proses CPU hingga 2-3x lipat.
        """
        min_val, max_val = tensor.min(), tensor.max()
        self.scale = (max_val - min_val) / 255.0
        self.zero_point = int(-min_val / self.scale)
        
        q_tensor = (tensor / self.scale + self.zero_point).astype(np.int8)
        return q_tensor

    def dequantize(self, q_tensor):
        return ((q_tensor.astype(np.float32) - self.zero_point) * self.scale)