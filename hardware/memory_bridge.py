import numpy as np

class MemoryBridge:
    def __init__(self):
        self.buffer = None

    def get_shared_buffer(self, frame):
        """
        Menggunakan view/stride agar tidak memakan RAM tambahan.
        """
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        
        # Mengembalikan pointer memori yang sama tanpa copy
        return frame.view()

    def clear_cache(self):
        self.buffer = None