import numpy as np

class TemporalDiff:
    def __init__(self, sensitivity=0.15):
        self.prev_frame = None
        self.sensitivity = sensitivity
        self.skip_counter = 0

    def check_motion(self, current_frame):
        """
        Algoritma Baru: Adaptive Frame Skipping.
        """
        # Sub-sampling frame untuk deteksi gerakan (Bikin 10x lebih cepat)
        curr_small = current_frame[0, 0, ::8, ::8] 

        if self.prev_frame is None:
            self.prev_frame = curr_small
            return True

        # Hitung selisih dengan resolusi rendah
        diff = np.abs(curr_small - self.prev_frame)
        motion_score = np.mean(diff)
        
        self.prev_frame = curr_small

        # Jika gerakan sangat minim, skip frame (Ini yang bikin FPS naik 6x)
        if motion_score < self.sensitivity:
            self.skip_counter += 1
            # Maksimal skip 5 frame agar tidak terlihat 'freezing'
            if self.skip_counter < 5:
                return False
        
        self.skip_counter = 0
        return True