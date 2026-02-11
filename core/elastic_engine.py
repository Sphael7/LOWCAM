import numpy as np

class ElasticEngine:
    def __init__(self):
        # Rentang YCrCb yang lebih toleran untuk berbagai kondisi cahaya
        self.cr_low, self.cr_high = 130, 180
        self.cb_low, self.cb_high = 75, 135

    def run(self, frame, perf_score):
        # Konversi Manual BGR ke YCrCb
        b, g, r = frame[:,:,0].astype(np.float32), frame[:,:,1].astype(np.float32), frame[:,:,2].astype(np.float32)
        cr = 0.500 * r - 0.419 * g - 0.081 * b + 128
        cb = -0.169 * r - 0.331 * g + 0.500 * b + 128

        # Masking warna kulit
        mask = (cr > self.cr_low) & (cr < self.cr_high) & (cb > self.cb_low) & (cb < self.cb_high)
        
        # Stride adaptif agar FPS tetap 60+
        stride = 10 if perf_score > 0.4 else 20
        h, w = mask.shape
        detected_faces = []

        # Raster Scan mencari konsentrasi warna kulit
        for y in range(0, h - 140, stride * 2):
            for x in range(0, w - 120, stride * 2):
                roi = mask[y:y+140, x:x+120]
                if np.count_nonzero(roi) / roi.size > 0.5: # 50% densitas kulit
                    detected_faces.append((x, y, 120, 140))
                    break # Ambil yang pertama ketemu di area tersebut

        return self._suppress_overlap(detected_faces)

    def _suppress_overlap(self, rects):
        if not rects: return []
        # Mengambil kotak paling tengah (biasanya wajah)
        return [rects[len(rects)//2]]