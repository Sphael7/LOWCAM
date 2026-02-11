import numpy as np
import cv2
import os

class MachineLearningCore:
    def __init__(self, data_path="data/"):
        self.data_path = data_path
        self.learned_signatures = []
        self.process_dataset()

    def process_dataset(self):
        # Membuat folder data jika belum ada
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            return

        print(f"[Machine] Mempelajari dataset dari {self.data_path}...")
        
        # Loop file di folder data
        for img_name in os.listdir(self.data_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.data_path, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Konversi ke YCrCb untuk mengambil signature warna kulit
                    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                    avg_cr = np.mean(ycrcb[:,:,1])
                    avg_cb = np.mean(ycrcb[:,:,2])
                    self.learned_signatures.append((avg_cr, avg_cb))
        
        print(f"[Machine] Berhasil memproses {len(self.learned_signatures)} wajah.")

    def get_closest_signature(self, current_cr, current_cb):
        if not self.learned_signatures:
            return 0.5 # Default trust level jika data kosong
        
        # Mencari jarak Euclidean terkecil (kemiripan warna)
        diffs = [np.sqrt((current_cr - lr[0])**2 + (current_cb - lr[1])**2) for lr in self.learned_signatures]
        min_diff = min(diffs)
        
        # Semakin kecil selisih (diff), semakin tinggi trust level (max 1.0)
        trust_score = 1.0 / (1.0 + (min_diff * 0.1))
        return trust_score