import numpy as np
import cv2

class RealtimePerception:
    def __init__(self):
        # Threshold warna gelap untuk rambut (0-255)
        self.hair_threshold = 60  
        # Buffer untuk menyimpan koordinat mata (untuk deteksi gerakan)
        self.eye_motion_history = []

    def verify_humanity(self, roi_frame, skin_mask, machine_trust):
        """
        Logika Hierarki:
        1. Cek Dominasi Warna Kulit (Sudah dilakukan di Engine)
        2. Cek Warna Rambut (Area atas ROI)
        3. Cek Dinamika Okular (Mata/Kedipan)
        """
        if roi_frame is None or roi_frame.size == 0:
            return False

        h, w, _ = roi_frame.shape
        
        # --- LAYER 1: RAMBUT (Hair Validation) ---
        # Mengambil 25% area paling atas dari kotak wajah
        hair_zone = roi_frame[0:int(h*0.25), :, :]
        gray_hair = cv2.cvtColor(hair_zone, cv2.COLOR_BGR2GRAY)
        
        # Hitung seberapa banyak pixel gelap (asumsi rambut hitam/gelap)
        dark_pixels = np.sum(gray_hair < self.hair_threshold)
        hair_ratio = dark_pixels / gray_hair.size
        has_hair = hair_ratio > 0.2  # Minimal 20% area atas adalah warna gelap

        # --- LAYER 2: DINAMIKA MATA (Ocular Motion) ---
        # Fokus ke area tengah wajah (tempat mata berada)
        eye_zone = roi_frame[int(h*0.3):int(h*0.6), int(w*0.2):int(w*0.8)]
        # Menghitung varians intensitas cahaya (kedipan/pergerakan bola mata)
        eye_variance = np.var(eye_zone)
        has_eye_activity = eye_variance > 20.0 # Nilai ambang batas aktivitas

        # --- FINAL CALCULATION: TRUST SCORE ---
        # machine_trust diambil dari machine.py (kemiripan dataset)
        total_trust = (machine_trust * 0.4) 
        
        if has_hair:
            total_trust += 0.3
        if has_eye_activity:
            total_trust += 0.3

        # Jika total trust lebih dari 65%, anggap sebagai Manusia Terverifikasi
        return total_trust > 0.65