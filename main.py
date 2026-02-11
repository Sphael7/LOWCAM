import cv2
import numpy as np
import time
import platform
import psutil
import os
import gc

# Import komponen internal LOWCAM
from hardware.thermal_monitor import ThermalMonitor
from brain.temporal_diff import TemporalDiff
from core.elastic_engine import ElasticEngine
from brain.rl_controller import RLController
from brain.temporal_vector_integrator import TemporalVectorIntegrator

# Import komponen baru untuk Logika Hierarki Wajah
from brain.machine import MachineLearningCore
from realtime import RealtimePerception

def get_current_ram_usage():
    """Mengambil penggunaan RAM spesifik untuk proses Python ini saja (dalam MB)."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    cv2.setNumThreads(4)
    
    # Inisialisasi ID Kamera dan Objek Kamera
    cam_id = 0
    cam = cv2.VideoCapture(cam_id)
    
    # Inisialisasi Komponen Utama
    monitor = ThermalMonitor(fps_target=30)
    eye = TemporalDiff(sensitivity=0.08)
    engine = ElasticEngine()
    rl = RLController()
    
    # Inisialisasi Algoritma Temporal Vector Integrator (TVI)
    tvi = TemporalVectorIntegrator(integration_limit=4)
    
    # Inisialisasi Komponen Hierarki Pendeteksi Wajah (Baru)
    machine = MachineLearningCore(data_path="data/")
    perception = RealtimePerception()

    print("\n" + "="*55)
    print("      LOWCAM LIBRARY - ACTIVE & READY")
    print("="*55)
    print("KONTROL:")
    print("  [k] : Mulai/Berhenti Log & Lihat Detail RAM/Spek")
    print("  [s] : Switch Kamera (Ganti ke Kamera Depan/Lain)")
    print("  [q] : Keluar dari Program")
    print("="*55 + "\n")

    last_time = time.time()
    last_log_time = time.time()
    fps = 0
    
    is_logging = False
    log_data = [] 

    while True:
        ret, frame = cam.read()
        if not ret: break

        # Efek Mirroring agar deteksi wajah sinkron dengan gerakan tangan
        frame = cv2.flip(frame, 1)

        # Pre-processing untuk input tensor
        input_frame = cv2.resize(frame, (960, 720), interpolation=cv2.INTER_NEAREST)
        input_tensor = input_frame.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float16) / 255.0

        current_score = monitor.get_performance_score(fps)
        
        # Cek gerakan dasar
        has_motion = eye.check_motion(input_tensor)

        if has_motion:
            # 1. Jalankan TVI untuk menstabilkan dan menajamkan deteksi gerakan
            optimized_frame, is_ready = tvi.synchronize_execution_rate(frame, current_score)
            
            if is_ready:
                # 2. Engine mendeteksi kandidat wajah menggunakan Manual Skin Color Segmenting
                faces = engine.run(optimized_frame, current_score)
                
                # 3. Analisis Lanjutan Menggunakan Machine Base & Realtime Perception
                for (x, y, w, h) in faces:
                    # Ambil area ROI (Region of Interest) untuk dianalisis lebih dalam
                    roi = frame[y:y+h, x:x+w]
                    if roi.size == 0: continue
                    
                    # A. Ekstrak signature warna dari wajah saat ini (YCrCb)
                    ycrcb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
                    cur_cr = np.mean(ycrcb_roi[:,:,1])
                    cur_cb = np.mean(ycrcb_roi[:,:,2])
                    
                    # B. Bandingkan dengan data dari machine.py (Knowledge Base)
                    m_trust = machine.get_closest_signature(cur_cr, cur_cb)
                    
                    # C. Validasi Hierarki (Kulit -> Rambut -> Mata) di realtime.py
                    is_human = perception.verify_humanity(roi, None, m_trust)
                    
                    if is_human:
                        # Hijau jika terverifikasi secara hierarki (Manusia Nyata)
                        color = (0, 255, 0)
                        label = "VERIFIED HUMAN"
                    else:
                        # Merah jika hanya terdeteksi warna kulit tapi tidak memenuhi syarat hierarki
                        color = (0, 0, 255)
                        label = "UNVERIFIED"

                    # Visualisasi hasil akhir
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Cleanup memori tensor
            del input_tensor
        
        # Paksa pembersihan RAM setiap interval tertentu
        if int(fps) % 30 == 0:
            gc.collect()
        
        # Hitung FPS real-time
        new_time = time.time()
        time_diff = new_time - last_time
        fps = 1 / time_diff if time_diff > 0 else 0
        last_time = new_time

        # --- LOGIKA PEREKAMAN DATA (Tombol K) ---
        current_time = time.time()
        if is_logging:
            if current_time - last_log_time >= 1.0:
                current_ram = get_current_ram_usage()
                log_data.append((len(log_data) + 1, round(fps, 2), round(current_score, 2), round(current_ram, 2)))
                last_log_time = current_time

        # UI Kamera
        rec_color = (0, 0, 255) if is_logging else (0, 255, 0)
        rec_label = "● REC" if is_logging else "○ STANDBY"
        
        cv2.putText(frame, f"FPS: {int(fps)} | Score: {current_score:.2f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{rec_label} | RAM: {get_current_ram_usage():.1f}MB", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec_color, 2)
        
        cv2.imshow("LOWCAM - Performance & Memory Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('k'):
            if not is_logging:
                is_logging = True
                log_data = []
                last_log_time = time.time()
                print("[!] Recording Data (FPS & RAM)...")
            else:
                is_logging = False
                print("\n" + "="*60)
                print(f"{'DETIK':<8} | {'FPS':<10} | {'SCALE':<10} | {'RAM (MB)':<10}")
                print("-" * 60)
                
                sum_fps, sum_score, sum_ram = 0, 0, 0
                for d, f, s, r in log_data:
                    print(f"{d:<8} | {f:<10} | {s:<10} | {r:<10}")
                    sum_fps += f
                    sum_score += s
                    sum_ram += r
                
                if log_data:
                    n = len(log_data)
                    avg_fps, avg_score, avg_ram = sum_fps/n, sum_score/n, sum_ram/n
                    status = "SANGAT LANCAR" if avg_fps > 25 else ("CUKUP" if avg_fps > 15 else "LAG")
                    
                    print("-" * 60)
                    print(f"{'AVG':<8} | {avg_fps:<10.2f} | {avg_score:<10.2f} | {avg_ram:<10.2f}")
                    print("="*60)
                    print(f"KESIMPULAN: {status} | RATA-RATA RAM: {avg_ram:.2f} MB")
                    print("="*60)
                    
                    print(f"{' [ DATA HARDWARE ] ':-^60}")
                    print(f"OS          : {platform.system()} {platform.release()}")
                    print(f"CPU         : {platform.processor()}")
                    ram_sys = psutil.virtual_memory()
                    print(f"Total RAM   : {ram_sys.total / (1024**3):.2f} GB")
                    print(f"Sisa Disk   : {psutil.disk_usage('/').free / (1024**3):.2f} GB")
                    battery = psutil.sensors_battery()
                    if battery:
                        print(f"Baterai     : {battery.percent}% ({'Charging' if battery.power_plugged else 'Battery'})")
                    print("="*60 + "\n")

        elif key == ord('s'):
            cam_id = 1 if cam_id == 0 else 0
            cam.release()
            cam = cv2.VideoCapture(cam_id)
            print(f"[!] Switching Camera to ID: {cam_id}")

        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()