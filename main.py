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

def get_current_ram_usage():
    """Mengambil penggunaan RAM spesifik untuk proses Python ini saja (dalam MB)."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    cv2.setNumThreads(4)
    cam = cv2.VideoCapture(0)
    monitor = ThermalMonitor(fps_target=30)
    eye = TemporalDiff(sensitivity=0.08)
    engine = ElasticEngine()
    rl = RLController()

    print("\n" + "="*55)
    print("      LOWCAM LIBRARY - ACTIVE & READY")
    print("="*55)
    print("KONTROL:")
    print("  [k] : Mulai/Berhenti Log & Lihat Detail RAM/Spek")
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

        input_frame = cv2.resize(frame, (960, 720), interpolation=cv2.INTER_NEAREST)
        input_tensor = input_frame.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32) / 255.0

        current_score = monitor.get_performance_score(fps)
        has_motion = eye.check_motion(input_tensor)

        if has_motion:
            output = engine.run(input_tensor, current_score)
            # Hapus input_tensor dari memori segera setelah diproses
            del input_tensor
        
        if int(fps) % 30 == 0:
            gc.collect()
        
        new_time = time.time()
        time_diff = new_time - last_time
        fps = 1 / time_diff if time_diff > 0 else 0
        last_time = new_time

        # --- LOGIKA PEREKAMAN DATA (Tombol K) ---
        current_time = time.time()
        if is_logging:
            if current_time - last_log_time >= 1.0:
                # Ambil RAM yang digunakan oleh aplikasi saat ini
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
                    
                    # Deteksi Spesifikasi Laptop
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

        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()