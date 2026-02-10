import psutil
import time
import platform

class ThermalMonitor:
    def __init__(self, fps_target=30):
        self.fps_target = fps_target
        self.os_type = platform.system()
        print(f"[LOWCAM] Thermal Monitor initialized on {self.os_type}")

    def get_cpu_load(self):
        """Mengambil persentase penggunaan CPU saat ini."""
        return psutil.cpu_percent(interval=None)

    def get_thermal_status(self):
        """
        Mengambil suhu CPU. 
        Catatan: Beberapa laptop Windows memerlukan akses Administrator 
        atau library tambahan seperti OpenHardwareMonitor.
        """
        temp = 0
        try:
            if self.os_type == "Linux":
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    temp = temps['coretemp'][0].current
            elif self.os_type == "Windows":
                # Windows tidak memberikan akses suhu via psutil secara native dengan mudah
                # Kita asumsikan suhu berdasarkan beban CPU jika sensor terkunci
                temp = self.get_cpu_load() # Proxy sederhana untuk simulasi awal
            return temp
        except Exception:
            return 0

    def get_performance_score(self, current_fps):
        """
        Algoritma 'Brain' awal: Menghitung skor kesehatan sistem.
        0.0 = Laptop sekarat (lag parah)
        1.0 = Laptop lancar jaya
        """
        cpu_usage = self.get_cpu_load()
        
        # Penalti jika FPS di bawah target
        fps_ratio = min(current_fps / self.fps_target, 1.0)
        
        # Penalti jika CPU di atas 85% (Throttling zone)
        cpu_penalty = 1.0
        if cpu_usage > 85:
            cpu_penalty = 0.5
        elif cpu_usage > 70:
            cpu_penalty = 0.8
            
        score = fps_ratio * cpu_penalty
        return score

# --- Testing Script ---
if __name__ == "__main__":
    monitor = ThermalMonitor()
    while True:
        score = monitor.get_performance_score(current_fps=15) # Contoh FPS drop
        print(f"CPU Load: {monitor.get_cpu_load()}% | System Health Score: {score:.2f}")
        time.sleep(1)