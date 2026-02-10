import datetime

class LowCamLogger:
    @staticmethod
    def log(message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [LOWCAM_LOG]: {message}")