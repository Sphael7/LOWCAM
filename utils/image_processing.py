import numpy as np
import cv2

def fast_preprocess(frame, target_size=(960, 720)):
    # Resize langsung ke buffer yang sudah ada
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Gunakan float16 untuk hemat RAM 50% dibanding float32
    blob = resized.transpose(2, 0, 1).astype(np.float16) 
    
    # In-place normalization (tidak membuat copy baru di RAM)
    blob /= 255.0 
    return np.expand_dims(blob, axis=0)