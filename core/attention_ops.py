import numpy as np

class PixelShufflingAttention:
    def __init__(self, block_size=2):
        self.block_size = block_size

    def forward(self, x):
        """
        Mengacak piksel antar channel untuk meningkatkan 'receptive field'
        tanpa menambah beban komputasi konvolusi.
        """
        b, c, h, w = x.shape
        bs = self.block_size
        
        # 1. Reshape untuk memisahkan blok spasial
        x = x.reshape(b, c, h // bs, bs, w // bs, bs)
        
        # 2. Transpose untuk mengacak urutan (shuffling)
        x = x.transpose(0, 1, 3, 5, 2, 4)
        
        # 3. Kembalikan ke bentuk semula dengan channel yang sudah 'tercampur'
        x = x.reshape(b, c * (bs**2), h // bs, w // bs)
        
        # Gunakan 1x1 conv (sederhana) untuk mengembalikan dimensi channel
        # Di versi Pure NumPy, kita cukup melakukan mean pooling di dimensi channel
        return np.mean(x, axis=1, keepdims=True)