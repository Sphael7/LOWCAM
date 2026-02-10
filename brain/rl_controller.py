import numpy as np

class RLController:
    def __init__(self):
        # Q-Table sederhana: [Status_Laptop, Aksi_Model]
        # Status: 0=Dingin, 1=Hangat, 2=Panas
        # Aksi: 0=Hemat, 1=Normal, 2=High-Power
        self.q_table = np.zeros((3, 3))
        self.learning_rate = 0.1
        self.last_state = 0
        self.last_action = 1

    def select_action(self, cpu_load):
        # Tentukan status berdasarkan beban CPU
        state = 0 if cpu_load < 40 else (1 if cpu_load < 75 else 2)
        
        # Pilih aksi terbaik dari pengalaman (Exploitation)
        action = np.argmax(self.q_table[state])
        
        self.last_state = state
        self.last_action = action
        return action # Output ini dikirim ke ElasticEngine

    def update_knowledge(self, current_fps, target_fps=30):
        # Berikan reward jika FPS mendekati target
        reward = 1 if current_fps >= target_fps else -1
        
        # Update Q-Table (Belajar dari hasil)
        old_val = self.q_table[self.last_state, self.last_action]
        self.q_table[self.last_state, self.last_action] = old_val + self.learning_rate * (reward - old_val)