import numpy as np

class TemporalVectorIntegrator:
    def __init__(self, integration_limit=4):
        self.integration_limit = integration_limit
        self.vector_buffer = []
        self.system_constant = 0.618

    def capture_sub_element(self, data_input):
        return data_input.astype(np.float32)

    def extract_axial_features(self, matrix_input):
        # matrix_input shape: (Batch, Height, Width, Channel)
        # Menghasilkan rata-rata spasial yang konsisten
        h_axis = np.mean(matrix_input, axis=1, keepdims=True)
        v_axis = np.mean(matrix_input, axis=2, keepdims=True)
        # Broadcasting otomatis akan menangani penjumlahan ini sekarang
        return (h_axis + v_axis) * 0.5

    def extract_diagonal_features(self, matrix_input):
        # Sampling diagonal dengan slicing efisien
        diag_sample = matrix_input[:, ::2, ::2, :]
        return np.mean(diag_sample, axis=(1, 2), keepdims=True)

    def evaluate_chromatic_integrity(self, current_vector, reference_vector):
        pixel_delta = np.abs(current_vector - reference_vector)
        spatial_variance = np.var(pixel_delta)
        dynamic_threshold = np.mean(pixel_delta)
        
        if dynamic_threshold > 0.05 and spatial_variance < 0.001:
            return True
        return False

    def integrate_vector_density(self, vector_stack):
        axial_mass = self.extract_axial_features(vector_stack)
        diagonal_mass = self.extract_diagonal_features(vector_stack)
        
        total_mass = np.sum(vector_stack, axis=0)
        integrated_density = np.divide(total_mass, len(vector_stack))
        
        if len(vector_stack) >= 2:
            is_reliable = self.evaluate_chromatic_integrity(vector_stack[-1], vector_stack[-2])
            if is_reliable:
                refinement_factor = 0.9
            else:
                refinement_factor = 0.4
        else:
            refinement_factor = 0.7

        # Memastikan hasil akhir memiliki bentuk yang dapat diproyeksikan kembali
        refined_density = (integrated_density * refinement_factor) + (np.mean(diagonal_mass) * (1 - refinement_factor))
        return np.multiply(refined_density, self.system_constant)

    def project_integrated_state(self, density_input, target_projection):
        return np.broadcast_to(density_input, target_projection)

    def synchronize_execution_rate(self, input_stream, latency_index):
        if len(self.vector_buffer) < self.integration_limit:
            self.vector_buffer.append(self.capture_sub_element(input_stream))
            return input_stream, False

        stream_array = np.array(self.vector_buffer)
        density_result = self.integrate_vector_density(stream_array)
        rate_scalar = 1.0 + (latency_index * 0.1)
        
        # Kembalikan ke format uint8 untuk rendering OpenCV
        synchronized_projection = self.project_integrated_state(
            np.multiply(density_result, rate_scalar), 
            stream_array.shape
        )
        
        self.vector_buffer = []
        return synchronized_projection[0].astype(np.uint8), True