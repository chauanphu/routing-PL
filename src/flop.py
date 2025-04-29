import numpy as np
import time

def estimate_flops(matrix_size=1000, num_trials=3):
    """
    Estimate FLOPS by timing a large matrix multiplication.
    Args:
        matrix_size (int): Size of the square matrices.
        num_trials (int): Number of trials to average.
    Returns:
        float: Estimated FLOPS (floating point operations per second).
    """
    flops_per_mul = 2 * (matrix_size ** 3)  # Each multiplication is 2*N^3 FLOPs
    times = []
    for _ in range(num_trials):
        A = np.random.rand(matrix_size, matrix_size)
        B = np.random.rand(matrix_size, matrix_size)
        start = time.time()
        np.dot(A, B)
        end = time.time()
        times.append(end - start)
    avg_time = sum(times) / num_trials
    estimated_flops = flops_per_mul / avg_time
    return estimated_flops, avg_time

if __name__ == "__main__":
    size = 1000
    trials = 3
    print(f"Estimating FLOPS with {size}x{size} matrix multiplication, {trials} trials...")
    flops, avg_time = estimate_flops(matrix_size=size, num_trials=trials)
    print(f"Average time per multiplication: {avg_time:.4f} seconds")
    print(f"Estimated FLOPS: {flops/1e9:.2f} GFLOPS")