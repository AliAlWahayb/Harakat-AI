import torch
import sys

def check_gpu():
    print("PyTorch version:", torch.__version__)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        print("CUDA version:", torch.version.cuda)
        
        # Get number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        # Print GPU information
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Test GPU computation
        print("\nTesting GPU computation...")
        
        # Create tensors on CPU and GPU
        x_cpu = torch.randn(1000, 1000)
        x_gpu = x_cpu.cuda()
        
        # Perform matrix multiplication
        import time
        
        # CPU timing
        start_time = time.time()
        result_cpu = torch.matmul(x_cpu, x_cpu)
        cpu_time = time.time() - start_time
        
        # GPU timing
        start_time = time.time()
        result_gpu = torch.matmul(x_gpu, x_gpu)
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
        
        print(f"CPU computation time: {cpu_time:.4f} seconds")
        print(f"GPU computation time: {gpu_time:.4f} seconds")
        print(f"GPU speedup: {cpu_time / gpu_time:.2f}x")
        
        # Check if results match
        result_from_gpu = result_gpu.cpu()
        is_close = torch.allclose(result_cpu, result_from_gpu, rtol=1e-3, atol=1e-3)
        print(f"Results match: {is_close}")
        
        return True
    else:
        print("CUDA is not available. Running on CPU only.")
        return False

if __name__ == "__main__":
    is_gpu_available = check_gpu()
    
    if not is_gpu_available:
        print("\nGPU is not available. Please check your CUDA installation.")
        sys.exit(1)
    else:
        print("\nGPU is working correctly!")
        sys.exit(0)