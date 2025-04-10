import torch
import sys

def print_gpu_info():
    """
    Print detailed information about available GPU devices.
    """
    print("\n" + "="*50)
    print("GPU INFORMATION")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU only.")
        return
    
    # Get number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")
    
    # Get current device
    current_device = torch.cuda.current_device()
    print(f"Current device index: {current_device}")
    
    # Print information for each GPU
    for i in range(gpu_count):
        print("\n" + "-"*40)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Device capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        if i == current_device:
            print("  [CURRENT DEVICE]")
            # Print memory usage for current device
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    
    print("\n" + "="*50)

def test_gpu():
    """
    Perform a simple test on GPU by creating a tensor and checking its device.
    """
    if torch.cuda.is_available():
        print("\nTesting GPU with a simple tensor operation...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        z = x @ y  # Matrix multiplication
        end.record()
        
        # Waits for everything to finish execution
        torch.cuda.synchronize()
        
        print(f"Device of result tensor: {z.device}")
        print(f"Time to perform matrix multiplication: {start.elapsed_time(end):.2f} ms")
        print("\nGPU test successful!")
    else:
        print("\nCannot test GPU - CUDA not available.")

if __name__ == "__main__":
    print_gpu_info()
    
    # Run GPU test if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_gpu()
