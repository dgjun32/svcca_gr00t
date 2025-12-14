#!/usr/bin/env python3
"""
Dummy GPU process that consumes GPU utilization (not just memory).
Continuously performs matrix multiplications to keep GPU busy.
"""

import torch
import argparse
import time
import sys


def consume_gpu(gpu_id=0, matrix_size=8192, utilization_target=0.8):
    """
    Consume GPU utilization by continuously performing matrix operations.
    
    Args:
        gpu_id: GPU device ID to use
        matrix_size: Size of matrices for computation (larger = more GPU usage)
        utilization_target: Target GPU utilization (0.0 to 1.0)
    """
    device = torch.device(f"cuda:{gpu_id}")
    
    print(f"Starting dummy GPU process on GPU {gpu_id}")
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"Target utilization: {utilization_target * 100:.1f}%")
    print(f"Press Ctrl+C to stop")
    print("-" * 50)
    
    # Create large matrices on GPU
    A = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
    B = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
    
    iteration = 0
    try:
        while True:
            # Perform GPU-intensive operations
            # Matrix multiplication is very GPU-intensive
            C = torch.matmul(A, B)
            
            # Additional operations to increase utilization
            D = torch.matmul(C, A)
            E = torch.sin(D)
            F = torch.exp(E * 0.01)  # Scale down to avoid overflow
            
            # Update matrices to prevent optimization
            A = F * 0.1 + A * 0.9
            B = C * 0.1 + B * 0.9
            
            # Synchronize to ensure operations complete
            torch.cuda.synchronize()
            
            iteration += 1
            
            # Print status every 100 iterations
            if iteration % 100 == 0:
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f"Iteration {iteration:6d} | "
                      f"Memory: {memory_allocated:.2f}GB allocated, "
                      f"{memory_reserved:.2f}GB reserved")
            
            # Adjust sleep time based on target utilization
            # Lower utilization = more sleep
            if utilization_target < 1.0:
                sleep_time = (1.0 - utilization_target) * 0.01
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\n" + "-" * 50)
        print("Stopping dummy GPU process...")
        print(f"Total iterations: {iteration}")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dummy GPU process that consumes GPU utilization"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0)"
    )
    parser.add_argument(
        "--matrix_size",
        type=int,
        default=40000,
        help="Size of matrices for computation (default: 8192). "
             "Larger = more GPU utilization and memory. "
             "Try 4096 for ~50%%, 8192 for ~80%%, 12288 for ~95%%"
    )
    parser.add_argument(
        "--utilization",
        type=float,
        default=0.8,
        help="Target GPU utilization from 0.0 to 1.0 (default: 0.8)"
    )
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        sys.exit(1)
    
    # Check if GPU ID is valid
    if args.gpu_id >= torch.cuda.device_count():
        print(f"ERROR: GPU {args.gpu_id} not found. "
              f"Available GPUs: {torch.cuda.device_count()}")
        sys.exit(1)
    
    # Validate utilization target
    if not 0.0 <= args.utilization <= 1.0:
        print("ERROR: utilization must be between 0.0 and 1.0")
        sys.exit(1)
    
    consume_gpu(args.gpu_id, args.matrix_size, args.utilization)

