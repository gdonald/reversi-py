import torch


def get_device(preferred_device=None, verbose=True):
    if preferred_device is not None:
        preferred_device = preferred_device.lower()

        if preferred_device == "mps":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                if verbose:
                    print(f"Using user-specified device: MPS (Apple Silicon)")
                return device
            else:
                if verbose:
                    print(
                        f"Warning: MPS requested but not available. Falling back to auto-detection."
                    )

        elif preferred_device == "cuda":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                if verbose:
                    print(
                        f"Using user-specified device: CUDA (GPU: {torch.cuda.get_device_name(0)})"
                    )
                return device
            else:
                if verbose:
                    print(
                        f"Warning: CUDA requested but not available. Falling back to auto-detection."
                    )

        elif preferred_device == "cpu":
            device = torch.device("cpu")
            if verbose:
                print(f"Using user-specified device: CPU")
            return device

        else:
            if verbose:
                print(
                    f"Warning: Unknown device '{preferred_device}'. Using auto-detection."
                )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print(f"Auto-detected device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Auto-detected device: CUDA (GPU: {gpu_name})")
    else:
        device = torch.device("cpu")
        if verbose:
            print(f"Auto-detected device: CPU (no GPU acceleration available)")

    return device


def print_device_info():
    print("\n=== PyTorch Device Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print()

    print("MPS (Apple Silicon):")
    if hasattr(torch.backends, "mps"):
        mps_available = torch.backends.mps.is_available()
        print(f"  Available: {mps_available}")
        if mps_available:
            print(f"  Built: {torch.backends.mps.is_built()}")
    else:
        print(f"  Not supported in this PyTorch version")
    print()

    print("CUDA (NVIDIA GPU):")
    cuda_available = torch.cuda.is_available()
    print(f"  Available: {cuda_available}")
    if cuda_available:
        print(f"  Version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
            )
    print()

    print("CPU:")
    print(f"  Available: True (always)")
    print(f"  Thread count: {torch.get_num_threads()}")
    print()

    device = get_device(verbose=False)
    print(f"Default selected device: {device}")
    print("=" * 35)
    print()


if __name__ == "__main__":
    print_device_info()
