import torch
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#nvidia-smi
print("PyTorch version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")
