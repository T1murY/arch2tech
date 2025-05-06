import torch
import tensorflow as tf

# Check PyTorch CUDA
pytorch_cuda = torch.cuda.is_available()
if pytorch_cuda:
    print(f"PyTorch CUDA is available: {pytorch_cuda}")
    print(f"PyTorch GPU count: {torch.cuda.device_count()}")
    print(f"PyTorch GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch CUDA is not available.")

# Check TensorFlow CUDA
tf_cuda = tf.config.list_physical_devices('GPU')
if tf_cuda:
    print(f"TensorFlow CUDA is available: {tf_cuda}")
else:
    print("TensorFlow CUDA is not available.")