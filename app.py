# From https://saturncloud.io/blog/how-to-install-pytorch-on-the-gpu-with-docker/

# importing torch
import torch

# get index of currently selected device  
print(torch.cuda.current_device())
# get number of GPUs available  
print(torch.cuda.device_count())
# get the name of the device  
print(torch.cuda.get_device_name(0))
