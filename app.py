import torch

from models import train_basic_model

if __name__ == '__main__':

    device = ("cuda"
              if torch.cuda.is_available()
              else "mps"
              if torch.backends.mps.is_available()
              else "cpu"
              )

    print("Device selected: %s" % device)
    if device == "cuda":
        print("CUDA device name: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))

    train_basic_model(device)
