import torch.nn as nn
import torch.utils.data

from models import train_basic_model, evaluate_basic_model
from dataset import load_data

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

    # Train model
    train_basic_model(device)

    # Evaluate model
    train_dataset, eval_dataset = load_data("02-14-2018", device)
    eval_dataset_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=256)
    loss_func = nn.MSELoss()
    evaluate_basic_model(torch.load("models/basic_model.pth"), eval_dataset_loader, loss_func)
