import torch.nn as nn
import torch.utils.data

from models import train_model, evaluate_model
from dataset import load_data, load_external_data

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

    print("Do you want to train, evaluate, or classify external data with the model (t/e/c)?")
    user_input = input()

    if user_input == "t":
        # Train model
        train_model(device)
    elif user_input == "e":
        # Evaluate model
        train_dataset, eval_dataset = load_data(device)
        eval_dataset_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=256)
        loss_func = nn.MSELoss()
        evaluate_model(torch.load("models/final_model.pth"), eval_dataset_loader, loss_func)
    else:
        print("Enter the path to the data you would like to classify.")
        path = input()
        data = load_external_data(device, path)
        data_loader = torch.utils.data.DataLoader(data, batch_size=256)
        loss_func = nn.MSELoss()
        evaluate_model(torch.load("models/final_model.pth"), data_loader, loss_func, True)
