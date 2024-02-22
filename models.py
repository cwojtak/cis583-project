import torch
import torch.nn as nn
import torch.utils.data

from dataset import load_data


# Basic model for classifying packets as part of various types of DDoS attacks
class BasicDDoSModel(nn.Module):
    # Construct the model with a three layer stack
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(75, 38),
            nn.ReLU(),
            nn.Linear(38, 15),
            nn.ReLU(),
            nn.Linear(15, 3),
            nn.Softmax(3)
        )


# Training function for the basic model
def train_basic_model(device):
    # Define hyperparameters
    epochs = 250
    batch_size = 128
    learning_rate = 0.1

    # Create model and define loss function and optimizer
    model = BasicDDoSModel().to(device)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset, eval_dataset = load_data("data/02-14-2018.csv")

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    eval_dataset_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

    print("Initial state of the model\n====================================")
    evaluate_basic_model(model, eval_dataset_loader, loss_func)

    for i in range(epochs):
        model.train()

        print("Epoch %4d\n====================================")
        for j, (inputs, true_class) in enumerate(train_dataset_loader):
            # Begin forward pass
            model_result = model(inputs)
            loss = loss_func(model_result, true_class)

            # Begin backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Evaluate model with evaluation data
        evaluate_basic_model(model, eval_dataset_loader, loss_func)

    # Save model
    print("Training complete! Saving model...")

    torch.save(model, "models/basic_model.pth")

    print("Save complete!")


# Evaluation function for the basic model
def evaluate_basic_model(model, eval_dataset_loader, loss_func):
    # Prepare statistics
    num_correct = 0
    total = len(eval_dataset_loader.dataset)
    avg_loss = 0

    # Evaluate without calculating gradients
    model.eval()
    with torch.no_grad():
        for i, (inputs, true_class) in enumerate(eval_dataset_loader):
            # Run data through the model and calculate loss
            model_result = model(inputs)
            avg_loss = loss_func(model_result, true_class)

            # Determine the raw number of correct guesses
            model_result = torch.round(model_result.argmax(1))
            num_correct += torch.sum(torch.logical_and(model_result, true_class))
            num_correct += torch.sum(torch.logical_and(torch.logical_not(model_result),
                                                       torch.logical_not(true_class)))

    avg_loss /= total

    print("Evaluation complete: \n Number Correct: (%6d/%6d) \n Average Loss: %2.5f" % (num_correct, total, avg_loss))
