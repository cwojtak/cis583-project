import torch
import torch.nn as nn
import torch.utils.data
import time

from dataset import load_data


# Basic model for classifying packets as part of various types of DDoS attacks
class BasicDoSModel(nn.Module):
    # Construct the model with a three layer stack
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(68, 38),
            nn.ReLU(),
            nn.Linear(38, 15),
            nn.ReLU(),
            nn.Linear(15, 7),
            nn.Sigmoid()
        )

    # Function to compute the forward pass
    def forward(self, x):
        self.flatten(x)
        return self.stack(x)


# Training function for the basic model
def train_basic_model(device):
    # Define hyperparameters
    epochs = 50
    batch_size = 256
    learning_rate = 0.75

    # Create model and define loss function and optimizer
    model = BasicDoSModel().to(device)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_dataset, eval_dataset = load_data(device)

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    eval_dataset_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

    start_time = time.time()

    print("Initial state of the model\n====================================")
    evaluate_basic_model(model, eval_dataset_loader, loss_func)

    for i in range(epochs):
        model.train()

        print("Epoch %4d\n====================================" % i)
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
    end_time = time.time()
    print("Total time elapsed: %.2f" % (end_time - start_time))
    print("Training complete! Saving model...")

    torch.save(model, "models/basic_model.pth")

    print("Save complete!")


# Evaluation function for the basic model
def evaluate_basic_model(model, eval_dataset_loader, loss_func):
    print("Evaluating the model...")

    # Prepare statistics
    num_correct = 0
    total = len(eval_dataset_loader.dataset)
    avg_loss = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # Evaluate without calculating gradients
    model.eval()
    with torch.no_grad():
        for i, (inputs, true_class) in enumerate(eval_dataset_loader):
            # Run data through the model and calculate loss
            model_result = model(inputs)

            avg_loss += loss_func(model_result, true_class).item()

            model_result = model_result.argmax(dim=1)
            true_class = true_class.argmax(dim=1)

            # Determine the raw number of correct guesses
            num_correct += (model_result == true_class).sum().item()

            # Update tp, tn, fp, fn
            model_results_dos = model_result != 0
            true_class_dos = true_class != 0
            current_tp = torch.logical_and(model_results_dos, true_class_dos).sum().item()
            current_tn = torch.logical_and(torch.logical_not(model_results_dos),
                                           torch.logical_not(true_class_dos)).sum().item()
            tp += current_tp
            tn += current_tn
            fp += model_results_dos.sum().item() - current_tp
            fn += torch.logical_not(model_results_dos).sum().item() - current_tn

    avg_loss /= total
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fp != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    accuracy = num_correct / total

    print("Evaluation complete:\n Number Correct: (%6d/%6d)\n Accuracy: %2.8f\n Precision: %2.8f\n Recall: %2.8f\n "
          "F1 Score: %2.8f\n Average Loss: %2.8f\n" % (num_correct, total, accuracy, precision, recall, f1_score,
                                                       avg_loss))
