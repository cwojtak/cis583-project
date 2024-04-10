import torch
import torch.nn as nn
import torch.utils.data
import time
from datetime import datetime
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import KFold

from dataset import load_data


# Model for classifying packets as part of various types of DDoS attacks
class BasicDoSModel(nn.Module):
    # Construct the model with a three layer stack
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(27, 24),
            nn.ReLU(),
            nn.Linear(24, 20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 9),
            nn.Sigmoid()
        )

    # Function to compute the forward pass
    def forward(self, x):
        self.flatten(x)
        return self.stack(x)

    def get_stack(self):
        return self.stack


# Training function for the model
def train_model(device):
    # Define hyperparameters
    epochs = 30
    batch_size = 128
    learning_rate = 0.001
    k_folds = 10

    # Create model and define loss function and optimizer
    model = BasicDoSModel().to(device)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset, eval_dataset = load_data(device)

    # train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    eval_dataset_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

    # Prepare method to store graph data
    graph_data = pd.DataFrame(columns=["Accuracy", "Binary Precision", "Binary Recall", "Binary F1 Score",
                                       "Average Loss", "All-Class Precision", "All-Class Recall",
                                       "All-Class F1 Score"])

    # Prepare folds
    kf = KFold(n_splits=k_folds, shuffle=True)

    # Keep track of best model
    best_model = copy.deepcopy(model)
    best_accuracy = 0

    start_time = time.time()

    print("Initial state of the model\n====================================")
    evaluate_model(model, eval_dataset_loader, loss_func)

    # Perform training loop
    for i in range(epochs):
        model.train()

        # Create folds and train on each of them
        for fold, (train_idx, eval_idx) in enumerate(kf.split(train_dataset)):
            fold_train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(train_idx)
            )

            print("Epoch %4d-F%4d\n====================================" % (i, fold))
            for j, (inputs, true_class) in enumerate(fold_train_loader):
                # Begin forward pass
                model_result = model(inputs)
                loss = loss_func(model_result, true_class)

                # Begin backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # Evaluate model with evaluation data
        data_row = evaluate_model(model, eval_dataset_loader, loss_func)
        graph_data = pd.concat([graph_data, data_row])

        # Keep track of the best model
        if data_row["Accuracy"][0] > best_accuracy:
            best_accuracy = data_row["Accuracy"][0]
            best_model = copy.deepcopy(model)

    # Save the best model and other metrics data
    end_time = time.time()
    print("Total time elapsed: %.2f" % (end_time - start_time))
    print("Training complete! Saving model...")

    model = best_model

    torch.save(model, "models/final_model.pth")
    graph_data.to_csv("raw_results/data_%s.csv" % datetime.now().strftime("%m-%d-%Y_%H-%M-%S"), index=False)

    print("Save complete!")


# Evaluation function for the basic model
def evaluate_model(model, eval_dataset_loader, loss_func, external_classify=False):
    print("Evaluating the model...")

    # Prepare statistics
    num_correct = 0
    total = len(eval_dataset_loader.dataset)
    avg_loss = 0
    tp = np.zeros(9)
    tn = np.zeros(9)
    fp = np.zeros(9)
    fn = np.zeros(9)

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
            for j in range(9):
                model_results_dos = model_result == j
                true_class_dos = true_class == j
                current_tp = torch.logical_and(model_results_dos, true_class_dos).sum().item()
                current_tn = torch.logical_and(torch.logical_not(model_results_dos),
                                               torch.logical_not(true_class_dos)).sum().item()
                tp[j] += current_tp
                tn[j] += current_tn
                fp[j] += model_results_dos.sum().item() - current_tp
                fn[j] += torch.logical_not(model_results_dos).sum().item() - current_tn

    # Calculate the metrics
    avg_loss /= total
    precision = np.divide(tp, (tp + fp), where=(tp + fp != 0))
    recall = np.divide(tp, (tp + fn), where=(tp + fn != 0))
    f1_score = np.divide(2 * precision * recall, (precision + recall), where=(precision + recall != 0))
    accuracy = num_correct / total

    # Output the metrics
    print("Evaluation complete:\n Number Correct: (%6d/%6d)\n Accuracy: %2.8f\n Binary Precision: %2.8f\n Binary "
          "Recall: %2.8f\n Binary F1 Score: %2.8f\n Average Loss: %2.8f\n All-Class Precision: %2.8f\n "
          "All-Class Recall: %2.8f\n All-Class F1 Score: %2.8f\n"
          % (num_correct, total, accuracy, precision[0], recall[0], f1_score[0], avg_loss,
             np.average(precision), np.average(recall), np.average(f1_score)))

    return pd.DataFrame({"Accuracy": [accuracy], "Binary Precision": [precision[0]], "Binary Recall": [recall[0]],
                         "Binary F1 Score": [f1_score[0]], "Average Loss": [avg_loss],
                         "All-Class Precision": [np.average(precision)], "All-Class Recall": [np.average(recall)],
                         "All-Class F1 Score": [np.average(f1_score)]})
