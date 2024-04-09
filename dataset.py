import os

import pandas
import torch.utils.data


class DoSDataset(torch.utils.data.Dataset):
    # Initialize the dataset. The inputs and classes should be stored in separate DataFrames
    def __init__(self, inputs, classes):
        self.inputs = inputs
        self.classes = classes

    # Function for finding the number of elements in the dataset
    def __len__(self):
        return len(self.classes)

    # Function for getting an input and its corresponding class from the dataset
    def __getitem__(self, i):
        return self.inputs[i], self.classes[i]


# Function for loading data
def load_data(device):
    # Try loading the master dataset
    try:
        training_data = pandas.read_csv("data/master_data_cleaned_training.csv")
        evaluation_data = pandas.read_csv("data/master_data_cleaned_evaluation.csv")
    except OSError:
        raise RuntimeError("Failed to open a data file. You may be able to use clean_data.py to produce this file.")

    # Separate labels from data
    train_classes = training_data["Label"]
    eval_classes = evaluation_data["Label"]
    train_inputs = training_data.drop("Label", axis=1)
    eval_inputs = evaluation_data.drop("Label", axis=1)

    # Create tensor representations of the data
    tensor_eval_classes = torch.tensor((), dtype=torch.float32).new_zeros((len(eval_classes), 9)).to(device)
    for i in range(len(eval_classes)):
        tensor_eval_classes[i, int(eval_classes[i])] = 1
    tensor_train_classes = torch.tensor((), dtype=torch.float32).new_zeros((len(train_classes), 9)).to(device)
    for i in range(len(train_classes)):
        tensor_train_classes[i, int(train_classes[i])] = 1
    tensor_eval_inputs = torch.from_numpy(eval_inputs.to_numpy()).float().to(device)
    tensor_train_inputs = torch.from_numpy(train_inputs.to_numpy()).float().to(device)

    # Print out dataset composition
    print("Training dataset composition: %i benign, %i SSH DoS, %i FTP DoS, %i GoldenEye DoS, %i Slowloris DoS, "
          "%i SlowHTTPTest DoS, %i Hulk DoS, %i HOIC DDoS, and %i Bot DDoS" % (
              (train_classes == 0).sum(), (train_classes == 1).sum(), (train_classes == 2).sum(),
              (train_classes == 3).sum(), (train_classes == 4).sum(), (train_classes == 5).sum(),
              (train_classes == 6).sum(), (train_classes == 7).sum(), (train_classes == 8).sum()
          ))

    print("Evaluation dataset composition: %i benign, %i SSH DoS, %i FTP DoS, %i GoldenEye DoS, %i Slowloris DoS, "
          "%i SlowHTTPTest DoS, %i Hulk DoS, %i HOIC DDoS, and %i Bot DDoS" % (
              (eval_classes == 0).sum(), (eval_classes == 1).sum(), (eval_classes == 2).sum(),
              (eval_classes == 3).sum(), (eval_classes == 4).sum(), (eval_classes == 5).sum(),
              (eval_classes == 6).sum(), (eval_classes == 7).sum(), (eval_classes == 8).sum()
          ))

    return (DoSDataset(tensor_train_inputs, tensor_train_classes),
            DoSDataset(tensor_eval_inputs, tensor_eval_classes))


# Load data from external files provided by the user
def load_external_data(device, path):
    # Try to load the data file and the normalization files
    try:
        data = pandas.read_csv(path)
        mean = pandas.read_csv("test/normalization/mean.csv")
        std = pandas.read_csv("test/normalization/std.csv")
        max = pandas.read_csv("test/normalization/max.csv")
    except OSError:
        raise RuntimeError("Failed to open the data file or normalization files. Try re-running clean_data.py?")

    # Separate labels from input
    classes = data["Label"]
    inputs = data.drop("Label", axis=1)

    # Apply the normalizations applied to the training and evaluation data to the user's data file
    for col in inputs.columns:
        inputs[col] = inputs[col] / max[col][0]
        inputs[col] = (inputs[col] - mean[col][0]) / std[col][0]

    # Create tensor representations of the data
    tensor_classes = torch.tensor((), dtype=torch.float32).new_zeros((len(classes), 9)).to(device)
    for i in range(len(classes)):
        tensor_classes[i, int(classes[i])] = 1
    tensor_inputs = torch.from_numpy(inputs.to_numpy()).float().to(device)

    # Print out dataset composition
    print("Dataset composition: %i benign, %i SSH DoS, %i FTP DoS, %i GoldenEye DoS, %i Slowloris DoS, "
          "%i SlowHTTPTest DoS, %i Hulk DoS, %i HOIC DDoS, and %i Bot DDoS" % (
              (classes == 0).sum(), (classes == 1).sum(), (classes == 2).sum(),
              (classes == 3).sum(), (classes == 4).sum(), (classes == 5).sum(),
              (classes == 6).sum(), (classes == 7).sum(), (classes == 8).sum()
          ))

    return DoSDataset(tensor_inputs, tensor_classes)
