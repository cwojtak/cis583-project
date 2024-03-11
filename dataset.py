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


def load_data(device):
    try:
        training_data = pandas.read_csv("data/master_data_cleaned_training.csv")
        evaluation_data = pandas.read_csv("data/master_data_cleaned_evaluation.csv")
    except OSError:
        raise RuntimeError("Failed to open a data file. You may be able to use clean_data.py to produce this file.")

    train_classes = training_data["Label"]
    eval_classes = evaluation_data["Label"]
    train_inputs = training_data.drop("Label", axis=1)
    eval_inputs = evaluation_data.drop("Label", axis=1)

    # Create tensor representations of the data
    tensor_eval_classes = torch.tensor((), dtype=torch.float32).new_zeros((len(eval_classes), 7)).to(device)
    for i in range(len(eval_classes)):
        tensor_eval_classes[i, eval_classes[i]] = 1
    tensor_train_classes = torch.tensor((), dtype=torch.float32).new_zeros((len(train_classes), 7)).to(device)
    for i in range(len(train_classes)):
        tensor_train_classes[i, train_classes[i]] = 1
    tensor_eval_inputs = torch.from_numpy(eval_inputs.to_numpy()).float().to(device)
    tensor_train_inputs = torch.from_numpy(train_inputs.to_numpy()).float().to(device)

    print("Training dataset composition: %i benign, %i SSH DoS, %i FTP DoS, %i GoldenEye DoS, %i Slowloris DoS, "
          "%i SlowHTTPTest DoS, and %i Hulk DoS" % (
              (train_classes == 0).sum(), (train_classes == 1).sum(), (train_classes == 2).sum(),
              (train_classes == 3).sum(), (train_classes == 4).sum(), (train_classes == 5).sum(),
              (train_classes == 6).sum()
          ))

    print("Evaluation dataset composition: %i benign, %i SSH DoS, %i FTP DoS, %i GoldenEye DoS, %i Slowloris DoS, "
          "%i SlowHTTPTest DoS, and %i Hulk DoS" % (
              (eval_classes == 0).sum(), (eval_classes == 1).sum(), (eval_classes == 2).sum(),
              (eval_classes == 3).sum(), (eval_classes == 4).sum(), (eval_classes == 5).sum(),
              (eval_classes == 6).sum()
          ))

    return (DoSDataset(tensor_train_inputs, tensor_train_classes),
            DoSDataset(tensor_eval_inputs, tensor_eval_classes))
