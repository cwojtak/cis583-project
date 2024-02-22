import os

import pandas
import torch.utils.data


class DDoSDataset(torch.utils.data.Dataset):
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


def load_data(path):
    if os.path.isfile(path):
        data = pandas.read_csv(path)

    train_inputs, train_classes, eval_inputs, eval_classes = None, None, None, None  # TODO: prepare data

    return DDoSDataset(train_inputs, train_classes), DDoSDataset(eval_inputs, eval_classes)
