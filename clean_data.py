import os

import math
import pandas as pd
import numpy as np


def main():
    # for files in data folder go through all the csvs and store the names in a list
    # files = os.listdir('data')
    #only cleaning one file for midterm
    files = ["02-14-2018.csv"]
    path = "data/"

    for file_name in files:
        data = pd.read_csv(os.path.join(path, file_name))

        # top = files.head()
        # description = files.describe()
        # description.to_csv("description.csv")

        # drop timestamp column for now
        # TODO: determine if the there is a way to use timestamp in neural network
        data.drop(columns=["Timestamp"], inplace=True)

        # remove columns that are all either 0 or 1
        non_zero_file = data.loc[:, (data != 0).any(axis=0)]
        # non_zero_data.to_csv("non_zero_data.csv", index=False)
        nz_description = non_zero_file.describe()
        # nz_description.to_csv("nz_description.csv")

        # remove columns with zero std. This would remove any cols with the same value
        # did not do anything to this file, but could help with other files
        cols = non_zero_file.select_dtypes([np.number]).columns
        std = non_zero_file[cols].std()
        cols_to_drop = std[std == 0].index
        nz_stdzero_file = non_zero_file.drop(cols_to_drop, axis=1)
        # nz_stdzero_file.to_csv("nz_stdzero.csv")
        nz_stdzero_desc = non_zero_file.describe()
        # nz_stdzero_desc.to_csv("nz_stdzero_desc.csv")

        # following section from geeksforgeeks.org/data-normalization-with-pandas/
        df_to_normalize = nz_stdzero_file.copy()

        # apply normalization techniques
        for column in df_to_normalize.columns:
            # check for string types in columns (can't take standard dev. of strings)
            if type(df_to_normalize[column][0]) is str:
                continue

            df_to_normalize[column] = df_to_normalize[column] / df_to_normalize[column].abs().max()

        # TODO: Look at standardize using z-score method? Returns values with mean=0 and std=1

        # Translate string labels to numbers. 0 = benign, 1 = FTP, 2 = SSH
        df_to_normalize["Label"] = df_to_normalize["Label"].map({"Benign": 0, "FTP-BruteForce": 1, "SSH-Bruteforce": 2})

        # drop rows with empty values
        labels = df_to_normalize["Label"]
        df_to_normalize = df_to_normalize.drop(["Label"], axis=1)
        df_to_normalize = df_to_normalize.astype(np.float32)
        df_to_normalize = df_to_normalize.join(labels)
        df_to_normalize = df_to_normalize.dropna()

        # Shuffle data and split into training and evaluation data
        df_to_normalize = df_to_normalize.sample(frac=1)
        training_df = df_to_normalize.head(math.floor(len(df_to_normalize.index) / 2))
        evaluation_df = df_to_normalize.tail(math.floor(len(df_to_normalize.index) / 2))

        # rename file to ***cleaned.csv
        training_file_name_cleaned_csv = file_name.replace(".csv", "_cleaned_training.csv")
        evaluation_file_name_cleaned_csv = file_name.replace(".csv", "_cleaned_evaluation.csv")
        training_df.to_csv(os.path.join(path, training_file_name_cleaned_csv), index=False)
        evaluation_df.to_csv(os.path.join(path, evaluation_file_name_cleaned_csv), index=False)
        print(file_name + " has been cleaned")


if __name__ == '__main__':
    main()
