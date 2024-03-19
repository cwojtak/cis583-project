import os

import math
import pandas as pd
import numpy as np


def main():
    # Prepare each data file we are going to use
    files = ["02-14-2018.csv", "02-15-2018.csv", "02-16-2018.csv", "02-21-2018.csv", "03-02-2018.csv"]
    path = "data/"

    big_data_df = pd.DataFrame()

    # Combine CSV files into one big DataFrame
    for file_name in files:
        print("Preparing " + file_name)

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

        big_data_df = pd.concat([big_data_df, df_to_normalize], ignore_index=True)

    # Apply normalization techniques
    print("Cleaning the data...")

    for column in big_data_df.columns:
        # Check for string types in columns (can't take standard dev. of strings)
        if type(big_data_df[column][0]) is str:
            continue

        big_data_df[column] = big_data_df[column] / big_data_df[column].abs().max()

    # TODO: Look at standardize using z-score method? Returns values with mean=0 and std=1

    # Translate string labels to numbers
    big_data_df["Label"] = big_data_df["Label"].map({"Benign": 0, "FTP-BruteForce": 1, "SSH-Bruteforce": 2,
                                                     "DoS attacks-GoldenEye": 3,
                                                     "DoS attacks-Slowloris": 4,
                                                     "DoS attacks-SlowHTTPTest": 5,
                                                     "DoS attacks-Hulk": 6,
                                                     "DDOS attack-LOIC-UDP": -1,
                                                     "DDOS attack-HOIC": 7,
                                                     "Bot": 8})

    # Not enough instances of LOIC to train, drop them
    bad_rows = big_data_df["Label"] == -1
    big_data_df = big_data_df[~bad_rows]

    # The dataset contains some rows that accidentally contain the column labels again; drop these
    bad_rows = big_data_df["Dst Port"] == "Dst Port"
    big_data_df = big_data_df[~bad_rows]

    # Drop rows with empty values
    labels = big_data_df["Label"]
    big_data_df = big_data_df.drop(["Label"], axis=1)
    big_data_df = big_data_df.astype(np.float32)
    big_data_df = big_data_df.join(labels)
    big_data_df = big_data_df.dropna()

    # Shuffle data and split into training and evaluation data
    big_data_df = big_data_df.sample(frac=1)
    big_training_df = big_data_df.head(math.floor(len(big_data_df.index) / 2))
    big_evaluation_df = big_data_df.tail(math.floor(len(big_data_df.index) / 2))

    # Stratified sampling of all the data
    print("Performing stratified sampling of the data")

    big_training_df = big_training_df.groupby("Label", group_keys=False).apply(
        # lambda group: group.sample(min(len(group), 50000)))
        lambda group: group.sample(frac=0.2))
    big_evaluation_df = big_evaluation_df.groupby("Label", group_keys=False).apply(
        # lambda group: group.sample(min(len(group), 50000)))
        lambda group: group.sample(frac=0.2))

    # Save sampled data
    big_training_df.to_csv("data/master_data_cleaned_training.csv", index=False)
    big_evaluation_df.to_csv("data/master_data_cleaned_evaluation.csv", index=False)


if __name__ == '__main__':
    main()
