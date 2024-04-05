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

        # Drop columns with little effect on the results
        """
        data = data[["ACK Flag Cnt", "PSH Flag Cnt", "RST Flag Cnt", "ECE Flag Cnt", "Init Fwd Win Byts",
                    "Dst Port", "Init Bwd Win Byts", "Protocol", "URG Flag Cnt", "Bwd IAT Tot", "Fwd Seg Size Min",
                    "SYN Flag Cnt", "Fwd PSH Flags", "Bwd Pkt Len Std", "Bwd Seg Size Avg", "Bwd Pkt Len Mean",
                    "Bwd Pkts/s", "Bwd IAT Max", "Fwd Pkts/s", "FIN Flag Cnt", "Bwd Pkt Len Max", "Bwd IAT Std",
                    "Pkt Size Avg", "Bwd IAT Mean", "Pkt Len Mean", "Flow Duration", "Tot Fwd Pkts",
                    "Tot Bwd Pkts", "Label"]]
        """
        # data.drop(columns=["Timestamp"], inplace=True)
        data.drop(columns=["Timestamp", "Flow Byts/s", "Flow Pkts/s", "Bwd Pkts/s",
                           "Idle Min", "Idle Max", "Idle Std", "Idle Mean",
                           "Active Min", "Active Max", "Active Std", "Active Mean",
                           "Flow IAT Std", "Fwd IAT Std", "Flow IAT Min", "Fwd IAT Min",
                           "Fwd IAT Max", "Flow IAT Max", "Fwd IAT Tot", "Flow IAT Mean",
                           "Fwd IAT Mean", "Fwd Act Data Pkts", "Subflow Fwd Pkts",
                           "Tot Fwd Pkts", "Fwd Header Len", "Pkt Len Var", "Bwd Header Len",
                           "TotLen Bwd Pkts", "Subflow Bwd Byts", "Subflow Fwd Byts",
                           "TotLen Fwd Pkts", "Subflow Bwd Pkts", "Tot Bwd Pkts", "Down/Up Ratio", "Fwd Pkt Len Mean",
                           "Fwd Seg Size Avg", "Fwd Pkt Len Max", "Pkt Len Max", "Fwd Pkt Len Min", "Pkt Len Std",
                           "Bwd IAT Min", "Bwd Pkt Len Min"], inplace=True)

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

    # Add additional data
    augmented_data_files = ["HULK.csv", "GoldenEye.csv", "Slowloris.csv", "GoldenEye2.csv", "Slowloris2.csv",
                            "GoldenEye3.csv", "GoldenEye4.csv"]
    for augmented_file in augmented_data_files:
        print("Adding augmented data file: " + augmented_file)

        data = pd.read_csv("data/augment/" + augmented_file)
        big_data_df = pd.concat([big_data_df, data], ignore_index=True)

    # Apply normalization techniques
    print("Cleaning the data...")

    temp_labels = big_data_df["Label"]
    big_data_df = big_data_df.drop("Label", axis=1)

    # Record max used to normalize the columns. These will be applied to any data we want to test with after training
    big_data_df.abs().max().to_frame().T.to_csv("test/normalization/max.csv", index=False)

    big_data_df = big_data_df.join(temp_labels)

    # Divide by max
    for column in big_data_df.columns:
        # Check for string types in columns (can't take standard dev. of strings)
        if type(big_data_df[column][0]) is str:
            continue

        big_data_df[column] = big_data_df[column] / big_data_df[column].abs().max()

    temp_labels = big_data_df["Label"]
    big_data_df = big_data_df.drop("Label", axis=1)

    # Record mean and std used to normalize the columns. These will be applied to any data we want to test with
    big_data_df.mean().to_frame().T.to_csv("test/normalization/mean.csv", index=False)
    big_data_df.std().to_frame().T.to_csv("test/normalization/std.csv", index=False)

    big_data_df = big_data_df.join(temp_labels)

    # Convert to z-scores
    for column in big_data_df.columns:
        # Check for string types in columns (can't take standard dev. of strings)
        if type(big_data_df[column][0]) is str:
            continue

        mean = big_data_df[column].mean()
        std = big_data_df[column].std()
        big_data_df[column] = (big_data_df[column] - mean) / std

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
