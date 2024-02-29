import os

import dataset
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

        # rename file to ***cleaned.csv
        file_name_cleaned_csv = file_name.replace(".csv", "_cleaned.csv")
        df_to_normalize.to_csv(os.path.join(path, file_name_cleaned_csv))
        print(file_name + " has been cleaned")


if __name__ == '__main__':
    main()