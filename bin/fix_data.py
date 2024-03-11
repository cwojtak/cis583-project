# RUN THIS ONLY ONCE!!!

import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("../data/02-16-2018.csv")
    df.drop(999999)
    df.to_csv("../data/02-16-2018.csv", index=False)
