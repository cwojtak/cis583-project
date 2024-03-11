import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab
import sys

if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])

    figure, axis = plt.subplots()
    axis.plot(df.index, df["Accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy over Epochs")
    plt.show()

    figure2, axis2 = plt.subplots()
    axis2.plot(df.index, df["Average Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model Average Loss over Epochs")
    plt.show()
