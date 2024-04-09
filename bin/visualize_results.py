import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab
import sys

# Basic visualization script that generates charts for loss and accuracy
# Call this script with "python3 ./bin/visualize_results.py ./raw_results/result_file.csv to get visualizations of a run

if __name__ == '__main__':
    # Read in CSV
    df = pd.read_csv(sys.argv[1])

    # Plot accuracy over epochs
    figure, axis = plt.subplots()
    axis.plot(df.index, df["Accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy over Epochs")
    plt.show()

    # Plot average loss over epochs
    figure2, axis2 = plt.subplots()
    axis2.plot(df.index, df["Average Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.title("Model Average Loss over Epochs")
    plt.show()
