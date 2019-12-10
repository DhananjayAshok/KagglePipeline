import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def correlation_heatmap(data):
    corrmat = data.corr()
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.title("Correlation Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.show()
    return


def scatter(data):

    pd.plotting.scatter_matrix(data, alpha=0.2)
    plt.show()

