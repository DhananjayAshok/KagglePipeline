import matplotlib.pyplot as plt
import seaborn as sns

def distplot(data, column, series=False):
    """
    Reveals a positive skew
    If you want to use this on a series instead of a column make series = True and put any value for column it doesn't matter
    """
    from scipy.stats import norm
    if (series):
        sns.distplot(data, fit=norm)
        plt.title(f"Distribution and Skew of target|Kurtosis {data.kurtosis()}")
    else:
        sns.distplot(data[column], fit=norm)
        plt.title(f"Distribution and Skew of {column}|Kurtosis {data[column].kurtosis()}")
    plt.xlabel(column)
    plt.ylabel("Value")
    plt.show()
    return


