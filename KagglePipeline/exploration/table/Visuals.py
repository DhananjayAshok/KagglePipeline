import matplotlib.pyplot as plt
import seaborn as sns

def distplot(data, column):
    """
    Reveals a positive skew
    """
    from scipy.stats import norm
    sns.distplot(data[column], fit=norm)
    plt.title(f"Distribution and Skew of Interest Rate|Kurtosis {data[column].kurtosis()}")
    plt.xlabel(column)
    plt.ylabel("Value")
    plt.show()
    return


