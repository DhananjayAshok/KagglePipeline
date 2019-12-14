import matplotlib.pyplot as plt
import seaborn as sns

def distplot(data, column,by=None,by_val="", series=False,):
    """
    Reveals a positive skew
    If you want to use this on a series instead of a column make series = True and put any value for column it doesn't matter
    by option is the categorical you want to guage the numericals distributions by: will produce a map for each label in the category
    """
    if (len(data[column]) <= 1):
        print(f"Can not show displot for {column} {by_val} has less than 2 data points")
        return
    from scipy.stats import norm
    if by is None:
        if (series):
            sns.distplot(data, fit=norm)
            plt.title(f"Distribution and Skew of target|Kurtosis {data.kurtosis()}")
        else:
            sns.distplot(data[column], fit=norm)
            plt.title(f"Distribution and Skew of {column}|Kurtosis {data[column].kurtosis()}")
        plt.xlabel(column+" "+by_val)
        plt.ylabel("Value")
        plt.show()
    else:
        for label in set(data[by]):
            a = data[data[by] == label]
            distplot(a, column, by=None, by_val=str(label))
    return


def categorical_numerical_relationship(data, categorical, numerical):
    sns.barplot(x=categorical, y = numerical, data=data)
    plt.ylabel(str(numerical))
    plt.xlabel(categorical)
    plt.show()
    return


