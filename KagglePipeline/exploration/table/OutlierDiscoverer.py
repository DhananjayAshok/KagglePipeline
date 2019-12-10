from .Utility import guess_types

def outlier_count(data, numericals=None, threshold_sigma=2):
    """
    Prints a dictionary which gives a report on outliers per numerical column in format (no:outliers, % outliers)
    Outliers are defined as the values that fall outside the range of |X-mean|<= threshold*stdev

    If no numerical list is provided a guess is made
    """
    if(numericals is None):
        numericals, s, c, d = guess_types(data)
    d = {}
    for col in data[numericals]:
        m = data[col].mean()
        s = data[col].std()
        c = 0
        for val in data[col]:
            if abs(m-val) > threshold_sigma*s:
                c+=1
        if(c != 0):
            d[col] =  (c, c/len(data[col]))
    return d





