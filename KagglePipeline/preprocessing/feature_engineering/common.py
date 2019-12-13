def groupby_statistics(data, group, col, strategy="mean"):
    """
    Returns a function which accepts data and creates a new column for the data (data[group+col+strategy] which is data.groupby(group)[col].transform(strategy)
    Can be used in the funclist of the engineer function
    """
    def func(data):
        data[group+col+strategy] = data.groupby(group)[col].transform(strategy)
    return func

def groupby_statistics_subtraction(data, group, col, strategy="mean"):
    """
    Returns a function which accepts data and creates a new column for the data (data[group+col+strategy]) which is data[col] - data.groupby(group)[col].transform(strategy)
    Can be used in the funclist of the engineer function
    """
    def func(data):
        data[group+col+strategy+"subtraction"] = data[col] - data.groupby(group)[col].transform(strategy)
    return func

def split_by_list(data, col, l):
    """
    Creates a new column in data (col_split) which is the band at which the data falls for given column
    l should be a list of values to split by e.g l = [0,1,2] all values less than 0 get assigned level0, all between 0 and 1 level1 etc.
    l must be sorted in strictly increasing order

    Is meant to work on numerical columns only 

    Returns a function that can be used in funclist of engineer
    """
    def func(data):
        def helper(x):
            for i, val in enumerate(l):
                if (x < val):
                    return f"Level {i}"
            return f"Level {len(l)}"
        data[col+"_split"] = data[col].transform(helper)

