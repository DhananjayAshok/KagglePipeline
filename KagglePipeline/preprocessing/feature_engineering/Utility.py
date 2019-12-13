
def engineer(train, test, real_test=None, funclist=[]):
    """
    Loop through all of the dataframes and apply each of the funclist functions on them, in order
    funclist must be a list of functions that accept data as a single parameter.
    """
    datasets = [train, test]
    if real_test is not None:
        datasets.append(real_test)
    for data in datasets:
        for func in funclist:
            func(data)
    return


def refresh(train, test, real_test=None):
    """
    Returns deep copies of all of the provided DataFrames for easily refreshing or undoing feature engineering
    """
    if real_test is not None:
        return train.copy(deep = True), test.copy(deep = True), real_test.copy(deep = True)
    else:
        return train.copy(deep = True), test.copy(deep = True)
