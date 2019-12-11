def extract_target(train, test, label_col):
    """
    Takes in two dataframes, train and test which have both X_train and y_train (for train) in them and returns 4 items in order
    X_train, X_test, y_train, y_test
    """
    X_train = train.drop(label_col, axis=1)
    y_train = train[label_col]
    X_test = test.drop(label_col, axis=1)
    y_test = test[label_col]
    return X_train, X_test, y_train, y_test