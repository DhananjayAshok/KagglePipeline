from ...exploration.table.Utility import guess_types

def handle_nan_entries(data, mean_cols=[], mode_cols=[], n_cols=[], skip_cols=[]):
    """
    Generic handling strategy for Nan entries in columns with 3 strategies
        fill with mean, 
        fill with mode
        fill with "N"
    By default if no columns are provided in any of the options the following is assumed
        All numericals are filled with mean
        All categoricals and dates are filled with Mode
        All strings are filled with N

    skip_cols is for columns you want to manually handle with a different strategy and so should not be messed with in this run
    returns None. Is an inplace function
    """
    numericals, strings, categoricals, dates = guess_types(data)
    l = [numericals, strings, categoricals, dates]
    for t in l:
        for el in t:
            if (el in mean_cols) or (el in mode_cols) or (el in n_cols) or (el in skip_cols):
                pass
            else:
                if (t == numericals):
                    mean_cols.append(el)
                elif (t == strings):
                    n_cols.append(el)
                elif (t==categoricals):
                    mode_cols.append(el)
                else:
                    mode_cols.append(el)

    l = [mean_cols, mode_cols, n_cols]
    for strategy in l:
        for col in strategy:
            val = None
            if(strategy == mean_cols):
                val = data[col].mean()
            elif(strategy == mode_cols):
                val = data[col].mode()
            elif(strategy == n_cols):
                val = "N"

            data[col].fillna(val, inplace=True)
    return
    
def handle_test_nan_entries(train, test, mean_cols=[], mode_cols=[], n_cols=[], skip_cols=[]):
    """
    Same as handle nan entries but meant for testing data, this will use the mode/mean of the training data to fill testing data to ensure integrity of statistics
    Do this step before you encode the training data in order to ensure that the guessing of numericals, categoricals, strings and dates works

    Generic handling strategy for Nan entries in columns with 3 strategies
        fill with mean, 
        fill with mode
        fill with "N"
    By default if no columns are provided in any of the options the following is assumed
        All numericals are filled with mean
        All categoricals and dates are filled with Mode
        All strings are filled with N

    skip_cols is for columns you want to manually handle with a different strategy and so should not be messed with in this run
    returns None. Is an inplace function
    """
    numericals, strings, categoricals, dates = guess_types(train)
    l = [numericals, strings, categoricals, dates]
    for t in l:
        for el in t:
            if (el in mean_cols) or (el in mode_cols) or (el in n_cols) or (el in skip_cols):
                pass
            else:
                if (t == numericals):
                    mean_cols.append(el)
                elif (t == strings):
                    n_cols.append(el)
                elif (t==categoricals):
                    mode_cols.append(el)
                else:
                    mode_cols.append(el)

    l = [mean_cols, mode_cols, n_cols]
    for strategy in l:
        for col in strategy:
            val = None
            if(strategy == mean_cols):
                val = train[col].mean()
            elif(strategy == mode_cols):
                val = train[col].mode()
            elif(strategy == n_cols):
                val = "N"

            test[col].fillna(val, inplace=True)
    return


def categorical_fillna(data, col, value="N"):
    """
    Fills a categorical features Nan Values with the provided value, returns a new column. 
    The pandas function sometimes does not work
    """
    def helper(x):
        if isinstance(x, float):
            return value
        return x
    return data[col].apply(helper)


def drop_nan_columns(data, ratio=1.0):
    """
    The ratio parameter (0.0<=ratio<1.0) lets you drop columns which has 'ratio'% of nans. (i.e if ratio is 0.8 then all columns with 80% or more entries being nan get dropped)
    Returns a new dataframe
    """
    col_list = []
    na_df = data.isna()
    total_size = na_df.shape[0]
    for col in na_df:
        a = na_df[col].value_counts()
        if False not in a.keys():
            col_list.append(col)
        elif True not in a.keys():
            pass
        else:
            if a[True]/total_size >= ratio:
                col_list.append(col)
    print(f"{len(col_list)} columns dropped- {col_list}")
    return data.drop(col_list, axis=1)

    