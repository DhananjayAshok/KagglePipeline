from ...exploration.table.Utility import guess_types

def _make_strategy_lists(data, mean_cols, median_cols, mode_cols, n_cols, skip_cols):
    numericals, strings, categoricals, dates = guess_types(data)
    l = [numericals, strings, categoricals, dates]
    for t in l:
        for el in t:
            if (el in mean_cols) or (el in mode_cols) or (el in median_cols) or(el in n_cols) or (el in skip_cols):
                pass
            else:
                if (t == numericals):
                    mean_cols = mean_cols + [el]
                elif (t == strings):
                    n_cols = n_cols + [el]
                elif (t==categoricals):
                    mode_cols = mode_cols + [el]
                else:
                    mode_cols = model_cols + [el]
    l = [mean_cols, median_cols, mode_cols, n_cols]
    return l

def handle_nan_entries(data, mean_cols=[],median_cols=[], mode_cols=[], n_cols=[], skip_cols=[]):
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
    l = _make_strategy_lists(data, mean_cols, median_cols, mode_cols, n_cols, skip_cols)
    for i, strategy in enumerate(l):
        for col in strategy:
            val = None
            if(i==0):
                val = data[col].mean()
            elif(i==1):
                val = data[col].median()
            elif(i==2):
                val = data[col].mode()
            elif(i==3):
                val = "N"
            if (i==4):
                continue
            else:
                data[col].fillna(val, inplace=True)
    return
    
def handle_test_nan_entries(train, test, mean_cols=[], median_cols=[], mode_cols=[], n_cols=[], skip_cols=[]):
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
    l = _make_strategy_lists(train, mean_cols, median_cols, mode_cols, n_cols, skip_cols)
    for i,strategy in enumerate(l):
        for col in strategy:
            val = None
            if(i == 0):
                val = train[col].mean()
            elif(i==1):
                val = train[col].median()
            elif(i==2):
                val = train[col].mode()
            elif(i==3):
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

    