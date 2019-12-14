def investigate_nan_columns(data, verbose=False)->None:
    """
    Prints an analysis of the nans in the dataframe
    Verbose is to actually show the database
    """
    col_dict = {}
    na_df = data.isna()
    total_size = na_df.shape[0]
    for col in na_df:
        a = na_df[col].value_counts()
        if False not in a.keys():
            col_dict[col] = 1.0
        elif True not in a.keys():
            pass
        else:
            col_dict[col] =  a[True]/total_size
    print(f"{col_dict}")
    if verbose:
        for col in col_dict:
            print(f"For {col}\n {data[data[col].isnull()]}")
    return


