def drop_nan_columns(data, ratio=1.0)->pd.DataFrame:
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

def investigate_nan_columns(data)->None:
        """
        Prints an analysis of the nans in the dataframe
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
        return

