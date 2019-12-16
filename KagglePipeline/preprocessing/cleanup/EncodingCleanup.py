from ...exploration.table.Utility import guess_types


def encode_categoricals(data, ordinals = None, nominals=None, all_categories = True):
    """
    Encode the categories using Encoders and then return in the format (data, label_dict)
    data is the new dataframe after encoding,
    label_dict stores each columns encoder
    Ordinals is a list of all colmnns that should be encoded with LabelEncoding to preserve Order
    Nominals is a list of all columns that should be pd.get_dummies
    if all_categories is true then any categories not in the ordinals or nominals will be encoded with pd.get_dummies
    """
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    n, s, categoricals,  dates = guess_types(data)
    d = {"dummy_data": []}
    l = [ordinals, nominals]
    if ordinals is not None:
        for i in ordinals:
            categoricals.remove(i)
            print(f"Now encoding {i}")
            encoder = LabelEncoder()
            data[i] = encoder.fit_transform(data[i])
            d[i] = encoder

    if nominals is not None:
        data = pd.get_dummies(data, columns=nominals)
        d['dummy_data'].extend(nominals)

    if (all_categories):
        n, s, categoricals, dates = guess_types(data)
        data = pd.get_dummies(data, columns = categoricals)
        d['dummy_data'].extend(categoricals)
    return data, d

def encode_dataframe_with_dict(data, train, d):
    """
    Can be used to encode the values of testing data after already encoding training data
    Returns a new dataframe
    """
    import numpy as np
    import pandas as pd
    for cat in d.keys():
        if cat not in data.columns or cat == "dummy_data":
            continue
        try:
            data[cat] = d[cat].transform(data[cat])
        except:
            working = set(data[cat])
            print(f"There was an error with column {cat}. Likely new label encountered. Classes {d[cat].classes_} and new labels encountered {working.difference(set(d[cat].classes_))}")
            print(f"You will have to manage this manually if you want to handle a special case. For now the function will encode all labels that do not fit the given classes to {len(d[cat].classes_)}")
            convoluted_string = "uNiQueOtHeRVaL"
            d[cat].classes_ = np.append(d[cat].classes_, convoluted_string) # The string is so convoluted because then no conflicts will occur with any actual values for sure. 
            def helper(x):
                if x not in d[cat].classes_:
                    return convoluted_string
                else:
                    return x
            data[cat] = data[cat].transform(helper)
            data[cat] = d[cat].transform(data[cat])
    if d.get("dummy_data", []) != []:
        data = pd.get_dummies(data, columns = d["dummy_data"])
        train_cols = set(train.columns)
        test_cols = set(data.columns)
        train_not_test = train_cols.difference(test_cols)
        if train_not_test != set():
            # Not everything in train_cols is there in test_cols
            print(f"There were columns that were in training but not in testing data. i.e labels not in testing - {train_not_test}")
            print("Adding 0 columns to the testing data of that name")
            for item in train_not_test:
                data[item] = np.zeros((data.shape[0],)).astype(int)
        test_not_train = test_cols.difference(train_cols)
        if test_not_train != set():
            print(f"There were columns in testing data not training data. i.e labels not in training - {test_not_train}")
            print("Dropping these")
            data = data.drop(list(test_not_train), axis=1)
        # Reorder
        train_cols = train.columns
        data = data[train_cols]
    return data


def decode_dataframe_categoricals(data, d):
    """
    Decode all the label encoding that was once done
    Is an inplace function
    d is the label_dictionary
    """
    for cat in d.keys():
        if cat not in data.columns:
            continue
        data[cat] = d[cat].inverse_transform(data[cat])
    return

def decode_single_value(value, col, d):
    """
    Decodes the value of a single object from a single column in the data
    Returns the value
    """
    return d[col].inverse_transform([val])



