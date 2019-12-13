from ...exploration.table.Utility import guess_types


def encode_categoricals(data, categoricals = None):
    """
    Encode the categories using LabelEncoder and then return in the format (data, label_dict)
    data is the new dataframe after encoding,
    label_dict stores each columns label encoder
    """
    from sklearn.preprocessing import LabelEncoder
    if categoricals is None:
        n, s, categoricals,  d = guess_types(data)
    d = {}
    for cat in categoricals:
        print(f"Now encoding {cat}")
        le = LabelEncoder()
        data[cat] = le.fit_transform(data[cat])
        d[cat] = le
    return data, d

def encode_dataframe_with_dict(data, d):
    """
    Can be used to encode the values of testing data after already encoding training data
    Is an inplace function
    """
    import numpy as np
    for cat in d.keys():
        if cat not in data.columns:
            continue
        try:
            data[cat] = d[cat].transform(data[cat])
        except:
            print(f"There was an error with column {cat}. Likely new label encountered. Classes {d[cat].classes_} and unique labels are {set(data[cat])}")
            print("You will have to manage this manually. Either encode it with your own method or clear the new labels and run this function again")
    return

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



