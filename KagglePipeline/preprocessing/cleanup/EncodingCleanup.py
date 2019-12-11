
def encode_categoricals(data, categoricals = None):
    """
    Encode the categories using LabelEncoder and then return in the format (data, label_dict)
    data is the new dataframe after encoding,
    label_dict stores each columns classes_ variable and is needed to reverse the encoding
    """
    from sklearn.preprocessing import LabelEncoder
    if categoricals is None:
        n, s, categoricals,  d = guess_types(data)
    d = {}
    for cat in categoricals:
        print(f"Now encoding {cat}")
        le = LabelEncoder()
        data[cat] = le.fit_transform(data[cat])
        d[cat] = le.classes_
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
        def helper(x):
            val = 0
            try:
                val = np.where(d[cat] == x)[0][0]
            except ValueError:
                raise ValueError(f"{x} was not found in encoded labels {d[cat]}. i.e new label encountered")
            return val
        data[cat] = data[cat].apply(helper)
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
        def helper(x):
            return d[cat][x]
        data[cat] = data[cat].apply(helper)
    return

def decode_single_value(value, col, d):
    """
    Decodes the value of a single object from a single column in the data
    Returns the value
    """
    return d[col][val]



