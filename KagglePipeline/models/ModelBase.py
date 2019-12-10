import pandas as pd

class ModelBase(object):
    """
    Base abstract class for all models with common functions and variables

    """
    def __init__(self):
        raise NotImplementedError
        

    def _create_model(self, X_train, y_train):
        raise NotImplementedError

    def _create_param_dict():
        raise NotImplementedError
    
    def fit(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def get_params(self, deep=True):
        return self.param_dict

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.params[parameter] = value
        return self


    def compare_pred_to_truth(self, X_test, y_test):
        """
        Prints a dataframe report comparing the predicted values to the actual values.
        Can be used when y_test contains only single values i.e labels or regression values.
        """
        preds = self.predict(X_test)
        d = {'Predictions': list(preds), 'Truth': list(y_test)}
        data = pd.DataFrame(d)
        print(data)
        return


