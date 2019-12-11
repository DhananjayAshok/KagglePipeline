class LinearModels(object):
    """
    A MultipleModels object designed to work on table data. 
    This set by default has the following tree models
        1. LinearRegression
        2. Lasso Regression
        3. Ridge Regression
    
    """
    def __init__(self, classification, extras=[]):
        MultipleModels.__init__(self, classification, extras)
        return

    def _generate_model_list(self, classification):
        import xgboost
        rf, et, ab, gb = None, None, None, None
        if(classification):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.ensemble import ExtraTreesClassifier
            from sklearn.ensemble import AdaBoostClassifier
            rf = RandomForestClassifier()
            et = ExtraTreesClassifier()
            ab = AdaBoostClassifier()
            gb = xgboost.XGBClassifier()
        else:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.ensemble import AdaBoostRegressor
            rf = RandomForestRegressor()
            et = ExtraTreesRegressor()
            ab = AdaBoostRegressor()
            gb = xgboost.XGBRegressor()
            
        self.model_list = [(rf, "RandomForest"),(et,"ExtraTrees"),(ab,"AdaBoost"),(gb, "Gradient Boosting")]
        return


