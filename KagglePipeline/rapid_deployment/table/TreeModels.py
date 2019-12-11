from ..MultipleModels import MultipleModels
class TreeModels(MultipleModels):
    """
    A MultipleModels object designed to work on table data. 
    This set by default has the following tree models
        1. RandomForest
        2. ExtraTrees
        3. AdaBoost
        4. Gradient Boosting Decision Tree
        5. 
    """
    def __init__(self, classification, extras=[]):
        """

       Extras is a list of additional models that can be called to fit on all in format (model, string name of model)
       

        """
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



