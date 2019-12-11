class MultipleModels(object):
    """
    Abstract Classes that create multiple models and store them in a list

    
    """
    def __init__(self, classification, extras=[]):
        """
        Extras is a list of additional models that can be called to fit on all in format (model, string name of model)
        """
        self.model_list = []
        self._generate_model_list(classification)
        self.model_list.extend(extras)
        self.classification = classification


    def fit(self, X_train, y_train):
        for model_tuple in self.model_list:
            model_tuple[0].fit(X_train, y_train)


    def _generate_model_list(self):
        raise NotImplementedError

    def display_comparison(self, X_val, y_val):
        """
        Displays a chart of each model and their performance on the datasets provideds
        """
        import matplotlib.pyplot as plt
        x = []
        y = []
        for model_tuple in self.model_list:
            x.append(model_tuple[1])
            y.append(model_tuple[0].score(X_val, y_val))
        plt.scatter(x, y)
        plt.show()


    def display_cross_val_comparison(self, X, y, cross_val= 5):
        """
        More rigerous cross validation is used to train the models and compare scores
        Plots the results
        returns a dataframe with the results
        """
        from sklearn.model_selection import KFold, cross_val_score
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        folds = KFold(n_splits=cross_val, shuffle=True, random_state=11)
        d = pd.DataFrame({self.model_list[0][1]: list(range(cross_val))})
        for model_tuple in self.model_list:
            d[model_tuple[1]] = cross_val_score(model_tuple[0], X, y, cv=folds)

        sns.boxplot(data=d)
        plt.xlabel("Classifier")
        plt.ylabel("R^2 Score (Higher is Better)")
        plt.title("Comparison between models")
        plt.show()
        return d


    def __len__(self):
        return len(self.model_list)

    def __str__(self):
        return str([model_tuple[1] for model_tuple in self.model_list])

    def parameters(self):
        return str([model_tuple[0] for model_tuple in self.model_list])
