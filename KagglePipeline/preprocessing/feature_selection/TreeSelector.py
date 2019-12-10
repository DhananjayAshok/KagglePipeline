def randomforest_feature_importance(X_train, y_train, classification):
    """
    Trains a RandomForestRegressor on the data and analyzes which features were weighed as more important
    Returns an eli5 weights option
    """
    import eli5
    from sklearn.model_selection import train_test_split
    train_X, val_X, train_y, val_y = train_test_split(X_train, y_train, random_state=1)

    clf = None
    if (classification):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier
    else:
        from sklearn.ensemble import RandomForestRegressor
        clf = RandomForestRegressor

    
    rfc_model = clf(random_state=0).fit(train_X, train_y)
    
    from eli5.sklearn import PermutationImportance
    perm = PermutationImportance(rfc_model, random_state=1).fit(val_X, val_y)
    return eli5.show_weights(perm, feature_names = val_X.columns.tolist(), top=7)


def gradientboosted_feature_importance(X_train, y_train, classification):
    """
    Fits a Gradient Boosted LGBM Tree to the data and prints an analysis of feature importance
    Returns a pandas DataFrame of feature importance
    """
    import numpy as np
    import lightgbm as lgb
    import pandas as pd

    # Initialize an empty array to hold feature importances
    feature_importances = np.zeros(X_train.shape[1])

    # Create the model with several hyperparameters
    model = None
    if (classification):
        model = lgb.LGBMClassifier(boosting_type = 'goss', n_estimators = 10000, class_weight = 'balanced')
    else:
        model = lgb.LGBMRegressor(boosting_type = 'goss', n_estimators = 10000, class_weight = 'balanced')

    # Fit the model twice to avoid overfitting
    from sklearn.model_selection import train_test_split
    for i in range(2):
    
        # Split into training and validation set
        train_features, valid_features, train_y, valid_y = train_test_split(X_train, y_train, test_size = 0.25, random_state = i)
    
        # Train using early stopping
        model.fit(train_features, train_y, early_stopping_rounds=100, eval_set = [(valid_features, valid_y)], 
                  eval_metric = 'auc', verbose = 200)
    
        # Record the feature importances
        feature_importances += model.feature_importances_
    feature_importances = feature_importances / 2
    feature_importances = pd.DataFrame({'feature': list(X_train.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)
    print(feature_importances.head())
    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
    print('There are %d features with 0.0 importance' % len(zero_features))
    print(feature_importances.tail())
    return feature_importances
