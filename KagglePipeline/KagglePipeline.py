if __name__ == "__main__":
    from models.neural_networks.mlp.SimpleMLP import SimpleMLPRegressor
    from rapid_deployment.table.TreeModels import TreeModels
    from tensorflow.keras.datasets.boston_housing import load_data


    train, test = load_data()
    x_train = train[0]
    y_train = train[1]
    x_test = test[0]
    y_test = test[1]
    model = SimpleMLPRegressor()
    model.fit(x_train, y_train, epochs=1000,x_val = x_test, y_val=y_test)
    model.compare_pred_to_truth(x_test[0:5], y_test[0:5])
    models = TreeModels(x_train, y_train, False, extras=[(model,"NeuralNet")])
    models.display_comparison(x_test, y_test)
    models.display_cross_val_comparison(x_train, y_train, cross_val=3)


