if __name__ == "__main__":
    from models.neural_networks.mlp.SimpleHiddenMLP import SimpleHiddenMLP
    from tensorflow.keras.datasets.boston_housing import load_data


    train, test = load_data()
    x_train = train[0]
    y_train = train[1]
    x_test = test[0]
    y_test = test[1]
    model = SimpleHiddenMLP(x_train, y_train, classification=False)
    model.fit(x_train, y_train, epochs=1000,x_val = x_test, y_val=y_test)
    model.compare_pred_to_truth(x_test[0:5], y_test[0:5])

