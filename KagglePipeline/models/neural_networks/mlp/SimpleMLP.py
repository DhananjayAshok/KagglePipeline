from tensorflow.keras.layers import Dense
from ..NeuralNetwork import NeuralNetwork

class SimpleMLP(NeuralNetwork):
    """A Densely Connected Feed Forward MLP which can be used for most classification and regression tasks

    Requirements:
        X is a 2 dimensional DataFrame i.e is in table data format
        y is either a single column of values if regression or a set of labels if classification
    
    Attributes
    -------------
    model : tf.model
        Stores the model reference
    classification: bool
        True iff the network solves a classification problem
        If false then it must be regression

    Methods
    -------------
    predict(X: pd.DataFrame, y: pd.Series) -> np.ndarray
        Predict the value from given data
    

    """
   

    def __init__(self, classification, hidden_shape = [32] ,optimizer=None, initial_learning_rate=0.001, loss=None):
        """
        Takes in hidden_shape argument: a list of format [layer0_n_neurons, layer1_n_neurons,......] with elu activation for all of them.
        Returns a fit model with Input -> Hidden_Shape -> Output
        Default Learning Rate Scheduling is an Adam optimizer with Exponential Decay off an initial learning rate of 0.001
        """
        params = {"hidden_shape":hidden_shape}
        NeuralNetwork.__init__(self, classification, params, optimizer, initial_learning_rate, loss)
        return

    def _create_internal_layers(self, i, X_train_shape, params):
        """
        Hidden shape is in format [X_train.shape, hidden_shape]
        """
        h_shape = params.get("hidden_shape", [32])
        x = Dense(X_train_shape[1], activation ="elu") (i)
        for layer in h_shape:
            x = Dense(layer, activation= "relu") (x) 
        return x



class SimpleMLPRegressor(SimpleMLP):
    """A Densely Connected Feed Forward MLP which can be used for most regression tasks

    Requirements:
        X is a 2 dimensional DataFrame i.e is in table data format
        y is either a single column of values if regression or a set of labels if classification
    
    Attributes
    -------------
    model : tf.model
        Stores the model reference
    
    Methods
    -------------
    predict(X: pd.DataFrame, y: pd.Series) -> np.ndarray
        Predict the value from given data
    

    """
    def __init__(self, hidden_shape=[32], optimizer=None, initial_learning_rate=0.001, loss=None):
        SimpleMLP.__init__(self, False, hidden_shape=hidden_shape, optimizer=optimizer, initial_learning_rate=initial_learning_rate, loss=loss)

class SimpleMLPClassifier(SimpleMLP):
    """A Densely Connected Feed Forward MLP which can be used for most classification tasks

    Requirements:
        X is a 2 dimensional DataFrame i.e is in table data format
        y is either a single column of values if regression or a set of labels if classification
    
    Attributes
    -------------
    model : tf.model
        Stores the model reference

    Methods
    -------------
    predict(X: pd.DataFrame, y: pd.Series) -> np.ndarray
        Predict the value from given data
    

    """
    def __init__(self, hidden_shape=[32], optimizer=None, initial_learning_rate=0.001, loss=None):
        SimpleMLP.__init__(self, True, hidden_shape=hidden_shape, optimizer=optimizer, initial_learning_rate=initial_learning_rate, loss=loss)


