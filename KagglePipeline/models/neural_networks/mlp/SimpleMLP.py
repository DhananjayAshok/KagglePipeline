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
   

    def __init__(self, classification, n_hidden_neurons=32 ,optimizer=None, initial_learning_rate=0.001, loss=None):
        """
        Returns a fit model with Input -> 32 Hidden Neurons -> Output for classification. Unless parameters are changed.
        Default Learning Rate Scheduling is an Adam optimizer with Exponential Decay off an initial learning rate of 0.001
        """
        params = {"n_hidden_neurons":n_hidden_neurons}
        NeuralNetwork.__init__(self, classification, params, optimizer, initial_learning_rate, loss)
        return

    def _create_internal_layers(self, i, X_train_shape, params):
        """
        Hidden shape is in format [X_train.shape, hidden_shape]
        """

        x = Dense(X_train_shape[1], activation ="elu") (i)
        x = Dense(params["n_hidden_neurons"], activation= "relu") (x) 
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
    def __init__(self, n_hidden_neurons=32, optimizer=None, initial_learning_rate=0.001, loss=None):
        SimpleMLP.__init__(self, False, n_hidden_neurons, optimizer, initial_learning_rate, loss)

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
    def __init__(self, n_hidden_neurons=32, optimizer=None, initial_learning_rate=0.001, loss=None):
        SimpleMLP.__init__(self, True, n_hidden_neurons, optimizer, initial_learning_rate, loss)


