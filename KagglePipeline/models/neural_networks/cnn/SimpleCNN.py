from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from ..NeuralNetwork import NeuralNetwork

class SimpleCNN(NeuralNetwork):
    """
    CNN with a simple archiecture of Input -> Conv2D(16 filters of size 3) -> MaxPool -> Dense 32 -> Output

    """
    def __init__(self, classification, conv2d_filters=16, final_dense=32 ,optimizer=None, initial_learning_rate=0.001, loss=None):
        """
        Is expecting hidden_shape of format [(Conv2D fmaps), (Dense Layer Nodes)]
        """
        params = {"conv2d_filters":conv2d_filters, "final_dense":final_dense}
        NeuralNetwork.__init__(self, classification, params, optimizer, initial_learning_rate, loss)
        return

    def _create_internal_layers(self, i,X_train_shape, params):
        conv1_fmaps = params['conv2d_filters']
        conv1_ksize = 3
        conv1_pad = "SAME"
        pool_fmaps = conv1_fmaps
        n_fc1 = params['final_dense']

        x = Conv2D(conv1_fmaps, conv1_ksize, padding=conv1_pad, activation='elu') (i)
        x = Flatten() (x)
        x = Dense(n_fc1, activation="relu") (x)
        return x



        
class SimpleCNNRegressor(SimpleCNN):
    def __init__(self, classification, conv2d_fmaps=16, final_dense=32 ,optimizer=None, initial_learning_rate=0.001, loss=None):
        SimpleCNN.__init__(self, False, conv2d_fmaps=16, final_dense=32 ,optimizer=None, initial_learning_rate=0.001, loss=None)

class SimpleCNNClassifier(SimpleCNN):
    def __init__(self, classification, conv2d_fmaps=16, final_dense=32 ,optimizer=None, initial_learning_rate=0.001, loss=None):
        SimpleCNN.__init__(self, True, conv2d_fmaps=16, final_dense=32 ,optimizer=None, initial_learning_rate=0.001, loss=None)
