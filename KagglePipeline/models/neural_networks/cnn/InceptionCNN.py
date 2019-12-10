from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from ..NeuralNetwork import NeuralNetwork

class InceptionCNN(NeuralNetwork):
    """
    CNN with a simple archiecture of Input -> Inception Module -> Conv2D(16 filters of size 3) -> MaxPool -> Dense 32 -> Output

    """
    def __init__(self, classification, conv2d_fmaps=16, final_dense=32 ,optimizer=None, initial_learning_rate=0.001, loss=None):
        """
        """
        params = {"conv2d_filters":conv2d_filters, "final_dense":final_dense}

        NeuralNetwork.__init__(self, classification, params, optimizer, initial_learning_rate, loss)
        return

    def _create_internal_layers(self, i, X_train_shape, params):
        conv1_fmaps = params['conv2d_filters']
        conv1_ksize = 3
        conv1_pad = "SAME"
        pool_fmaps = conv1_fmaps
        n_fc1 = params['final_dense']

        inception = self._inception_module(i)

        conv1 = Conv2D(conv1_fmaps, conv1_ksize, padding="SAME" , activation='elu')(inception)

        pool = MaxPooling2D()(conv1)

        pool_flat = Flatten()(pool)

        dropped = Dropout(0.5) (pool_flat)

        fc = Dense(n_fc1, activation='elu')(dropped)
       
        fc_dropped = Dropout(0.5)(fc)
        return fc_dropped

    def _inception_module(self, i):
        tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(i)
        tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

        tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(i)
        tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)

        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(i)
        tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)

        output = concatenate([tower_1, tower_2, tower_3], axis=1)

        return output

class InceptionCNNRegressor(InceptionCNN):
    def __init__(self, classification, conv2d_fmaps=16, final_dense=32 ,optimizer=None, initial_learning_rate=0.001, loss=None):
        InceptionCNN.__init__(self, False, conv2d_fmaps=16, final_dense=32 ,optimizer=None, initial_learning_rate=0.001, loss=None)

class InceptionCNNClassifier(InceptionCNN):
        def __init__(self, classification, conv2d_fmaps=16, final_dense=32 ,optimizer=None, initial_learning_rate=0.001, loss=None):
        InceptionCNN.__init__(self, True, conv2d_fmaps=16, final_dense=32 ,optimizer=None, initial_learning_rate=0.001, loss=None)

