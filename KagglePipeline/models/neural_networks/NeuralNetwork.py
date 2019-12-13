from ..ModelBase import ModelBase
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork(ModelBase):
    """Abstract MLP Implementation with fitting and predicting implemented
    """
    def __init__(self, classification, params={}, optimizer=None, initial_learning_rate = 0.001, loss=None):
        """
        Returns a Feed Forward fit model with Input -> Hidden Neurons -> Output. Unless parameters are changed.
        Default Learning Rate Scheduling is an Adam optimizer with Exponential Decay off an initial learning rate of 0.001
        """
        self.classification = classification
        self._create_param_dict(params, optimizer, initial_learning_rate, loss)
        return

    def _create_model(self, X_train, y_train):
        classification = self.classification
        optimizer = self.param_dict['optimizer']
        initial_learning_rate = self.param_dict['initial_learning_rate']
        loss = self.param_dict['loss']


        i = Input(shape=X_train.shape[1:])

        x = self._create_internal_layers(i, X_train.shape, self.param_dict)
        
        l = None
        metrics = []
        if(classification):
            n_outputs = self._get_num_categories(y_train, in_ohe=True)
            x = Dense(n_outputs, activation = "softmax") (x)

            l = CategoricalCrossentropy()
            metrics.append('accuracy')

        else:
            n_outputs = 1
            x = Dense(n_outputs, activation = "relu") (x)
            l = MeanSquaredError()

        lr_schedule = ExponentialDecay(initial_learning_rate,decay_steps=100000,decay_rate=0.96, staircase=True)
        opt = Adam(learning_rate = lr_schedule)

        if(loss is not None):
            l = loss
        if (optimizer is not None):
            opt = optimizer

        self.model = Model(i, x)

        self.model.compile(optimizer=opt,
              loss=l,
              metrics=metrics)
        return 

    def _create_internal_layers(self, i, X_train_shape, params):
        raise NotImplementedError

    def fit(self, X_train, y_train, x_val=None, y_val=None, epochs=200, early_stopping_patience=10, plot=False):
        """
        Fits data to the model
        Defaults:
            Epochs:200
            Early Stopping Patience: 10

        Does not require y_train to be in one hot encoding as the fit function handles for that manually
        """

        if (self.classification):
            # If its not ohe make it
            if y_train.shape == (len(y_train), ):
                y_train = self._get_ohe_labels(y_train)
                if (y_val is not None):
                    y_val = self._get_ohe_labels(y_val)

        self._create_model(X_train, y_train)

        callbacks = []
        es = EarlyStopping(monitor='loss', patience=early_stopping_patience)
        validation_data=None
        if (x_val is not None):
            validation_data = (x_val, y_val)
            es = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
        callbacks.append(es)
        r = self.model.fit(X_train, y_train, epochs=epochs,callbacks=callbacks, validation_data=validation_data)

        if(plot):
            plt.plot(r.history['loss'], label='loss')
            if(x_val is not None):
                plt.plot(r.history['val_loss'], label='val_loss')

            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()

            if (self.classification):
                plt.plot(r.history['acc'], label='acc')
                if(x_val is not None):
                    plt.plot(r.history['val_acc'], label='val_acc')
                plt.title('Model accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                plt.show()
        return

    def _get_ohe_labels(self, y):
        from keras.utils import to_categorical
        return to_categorical(y)
    
    def _get_num_categories(self, y, in_ohe=False):
        if in_ohe:
            return y.shape[1]
        return len(set(y))

    def predict(self, X):
        if self.classification:
            return np.argmax(self.model.predict(x), axis=1)
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities
        """
        if (self.classification):
            return self.model.predict(X)
        else:
            raise RuntimeError("Calling PredictProba on a regression network is forbidden")

    def score(self, X, y):
        """
        If the net is a regressor then it will return r^2 loss of the predictions (This is what SKLearn uses)
        If the net is a classifier then it will return accuracy
        """
        if self.classification:
            if(y.shape == (len(y), )):
                y = self._get_ohe_labels(y)
            return self.model.evaluate(X, y)
        else:
            from sklearn.metrics import r2_score
            return r2_score(y, self.predict(X))


    def _create_param_dict(self, params, optimizer, initial_learning_rate, loss):
        params['optimizer'] = optimizer
        params['initial_learning_rate'] = initial_learning_rate
        params['loss'] = loss
        self.param_dict = params
        return




