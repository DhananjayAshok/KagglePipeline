from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
import matplotlib.pyplot as plt
from ...ModelBase import ModelBase

class SimpleHiddenMLP(ModelBase):
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
   

    def __init__(self, X_train, y_train, classification, n_hidden = 32 ,optimizer=None, loss=None):
        """
        Returns a fit model with Input -> 32 Hidden Neurons -> Output for classification. Unless parameters are changed.
        Default Learning Rate Scheduling is an Adam optimizer with Exponential Decay off an initial learning rate of 0.001
        """
        self.classification = classification
        self._create_model(X_train, y_train, n_hidden, classification, optimizer, loss)
        return

    def _create_model(self, X_train, y_train, n_hidden, classification, optimizer, loss):
        
        i = Input(shape=X_train.shape[1:])
        x = Dense(X_train.shape[1], activation ="elu") (i)
        x = Dense(n_hidden, activation= "relu") (x) 
        
        l = None
        metrics = []
        if(classification):
            n_outputs = self._get_num_categories(y_train)
            x = Dense(n_outputs, activation = "softmax") (x)

            l = CategoricalCrossentropy()
            metrics.append('accuracy')

        else:
            n_outputs = 1
            x = Dense(n_outputs, activation = "relu") (x)
            l = MeanSquaredError()

        initial_learning_rate = 0.001
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

    def fit(self, X_train, y_train, x_val=None, y_val=None, epochs=200, early_stopping_patience=10):
        """
        Fits data to the model
        Defaults:
            Epochs:200
            Early Stopping Patience: 10
        """
        callbacks = []
        es = EarlyStopping(monitor='loss', patience=early_stopping_patience)
        validation_data=None
        if (x_val is not None):
            validation_data = (x_val, y_val)
            es = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
        callbacks.append(es)
        r = self.model.fit(X_train, y_train, epochs=epochs,callbacks=callbacks, validation_data=validation_data)


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

    def _get_num_categories(self, y):
        return len(set(y))

    def predict(self, X):
        return self.model.predict(X)






