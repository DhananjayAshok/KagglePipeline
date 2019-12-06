from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
import matplotlib.pyplot as plt

class SimpleHiddenMLP(object):
    """A Densely Connected Feed Forward MLP which can be used for most classification and regression tasks

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
   

    def __init__(self, X_train, y_train, X_val = None, y_val = None, n_hidden = 32, classification=True,optimizer=None, loss=None, epochs=100):
        """
        Returns a fit model with Input -> 32 Hidden Neurons -> Output for classification. Unless parameters are changed.
        """
        self._create_model(X_train, y_train, n_hidden, classification, optimizer, loss)
        self._fit_model(X_train, y_train, epochs)
        return

    def _create_model(self, X_train, y_train, n_hidden, classification, optimizer=None, loss=None):
        
        i = Input(shape=X_train.shape[1:])
        x = Dense(X_train.shape[1], activation ="elu") (i)
        x = Dense(n_hidden, activation= "relu") (x)
        
        
        loss = None
        if(classification):
            n_outputs = self._get_num_categories(y_train)
            x = Dense(n_outputs, activation = "softmax") (x)

            l = CategoricalCrossentropy()


        else:
            n_outputs = 1
            x = Dense(n_outputs, activation = "relu") (x)
            l = MeanSquaredError()

        opt = Adam(0.001)

        if(loss):
            l = loss
        if (optimizer):
            opt = optimizer

        self.model = Model(i, x)

        self.model.compile(optimizer=opt,
              loss=l,
              metrics=['accuracy'])
        return




           

        

        


        return 

    def _fit_model(self, X_train, y_train, epochs):
        r = self.model.fit(X_train, y_train, epochs=epochs)
        #plt.plot(r.history['loss'], label='loss')
        #plt.show()
        return

    def _get_num_categories(self, y):
        return len(set(y))

    def predict(self, X, y):
        return self.model.predict(X, y)





