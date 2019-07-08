from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle as shfle
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from .validator import validate_keys
import json
from keras.models import load_model


# valid kwargs for the class
_VALID_KWARGS = {"input_dim": None, 
                 "output_dim": None,
                 "n": 2,
                 "layer_size": 2048,
                 "scale": False,
                 "loss": "mean_squared_error",
                 "optimizer": "adam",
                 "metrics": ["mae", "accuracy"],
                 "model": None,
                 "history": None,
                 "early-stopping": EarlyStopping(monitor="val_loss", patience=4, verbose=True, mode="min", min_delta=0.001),
                 "scaler_input": None,
                 "scaler_output": None,
                 "epochs": 10,
                 "batch_size": 256,
                 "validation_split": 0.2,
                 "shuffle": True,
                 }

# coefficient of determination (R^2) for regression  (only for Keras tensors)
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# root mean squared error (rmse) for regression (only for Keras tensors)
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression  (only for Keras tensors)
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)


# class regressor to create the NN for regression
class Regressor(object):
    
    def __init__(self, **kwargs):
        """ initialize the regressor with the option to define the kwargs"""
        self.kwargs = validate_keys(kwargs, _VALID_KWARGS)

    def __getitem__(self, key):
        """ get a kwarg value by his key """
        if key in self.kwargs.keys():
            return self.kwargs[key]
        else:
            return getattr(self, key)

    def set(**kwargs):
        """ set the kwargs """
        self.kwargs = validate_keys(kwargs, self.kwargs)
        
    def build(self):
        """ build the NN model using the set kwargs as configurations """
        if self['input_dim'] is None or self['output_dim'] is None:
            # if no inputs/outputs dim are defined, we cannot build the NN
            raise ValueError("cannot build model. no input/output dimension defined")
        model = Sequential()
        # all hiden layers has relu activators except for the last one
        model.add(Dense(self["layer_size"], input_dim=self["input_dim"], kernel_initializer='normal', activation='relu'))
        i = self["n"] - 1
        while i > 0:
            model.add(Dense(self["layer_size"], activation='relu'))
            i -= 1
        # last layer has linear activator (i.e. no activator)
        model.add(Dense(self["output_dim"], activation='linear'))
        model.summary()
        # set loss, optimizer and metrics using the kwargs values
        model.compile(loss=self["loss"], optimizer=self["optimizer"], metrics=self["metrics"])
        # save to model to kwargs
        self.kwargs["model"] = model
        
    def fit(self, X_train, y_train, **kwargs):
        """ fit the NN using X_train as inputs and y_rain as expected outputs for the backpropagation """
        self.kwargs = validate_keys(kwargs, self.kwargs)
        # shuffle data, in order to obtain a trully random validation set
        X_train, y_train = shfle(X_train, y_train, random_state=0)

        # we implements the option to scale inputs. But in our tests, we never scale them
        if self["scale"]:
            self.kwargs["scaler_input"] = MaxAbsScaler()
            self.kwargs["scaler_input"].fit(X_train)
            X_train = self.kwargs["scaler_input"].transform(X_train)
            self.kwargs["scaler_output"] = MaxAbsScaler()
            self.kwargs["scaler_output"].fit(y_train)
            y_train = self.kwargs["scaler_output"].transform(y_train)

        # fit the model using the kwargs configuration.
        # get the history callback
        history=self.kwargs["model"].fit(X_train,y_train, epochs=self["epochs"], 
                                         validation_split=self["validation_split"],shuffle=self["shuffle"],
                                         batch_size=self["batch_size"], callbacks=[self.kwargs['early-stopping']])
        # save the history callback dictionary in a kwarg parameter
        self.kwargs["history"] = history.history
        return history
    
    def save(self, name, verbose=True):
        """ save the trained model to disk  in h5 files, also save history dictionary in a json """
        if verbose:
            print("saving model to disk ...", end="")
        # save model to hdf5
        self.kwargs["model"].save("./model/" + name + ".h5")

        # save history
        json.dump(self.kwargs["history"], open("./model/" + name + "_history.json", "w"))
        if verbose:
            print(" done!")
    
        
    def load(self, name, custom_objects=None, verbose=True):
        """ load model and history from disk. It could need to define some custom objects (metrics rmse and r_square )"""
        if verbose:
            print("loading model from disk ...", end="")
        self.kwargs["model"] = load_model("./model/" + name + ".h5", custom_objects=custom_objects)
        # load history
        self.kwargs["history"] = json.load(open("./model/" + name + "_history.json", 'r'))
        if verbose:
            print(" done!")
        return self.kwargs["history"]
    
    def evaluate(self, X_test, y_test, verbose=True):
        """ evaluate a trained model with a test dataset """
        if verbose:
            print("evaluando test")
        if self["scale"]:
            X_test = self.kwargs["scaler_input"].transform(X_test)
            y_test = self.kwargs["scaler_output"].transform(y_test)
        score = self.kwargs["model"].evaluate(X_test, y_test, verbose=verbose)
        if verbose:
            print("    EVALUATION RESULTS :  ")
            for i in range(len(score)):
                print("%s: %.2f" % (self.kwargs["model"].metrics_names[i], score[i]))
        return score
    
    def set_scaled(self, X_train, y_train):
        """ set the scalers. This is never used in our experiments """
        self.kwargs["scaler_input"] = MaxAbsScaler()
        self.kwargs["scaler_input"].fit(X_train)
        self.kwargs["scaler_output"] = MaxAbsScaler()
        self.kwargs["scaler_output"].fit(y_train)
        self.kwargs["scale"] = True
        
    def predict(self, X):
        """ use the trained NN to predict the outputs for a given input (could be a vector or a matrix)"""
        if self["scale"]:
            X = self.kwargs["scaler_input"].transform(X)
        y_pred = self.kwargs["model"].predict(X)

        if self["scale"]:
            y_pred = self.kwargs["scaler_output"].inverse_transform(y_pred) 
            X_reversed = self.kwargs["scaler_input"].inverse_transform(X)
            assert X == X_reversed
        return y_pred

    def plot_metrics(self, main_title="metricas", name=None):
        """ plot the metrics from the history callback """
        metrics = ["loss"]
        metrics.extend(self["model"].__dict__["metrics"])
        fig, ax = plt.subplots(1, len(metrics), figsize=(16, 4))
        for i, k in enumerate(metrics):
            if not isinstance(k, str):
                k = k.__name__
            elif k == "mae":
                k = "mean_absolute_error"
            ax[i].plot(self['history'][k])
            ax[i].plot(self['history']['val_' + k])
            ax[i].set_title('model ' + k, fontsize=15)
            ax[i].set_ylabel(k, fontsize=14)
            ax[i].set_xlabel('epoch', fontsize=14)
            ax[i].legend(['train', 'validation'], loc='upper left', fontsize=13)

        fig.suptitle(main_title, fontsize=16)
        plt.tight_layout()
        fig.subplots_adjust(top=0.86)
        if name is not None:
            plt.savefig("./figures/" + name + ".png", dpi=300)

        return fig, ax
