from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle as shfle
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import numpy as np
import matplotlib.pyplot as plt
from .validator import validate_keys
import json
from keras.models import load_model


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


class Regressor(object):
    
    def __init__(self, **kwargs):
        self.kwargs = validate_keys(kwargs, _VALID_KWARGS)

    def __getitem__(self, key):
        if key in self.kwargs.keys():
            return self.kwargs[key]
        else:
            return getattr(self, key)

    def set(**kwargs):
        self.kwargs = validate_keys(kwargs, self.kwargs)
        
    def build(self):
        if self['input_dim'] is None or self['output_dim'] is None:
            raise ValueError("cannot build model. no input/output dimension defined")
        model = Sequential()
        model.add(Dense(self["layer_size"], input_dim=self["input_dim"], kernel_initializer='normal', activation='relu'))
        i = self["n"] - 1
        while i > 0:
            model.add(Dense(self["layer_size"], activation='relu'))
            i -= 1
        model.add(Dense(self["output_dim"], activation='linear'))
        model.summary()
        model.compile(loss=self["loss"], optimizer=self["optimizer"], metrics=self["metrics"])
        self.kwargs["model"] = model
        
    def fit(self, X_train, y_train, **kwargs):
        self.kwargs = validate_keys(kwargs, self.kwargs)
        # shuffle data
        X_train, y_train = shfle(X_train, y_train, random_state=0)
        # necesitamos desordenar los datos antes para que la particion de validacion si tome datos variados.
        if self["scale"]:
            self.kwargs["scaler_input"] = MaxAbsScaler()
            self.kwargs["scaler_input"].fit(X_train)
            X_train = self.kwargs["scaler_input"].transform(X_train)
            self.kwargs["scaler_output"] = MaxAbsScaler()
            self.kwargs["scaler_output"].fit(y_train)
            y_train = self.kwargs["scaler_output"].transform(y_train)

        history=self.kwargs["model"].fit(X_train,y_train, epochs=self["epochs"], 
                                         validation_split=self["validation_split"],shuffle=self["shuffle"],
                                         batch_size=self["batch_size"], callbacks=[self.kwargs['early-stopping']])
        self.kwargs["history"] = history.history
        return history
    
    def save(self, name):
        print("saving model to disk ...", end="")
        # save model to hdf5
        self.kwargs["model"].save("./model/" + name + ".h5")

        # save history
        json.dump(self.kwargs["history"], open("./model/" + name + "_history.json", "w"))
        print(" done!")
    
        
    def load(self, name, custom_objects=None):
        print("loading model from disk ...", end="")
        self.kwargs["model"] = load_model("./model/" + name + ".h5", custom_objects=custom_objects)
        # load history
        self.kwargs["history"] = json.load(open("./model/" + name + "_history.json", 'r'))
        print(" done!")

        # print("Loaded model from disk")
        # model.compile(loss=self["loss"], optimizer=self["optimizer"], metrics=self["metrics"])
        # self.kwargs["model"] = model
        return self.kwargs["history"]
    
    def evaluate(self, X_test, y_test):
        print("evaluando test")
        if self["scale"]:
            X_test = self.kwargs["scaler_input"].transform(X_test)
            y_test = self.kwargs["scaler_output"].transform(y_test)
        score = self.kwargs["model"].evaluate(X_test, y_test, verbose=True)
        print("    EVALUATION RESULTS :  ")
        for i in range(len(score)):
            print("%s: %.2f" % (self.kwargs["model"].metrics_names[i], score[i]))
        return score
        
    def get_model(self):
        return self["model"]
    
    def set_scaled(self, X_train, y_train):
        self.kwargs["scaler_input"] = MaxAbsScaler()
        self.kwargs["scaler_input"].fit(X_train)
        self.kwargs["scaler_output"] = MaxAbsScaler()
        self.kwargs["scaler_output"].fit(y_train)
        self.kwargs["scale"] = True
        
    def predict(self, X):
        if self["scale"]:
            X = self.kwargs["scaler_input"].transform(X)
        y_pred = self.kwargs["model"].predict(X)

        if self["scale"]:
            y_pred = self.kwargs["scaler_output"].inverse_transform(y_pred) 
            X_reversed = self.kwargs["scaler_input"].inverse_transform(X)
            assert X == X_reversed
        return y_pred

    def plot_metrics(self, main_title="metricas", name=None):
        metrics = self["model"].__dict__["metrics"]
        fig, ax = plt.subplots(1, len(metrics), figsize=(16, 4))
        for i, k in enumerate(metrics):
            if not isinstance(k, str):
                k = k.__name__
            elif k == "mae":
                k = "mean_absolute_error"
            ax[i].plot(self['history'][k])
            ax[i].plot(self['history']['val_' + k])
            ax[i].set_title('model ' + k)
            ax[i].set_ylabel(k)
            ax[i].set_xlabel('epoch')
            ax[i].legend(['train', 'validation'], loc='upper left')

        fig.suptitle(main_title)
        plt.tight_layout()
        fig.subplots_adjust(top=0.88)
        if name is not None:
            plt.savefig("./figures/" + name + ".png", dpi=300)

        return fig, ax



def plot_metrics(history, metrics_keys, name="model"):
    metrics = self["model"].__dict__["kwargs"]["metrics"]

    fig, ax = plt.subplots(1, len(metrics_keys), figsize=(16, 4))
    for i, k in enumerate(metrics_keys):
        ax[i].plot(history[k])
        ax[i].plot(history['val_' + k])
        ax[i].set_title('model ' + k)
        ax[i].set_ylabel(k)
        ax[i].set_xlabel('epoch')
        ax[i].legend(['train', 'validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig("./figures/" + name + ".png", dpi=300)

    return fig, ax