from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from sklearn.utils import shuffle as shfle
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import numpy as np
import matplotlib.pyplot as plt
from .validator import validate_keys


_VALID_KWARGS = {"input_dim": None, 
                 "output_dim": None,
                 "n": 2,
                 "layer_size": 2048,
                 "scale": False,
                 "loss": "mean_squared_error",
                 "optimizer": "adam",
                 "metrics": ["mae", "accuracy"],
                 "model": None,
                 "scaler_input": None,
                 "scaler_output": None,
                 "epochs": 10,
                 "batch_size": 256,
                 "validation_split": 0.2,
                 "shuffle": True,
                 }

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

        results=self.kwargs["model"].fit(X_train,y_train, epochs=self["epochs"], 
                                         validation_split=self["validation_split"],shuffle=self["shuffle"],
                                         batch_size=self["batch_size"])
        return results
    
    def save(self, model_file="model"):
        # serialize model to JSON
        model_json = self.kwargs["model"].to_json()
        with open("./model/" + model_file + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.kwargs["model"].save_weights("./model/" + model_file + ".h5")
        print("Saved model to disk")
    
        
    def load(self, model_file="model"):
        # load json and create model
        json_file = open("./model/" + model_file + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("./model/" + model_file + ".h5")
        print("Loaded model from disk")
        model.compile(loss=self["loss"], optimizer=self["optimizer"], metrics=self["metrics"])
        self.kwargs["model"] = model
    
    def evaluate(self, X_test, y_test):
        print("evaluando test")
        if self["scale"]:
            X_test = self.kwargs["scaler_input"].transform(X_test)
            y_test = self.kwargs["scaler_output"].transform(y_test)
        score = self.kwargs["model"].evaluate(X_test, y_test, verbose=True)
        print("%s: %.2f" % (self.kwargs["model"].metrics_names[1], score[1]))
        print("%s: %.2f" % (self.kwargs["model"].metrics_names[0], score[0]))
        print("%s: %.2f%%" % (self.kwargs["model"].metrics_names[2], score[2]*100))
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
    
def plot_metrics(history):
    print(history.history.keys())
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('model loss')
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'validation'], loc='upper left')

    ax[1].plot(history.history['acc'])
    ax[1].plot(history.history['val_acc'])
    ax[1].set_title('model acc')
    ax[1].set_ylabel('acc')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'validation'], loc='upper left')

    ax[2].plot(history.history['mean_absolute_error'])
    ax[2].plot(history.history['val_mean_absolute_error'])
    ax[2].set_title('model MSE')
    ax[2].set_ylabel('MSE')
    ax[2].set_xlabel('epoch')
    ax[2].legend(['train', 'validation'], loc='upper left')
    return fig, ax