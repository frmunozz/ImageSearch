import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from stop_words import get_stop_words
sns.set()

from src.regressor import Regressor, plot_metrics, r_square, rmse
from src.filehandler import DataHandler

'''
INSTRUCCIONES:

deben ejecutar este script estando en la carpeta raiz (al mismo nivel que /src, /model y los notebooks), basta con correr

python script_fit_models.py

'''
# aqui entrenare los modelos, despues esta celda desaparecera

stop_words = get_stop_words('spanish')
# debemos indicar el directorio que contiene los datos y los sub-directorios que contienen cada dataset
data = DataHandler(root_folder="./data", train_folder="/train_data", test_folder="/test_A_data")

# CV ngram (1, 1) none
X_train, X_test, y_train, y_test = data.get_data(method="count-vectorizer", stop_words=None, ngram_range=(1, 1))
print("shapes")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
in_dim = X_train.shape[1]
out_dim = y_train.shape[1]
reg = Regressor(input_dim=in_dim, output_dim=out_dim, n=2, layer_size=200, 
                loss='mean_squared_error', optimizer='adam', metrics=['mae', rmse, r_square])
reg.build()
history = reg.fit(X_train, y_train, epochs=100, batch_size=200, validation_split=0.2, shuffle=True)
reg.save("cv_ngram11_none")

# CV ngram (1, 4) none
X_train, X_test, y_train, y_test = data.get_data(method="count-vectorizer", stop_words=None, ngram_range=(1, 4))
print("shapes")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
in_dim = X_train.shape[1]
out_dim = y_train.shape[1]
reg = Regressor(input_dim=in_dim, output_dim=out_dim, n=2, layer_size=200, 
                loss='mean_squared_error', optimizer='adam', metrics=['mae', rmse, r_square])
reg.build()
history = reg.fit(X_train, y_train, epochs=100, batch_size=200, validation_split=0.2, shuffle=True)
reg.save("cv_ngram14_none")

# CV ngram (1, 4) spanish
X_train, X_test, y_train, y_test = data.get_data(method="count-vectorizer", stop_words=stop_words, ngram_range=(1, 4))
print("shapes")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
in_dim = X_train.shape[1]
out_dim = y_train.shape[1]
reg = Regressor(input_dim=in_dim, output_dim=out_dim, n=2, layer_size=200, 
                loss='mean_squared_error', optimizer='adam', metrics=['mae', rmse, r_square])
reg.build()
history = reg.fit(X_train, y_train, epochs=100, batch_size=200, validation_split=0.2, shuffle=True)
reg.save("cv_ngram14_stop_words")


# aqui entrenare los modelos, despues esta celda desaparecera

stop_words = get_stop_words('spanish')
# debemos indicar el directorio que contiene los datos y los sub-directorios que contienen cada dataset
data = DataHandler(root_folder="./data", train_folder="/train_data", test_folder="/test_A_data")

# CV ngram (1, 1) none
X_train, X_test, y_train, y_test = data.get_data(method="tf-idf", stop_words=None, ngram_range=(1, 1))
print("shapes")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
in_dim = X_train.shape[1]
out_dim = y_train.shape[1]
reg = Regressor(input_dim=in_dim, output_dim=out_dim, n=2, layer_size=200, 
                loss='mean_squared_error', optimizer='adam', metrics=['mae', rmse, r_square])
reg.build()
history = reg.fit(X_train, y_train, epochs=100, batch_size=200, validation_split=0.2, shuffle=True)
reg.save("tf_idf_ngram11_none")

# CV ngram (1, 4) none
X_train, X_test, y_train, y_test = data.get_data(method="tf-idf", stop_words=None, ngram_range=(1, 4))
print("shapes")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
in_dim = X_train.shape[1]
out_dim = y_train.shape[1]
reg = Regressor(input_dim=in_dim, output_dim=out_dim, n=2, layer_size=200, 
                loss='mean_squared_error', optimizer='adam', metrics=['mae', rmse, r_square])
reg.build()
history = reg.fit(X_train, y_train, epochs=100, batch_size=200, validation_split=0.2, shuffle=True)
reg.save("tf_idf_ngram14_none")

# CV ngram (1, 4) spanish
X_train, X_test, y_train, y_test = data.get_data(method="tf-idf", stop_words=stop_words, ngram_range=(1, 4))
print("shapes")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
in_dim = X_train.shape[1]
out_dim = y_train.shape[1]
reg = Regressor(input_dim=in_dim, output_dim=out_dim, n=2, layer_size=200, 
                loss='mean_squared_error', optimizer='adam', metrics=['mae', rmse, r_square])
reg.build()
history = reg.fit(X_train, y_train, epochs=100, batch_size=200, validation_split=0.2, shuffle=True)
reg.save("tf_idf_ngram14_stop_words")