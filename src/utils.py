from joblib import Parallel, delayed
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

tree = None
pred_vectors = None

def ordered_distance(i):
	return tree.query(pred_vectors[i],k=1000)[1]

def load(a_tree, a_pred_vectors):
	global tree, pred_vectors
	tree = a_tree
	pred_vectors = a_pred_vectors

def paral_query(tree, pred_vectors):
	load(tree,pred_vectors)
	n = len(pred_vectors)
	res = Parallel(n_jobs=-1, verbose=4, backend="multiprocessing", batch_size = 5)(
             map(delayed(ordered_distance), np.array(range(n))))
	res = np.array(res)
	return res

def print_histo(rank, titulo='graph',bins = 1000):
	plt.title(titulo)
	plt.hist(rank,bins)
	plt.show()

def get_table(data, indexs, columns_labels):
	return pd.DataFrame(data=data, index=indexs, columns=columns_labels)

