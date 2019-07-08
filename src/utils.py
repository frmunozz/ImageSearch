from joblib import Parallel, delayed
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
import copy
import time


tree = None
pred_vectors = None

def ordered_distance(i):
	return tree.query(pred_vectors[i],k=1000)[1]

def load(a_tree, a_pred_vectors):
	global tree, pred_vectors
	tree = a_tree
	pred_vectors = a_pred_vectors

def paral_query(my_tree, my_pred_vectors):
	# load(tree,pred_vectors)
	tree = my_tree
	pred_vectors = my_pred_vectors
	n = len(pred_vectors)
	res = Parallel(n_jobs=-1, verbose=4, backend="multiprocessing", batch_size = 2)(
             map(delayed(ordered_distance), np.array(range(n))))
	res = np.array(res)
	return res

def print_histo(rank, titulo='graph',bins = 1000):
	plt.title(titulo)
	plt.hist(rank,bins)
	plt.show()

def get_table(data, indexs, columns_labels):
	return pd.DataFrame(data=data, index=indexs, columns=columns_labels)


def worker(kdtree, query_list, ini, end, queue, job_number):
	arr = [kdtree.query(vec, k=1000)[1] for vec in query_list[ini:end]]
	queue.put((job_number, arr))

def parallel_query(kdtree, vectors, n_jobs=-1):
	if n_jobs == -1:
		n_jobs = mp.cpu_count()
	ini_time = time.time()
	print("-> launching %d jobs: " % n_jobs, end="")
	
	m = mp.Manager()
	r_queue = m.Queue()

	N = len(vectors)
	segments_size = len(vectors) // n_jobs
	jobs = []
	ini = 0
	for i in range(n_jobs-1):
		end = ini + segments_size + 1
		jobs.append(mp.Process(target=worker, args=(kdtree, vectors, ini, end, r_queue, i)))
		print(".", end="")
		jobs[-1].start()
		ini = end

	jobs.append(mp.Process(target=worker, args=(kdtree, vectors, ini, N, r_queue, n_jobs-1)))
	print(". DONE!", end="")
	jobs[-1].start()

	print(" || waiting %d jobs: " % n_jobs, end="")
	for p in jobs:
		p.join()
		print(".", end="")
	print(" DONE!", end="")

	print(" || grouping data ---", end="")
	result = {}
	while not r_queue.empty():
		par = r_queue.get()
		result[par[0]] = par[1]

	dists = []
	for i in range(n_jobs):
		dists.extend(result[i])
	dists = np.array(dists)
	end_time = time.time()
	print(" DONE! [elapse time: {}]".format(round(end_time-ini_time, 3)))
	return dists


	

