from scipy import spatial
from .validator import validate_keys
import numpy as np
import time
from .utils import paral_query, print_histo, parallel_query

_VALID_KWARGS = {"k1": None, 
				 "k5": None,
				 "k10": None,
				 "min_dist_vec": None,
				 "n": None,
				 "MRR": None,
				 "mean": None}


class Indicators(object):
	def __init__(self, **kwargs):
		self.kwargs = validate_keys(kwargs, _VALID_KWARGS)

	def __getitem__(self, key):
		if key in self.kwargs.keys():
			return self.kwargs[key]
		else:
			return getattr(self, key)

	def get_pos_vec(self,real_vectors,pred_vectors, n_jobs=-1, verbose=True):
		if verbose:
			print("initializing KDTRee ... ", end="")
		tree = spatial.KDTree(real_vectors)
		if verbose:
			print("done!")
			print(" :::: computing distances with L2 metric :::: ")
		ini = time.time()
		# dist_vec = paral_query(tree,pred_vectors)
		dist_vec = parallel_query(tree, pred_vectors, n_jobs=n_jobs) 
		end = time.time()
		if verbose:
			print(" :::: done! (elapse time: {} secs.) :::: ".format(round(end - ini, 2)))
		return dist_vec

	def similarity_search(self, real_vectors, pred_vectors, n_jobs=-1, verbose=True):
		dist_vec = self.get_pos_vec(real_vectors,pred_vectors, n_jobs=n_jobs, verbose=verbose)
		count_v = 0
		count_i = 0
		k1 = 0
		k5 = 0
		k10 = 0

		min_dist_vec = []
		if verbose:
			print("getting sorted ranking ...", end="")
		ini = time.time()
		for v in dist_vec:
			valor_aux = np.where(v==count_i)[0][0]+1 #le puse +1 para ajustar los indices desde el 1 al 1000#estos if para ir contando cuantos valores estan bajo k
			if valor_aux<11:
				k10+=1
				if valor_aux<6:
					k5+=1
					if valor_aux<2:
						k1+=1
			min_dist_vec.append(valor_aux)
			count_v+=1
			if count_v>4:
				count_v = 0
				count_i+=1
		end = time.time()
		if verbose:
			print("done! (elapse time: {} secs.)".format(round(end - ini, 2)))
		self.kwargs["min_dist_vec"] = np.array(min_dist_vec)
		self.kwargs["k1"] = k1
		self.kwargs["k5"] = k5
		self.kwargs["k10"] = k10
		self.kwargs["n"] = len(pred_vectors)
		return self.kwargs["min_dist_vec"]

	def recall_at_1(self):
		return self.kwargs["k1"] / self.kwargs["n"]

	def recall_at_5(self):
		return self.kwargs["k5"] / self.kwargs["n"]

	def recall_at_10(self):
		return self.kwargs["k10"] / self.kwargs["n"]

	def MRR(self):
		_sum = 0
		for ri in self.kwargs["min_dist_vec"]:
			_sum += 1.0 / ri

		self.kwargs["MRR"] = _sum / self.kwargs["n"]
		return self.kwargs["MRR"]

	def mean_rank(self):
		self.kwargs["mean"] = np.mean(self.kwargs["min_dist_vec"])
		return self.kwargs["mean"]

	def get_formated_data(self):
		return [self.recall_at_1(),self.recall_at_5(),self.recall_at_10(), self.MRR(), self.mean_rank()]


