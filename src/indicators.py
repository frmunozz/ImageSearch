from scipy import spatial
from .validator import validate_keys
import numpy as np
import time


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

	def similarity_search(self, real_vectors, pred_vectors):
		print("initializing KDTRee ... ", end="")
		tree = spatial.KDTree(real_vectors)
		print("done!")
		print("computing distances with L2 metric ...", end="")
		ini = time.time()
		dist_vec = [tree.query(i,k=1000)[1] for i in pred_vectors]
		end = time.time()
		print("done! (elapse time: {} secs.)".format(round(end - ini, 2)))

		count_v = 0
		count_i = 0
		k1 = 0
		k5 = 0
		k10 = 0

		min_dist_vec = []
		print("getting sorted ranking ...", end="")
		for v in dist_vec:
			valor_aux = np.where(v==count_i)
			if valor_aux[0].size == 0:
				print("problemas!")
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

		print("done!")
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
