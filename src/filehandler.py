import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def load_file(file_names, file_vectors, num_vectors, vector_dimensions, verbose=True):
	""" function to load a dataset image descriptors """
	assert os.path.isfile(file_names), "no existe archivo " + file_names
	assert os.path.isfile(file_vectors), "no existe archivo " + file_vectors
	if verbose:
		print("leyendo " + file_names)
	names = [line.strip() for line in open(file_names)]
	assert num_vectors == len(names), "no cuadra largo archivo " + len(names)
	if verbose:
		print("leyendo " + file_vectors)
	mat = np.fromfile(file_vectors, dtype=np.float32)
	vectors = np.reshape(mat, (num_vectors, vector_dimensions))
	if verbose:
		print(str(num_vectors) + " vectores de largo " + str(vector_dimensions))
	return (names, vectors)


def load_captions(file_captions):
	""" function to load a dataset captions """
	assert os.path.isfile(file_captions), "no existe archivo " + file_captions
	return [line.strip().split("\t") for line in open(file_captions, encoding='utf-8')]

# class to handle the dataset read
class DataHandler(object):
	def __init__(self, root_folder="./", train_folder="", test_folder=""):
		""" we need to define a root folder where data are stored and the subfolders for the
		    train dataset and the test dataset """
		self.root = root_folder
		self.vectors_dimension = 2048
		self.test_folder = test_folder
		self.train_folder = train_folder

	def set_folders(root_folder=None, train_folder=None, test_folder=None):
		""" we can also set these folders """
		if root_folder is not None:
			self.root_folder = root_folder
		if train_folder is not None:
			self.train_folder = train_folder
		if test_folder is not None:
			self.test_folder = test_folder 

	def _load(self, names_file, vectors_file, captions_file, num_vectors, verbose=True, n=5):
		""" internal method to read image descriptors and captions from a dataset.
		    this method will repeate every image descriptor 5 times to match with the
		    captions matrix shape  """
		(names, vectors) = load_file(names_file, vectors_file, num_vectors, self.vectors_dimension, verbose=verbose)
		captions = load_captions(captions_file)
		vectors_for_captions = []
		for vector in vectors:
			for _ in range(n):
				vectors_for_captions.append(vector)
		return names, np.array(vectors_for_captions), np.array(captions)

	def load_train(self, verbose=True):
		""" load the data from the train set """
		names_file = "{}{}/train_images_names.txt".format(self.root, self.train_folder)
		captions_file = "{}{}/train_captions.txt".format(self.root, self.train_folder)
		vectors_file = "{}{}/train_images_vectors.bin".format(self.root, self.train_folder)
		return self._load(names_file, vectors_file, captions_file, 20000, verbose=verbose)

	def load_test(self, verbose=True, n=5):
		""" load the data from the test set """
		names_file = "{}{}/{}images_names.txt".format(self.root, self.test_folder,self.test_folder.replace('data',''))
		captions_file = "{}{}/{}captions.txt".format(self.root, self.test_folder,self.test_folder.replace('data',''))
		vectors_file = "{}{}/{}images_vectors.bin".format(self.root, self.test_folder,self.test_folder.replace('data',''))
		return self._load(names_file, vectors_file, captions_file, 1000, verbose=verbose, n=n)

	def load_simple_test(self, abc = "A", verbose=True):
		""" load just the image descriptors from the test set """
		names_file = "{}{}/test_{}_images_names.txt".format(self.root, self.test_folder, abc)
		captions_file = "{}{}/test_{}_captions.txt".format(self.root, self.test_folder, abc)
		vectors_file = "{}{}/test_{}_images_vectors.bin".format(self.root, self.test_folder,abc)
		(names, vectors) = load_file(names_file, vectors_file, 1000, self.vectors_dimension, verbose=verbose)
		return np.array(vectors)

	def get_data(self, method="tf-idf", stop_words=None, ngram_range=(1, 3), max_df=0.8, min_df=0.002, verbose=True):
		""" loadd all data  and compute the text descriptors using one of two methods: count-vectorizer
		    or tf-idf """
		train_names, train_vectors, train_image_captions = self.load_train(verbose=verbose)
		test_names, test_vectors, test_image_captions = self.load_test(verbose=verbose)

		train_captions = train_image_captions[:,1]
		test_captions = test_image_captions[:,1]

		if method == "tf-idf":
			if verbose:
				print("doing vectorization with TfidfVectorizer")
			vectors = TfidfVectorizer(lowercase=True,ngram_range=ngram_range,max_df=max_df,min_df=min_df, stop_words=stop_words)
		elif method == "count-vectorizer":
			if verbose:
				print("doing vectorization with CountVectorizer")
			vectors = CountVectorizer(lowercase=True,ngram_range=ngram_range,max_df=max_df,min_df=min_df,binary=False, stop_words=stop_words)
		else:
			raise ValueError("method '{}' is not valid".format(method))

		if verbose:
			print("fitting ...", end="")
		vectors.fit(train_captions)
		if verbose:
			print("done")

		if verbose:
			print("getting vectors transforms ...", end="")
		train_text_descriptors = vectors.transform(train_captions)
		test_text_descriptors = vectors.transform(test_captions)
		if verbose:
			print("done")

		return train_text_descriptors, test_text_descriptors, train_vectors, test_vectors

	def get_data_test(self, method="tf-idf", stop_words=None, ngram_range=(1, 3), max_df=0.8, min_df=0.002, verbose=True):
		""" loadd test data  and compute the text descriptors using one of two methods: count-vectorizer
		    or tf-idf """
		train_names, train_vectors, train_image_captions = self.load_train(verbose=verbose)
		test_names, test_vectors, test_image_captions = self.load_test(verbose=verbose)

		train_captions = train_image_captions[:,1]
		test_captions = test_image_captions[:,1]
		if method == "tf-idf":
			if verbose:
				print("doing vectorization with TfidfVectorizer")
			vectors = TfidfVectorizer(lowercase=True,ngram_range=ngram_range,max_df=max_df,min_df=min_df, stop_words=stop_words)
		elif method == "count-vectorizer":
			if verbose:
				print("doing vectorization with CountVectorizer")
			vectors = CountVectorizer(lowercase=True,ngram_range=ngram_range,max_df=max_df,min_df=min_df,binary=False, stop_words=stop_words)
		else:
			raise ValueError("method '{}' is not valid".format(method))

		if verbose:
			print("fitting ...", end="")
		vectors.fit(train_captions)
		if verbose:
			print("done")

		if verbose:
			print("getting test vectors transforms ...", end="")
		test_text_descriptors = vectors.transform(test_captions)
		if verbose:
			print("text descriptor shape:", test_text_descriptors.shape)
			print("done")

		return test_text_descriptors

