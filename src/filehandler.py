import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def load_file(file_names, file_vectors, num_vectors, vector_dimensions):
	assert os.path.isfile(file_names), "no existe archivo " + file_names
	assert os.path.isfile(file_vectors), "no existe archivo " + file_vectors
	print("leyendo " + file_names)
	names = [line.strip() for line in open(file_names)]
	assert num_vectors == len(names), "no cuadra largo archivo " + len(names)
	print("leyendo " + file_vectors)
	mat = np.fromfile(file_vectors, dtype=np.float32)
	vectors = np.reshape(mat, (num_vectors, vector_dimensions))
	print(str(num_vectors) + " vectores de largo " + str(vector_dimensions))
	return (names, vectors)


def load_captions(file_captions):
	assert os.path.isfile(file_captions), "no existe archivo " + file_captions
	return [line.strip().split("\t") for line in open(file_captions, encoding='utf-8')]

# crearemos una clase que simplifique todo el proceso de adquirir los datos
class DataHandler(object):
	def __init__(self, root_folder="./", train_folder="", test_folder=""):
		self.root = root_folder
		self.vectors_dimension = 2048
		self.test_folder = test_folder
		self.train_folder = train_folder

	def set_folders(root_folder=None, train_folder=None, test_folder=None):
		if root_folder is not None:
			self.root_folder = root_folder
		if train_folder is not None:
			self.train_folder = train_folder
		if test_folder is not None:
			self.test_folder = test_folder 

	def _load(self, names_file, vectors_file, captions_file, num_vectors):
		(names, vectors) = load_file(names_file, vectors_file, num_vectors, self.vectors_dimension)
		captions = load_captions(captions_file)
		vectors_for_captions = []
		for vector in vectors:
			for _ in range(5):
				vectors_for_captions.append(vector)
		return names, np.array(vectors_for_captions), np.array(captions)

	def load_train(self):
		names_file = "{}{}/train_images_names.txt".format(self.root, self.train_folder)
		captions_file = "{}{}/train_captions.txt".format(self.root, self.train_folder)
		vectors_file = "{}{}/train_images_vectors.bin".format(self.root, self.train_folder)
		return self._load(names_file, vectors_file, captions_file, 20000)

	def load_test(self):
		names_file = "{}{}/test_A_images_names.txt".format(self.root, self.test_folder)
		captions_file = "{}{}/test_A_captions.txt".format(self.root, self.test_folder)
		vectors_file = "{}{}/test_A_images_vectors.bin".format(self.root, self.test_folder)
		return self._load(names_file, vectors_file, captions_file, 1000)

	def load_simple_test(self):
		names_file = "{}{}/test_A_images_names.txt".format(self.root, self.test_folder)
		captions_file = "{}{}/test_A_captions.txt".format(self.root, self.test_folder)
		vectors_file = "{}{}/test_A_images_vectors.bin".format(self.root, self.test_folder)
		(names, vectors) = load_file(names_file, vectors_file, 1000, self.vectors_dimension)
		return np.array(vectors)

	def get_data(self, method="tf-idf", stop_words=None, ngram_range=(1, 3), max_df=0.8, min_df=0.01):
		train_names, train_vectors, train_image_captions = self.load_train()
		test_names, test_vectors, test_image_captions = self.load_test()

		train_captions = train_image_captions[:,1]
		test_captions = test_image_captions[:,1]

		if method == "tf-idf":
			print("doing vectorization with TfidfVectorizer")
			vectors = TfidfVectorizer(lowercase=True,ngram_range=ngram_range,max_df=max_df,min_df=min_df, stop_words=stop_words)
		elif method == "count-vectorizer":
			print("doing vectorization with CountVectorizer")
			vectors = CountVectorizer(lowercase=True,ngram_range=ngram_range,max_df=max_df,min_df=min_df,binary=False, stop_words=stop_words)
		else:
			raise ValueError("method '{}' is not valid".format(method))

		print("fitting ...", end="")
		vectors.fit(train_captions)
		print("done")

		print("getting vectors transforms ...", end="")
		train_text_descriptors = vectors.transform(train_captions)
		test_text_descriptors = vectors.transform(test_captions)
		print("done")

		return train_text_descriptors, test_text_descriptors, train_vectors, test_vectors


    # def get_data_count_vectorizer(self, train_folder="/train_data", test_folder="/test_A_data", **kwargs):
    #     train_names, train_vectors, train_image_captions = self.load_train(folder=train_folder)
    #     test_names, test_vectors, test_image_captions = self.load_test(folder=test_folder)
        
    #     print("doing vectorization with CountVectorizer")
    #     train_captions = train_image_captions[:,1]
    #     test_captions = test_image_captions[:,1]
    #     vectorizer = CountVectorizer(lowercase=False,ngram_range=(1,1),max_df=0.8,min_df=0.01,binary=True)
    #     print("fitting bag of words ...", end="")
    #     vectorizer.fit(train_captions)
    #     print("done")
        
    #     print("getting vectors transforms ...", end="")
    #     train_text_descriptors = vectorizer.transform(train_captions)
    #     test_text_descriptors = vectorizer.transform(test_captions)
    #     print("done")

    #     return train_text_descriptors, test_text_descriptors, train_vectors, test_vectors

    # #esta es una version de get_data_count_vectorizer, solo cambia el vectorize por tfidf, se puede
    # #hacer en una sola funcion, esta es solo para probar
    # def get_data_tfidf_vectorizer(self, train_folder="/train_data", test_folder="/test_A_data", **kwargs):
    #     train_names, train_vectors, train_image_captions = self.load_train(folder=train_folder)
    #     test_names, test_vectors, test_image_captions = self.load_test(folder=test_folder)
        
    #     print("doing vectorization with TfidfVectorizer")
    #     train_captions = train_image_captions[:,1]
    #     #print(train_captions[0])
    #     test_captions = test_image_captions[:,1]
    #     vectorizer = TfidfVectorizer(lowercase=False,ngram_range=(1,4),max_df=0.8,min_df=0.01,
    #                                  stop_words=stop_words)
    #     print("fitting tf idf ...", end="")
    #     vectorizer.fit(train_captions)
    #     #print(vectorizer.get_feature_names())
    #     print("done")
        
    #     print("getting vectors transforms ...", end="")
    #     train_text_descriptors = vectorizer.transform(train_captions)
    #     test_text_descriptors = vectorizer.transform(test_captions)
    #     print("done")

    #     return train_text_descriptors, test_text_descriptors, train_vectors, test_vectors