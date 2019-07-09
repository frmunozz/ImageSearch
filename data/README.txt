Para cargar los descriptores en Python se puede usar la siguiente función:
----------------------------------------------------------
import numpy
import os

def load_file(file_names, file_vectors, num_vectors, vector_dimensions):
    assert os.path.isfile(file_names), "no existe archivo " + file_names
    assert os.path.isfile(file_vectors), "no existe archivo " + file_vectors
    print("leyendo " + file_names)
    names = [line.strip() for line in open(file_names)]
    assert num_vectors == len(names), "no cuadra largo archivo " + len(names)
    print("leyendo " + file_vectors)
    mat = numpy.fromfile(file_vectors, dtype=numpy.float32)
    vectors = numpy.reshape(mat, (num_vectors, vector_dimensions))
    print(str(num_vectors) + " vectores de largo " + str(vector_dimensions))
    return (names, vectors)

def load_train_vectors():
    return load_file("train_images_names.txt", "train_images_vectors.bin", 20000, 2048)
    
def load_test_vectors():
    return load_file("test_A_images_names.txt", "test_A_images_vectors.bin", 1000, 2048)

(train_names, train_vectors) = load_train_vectors()
(test_names, test_vectors) = load_test_vectors()

print("Imagen \"" + train_names[0] + "\" tiene descriptor visual " + str(train_vectors[0]) + " de dimension " + str(len(train_vectors[0])))
----------------------------------------------------------


Para leer los captions de train y test se puede usar la siguiente funcion:

----------------------------------------------------------
def load_captions(file_captions):
    assert os.path.isfile(file_captions), "no existe archivo " + file_captions
    return [line.strip().split("\t") for line in open(file_captions, encoding='utf-8')]

test_captions = load_file("test_A_captions.txt")
train_captions = load_file("train_captions.txt")

for i in range(6):
    print("Imagen \"" + train_captions[i][0] + "\" tiene caption \"" + train_captions[i][1] + "\"")
----------------------------------------------------------
