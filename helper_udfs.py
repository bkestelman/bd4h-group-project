from pyspark.ml.linalg import Vectors, VectorUDT

def list_to_vector(l):
    return Vectors.dense(l)

def vector_get(vec, i):
    return float(vec[i])

