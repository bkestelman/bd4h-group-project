from pyspark.ml.linalg import Vectors, VectorUDT

def list_to_vector(l):
    return Vectors.dense(l)

