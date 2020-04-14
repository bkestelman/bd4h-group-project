import pytest
from util import get_spark
from word2vec import BasicWord2Vec,GloveWordEmbeddings
from helper_udfs import list_to_vector
import sparknlp
from pyspark.ml.linalg import DenseVector,VectorUDT

spark = sparknlp.start()
spark.udf.register('list_to_vector_udf', list_to_vector, VectorUDT())

class TestWord2Vec(object):

    def test_word2vec(self):

        source_df = get_spark().createDataFrame(
            [
            (0,"logistic regression models are neat"),
            (1, "we should try cnn models too")
        ],
            ["LABEL", "TEXT"]
        )

        pipelineModel = BasicWord2Vec(inputCol='TEXT', outputCol='FEATURES').fit(source_df)
        actual_df = pipelineModel.transform(source_df)

        expected_df = get_spark().createDataFrame(
             [
            (0,["logistic", "regression", "models", "are", "neat"],DenseVector([-0.0027, 0.0016, 0.0051])),
            (1,["we", "should", "try", "cnn","models" ,"too"],DenseVector([-0.0027, 0.0016, 0.0051]))
        ],
            ["LABEL","FINISHED_TOKENS","FEATURES"]
        )

        actual_result = actual_df.select("LABEL","FINISHED_TOKENS","FEATURES").collect()
        expected_result = expected_df.collect()
        assert(actual_result[0]["FINISHED_TOKENS"] == expected_result[0]["FINISHED_TOKENS"])
        assert(actual_result[1]["FINISHED_TOKENS"] == expected_result[1]["FINISHED_TOKENS"])
        actual_feature = [row['FEATURES'] for row in actual_result]  
        assert(len(actual_result) == len(expected_result))
        assert(type(actual_feature[0]) == DenseVector)

    def test_GloveWordEmbeddings(self):

        source_df = get_spark().createDataFrame(
            [
            (0,"logistic regression models are neat"),
            (1, "we should try cnn models too")
        ],
            ["LABEL", "TEXT"]
        )
        
        pipelineModel = GloveWordEmbeddings(inputCol='TEXT', outputCol='FEATURES').fit(source_df)
        actual_df = pipelineModel.transform(source_df)

        actual_result = actual_df.collect()
        result = [row['result'] for row in actual_result[0]['EMBEDDINGS']]
        assert(result == ["logistic", "regression", "models", "are", "neat"])
        actual_feature = [row['FEATURES'] for row in actual_result]  
        assert(type(actual_feature[0]) == DenseVector)