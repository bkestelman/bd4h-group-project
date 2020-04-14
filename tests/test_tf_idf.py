import pytest
from util import get_spark
from tf_idf import TfIdf
from pyspark.ml.linalg import SparseVector

class TestTF_IDF(object):

    def test_tf_idf(self):

        source_df = get_spark().createDataFrame(
            [
            (0,"logistic regression models are neat"),
            (1, "we should try cnn models too")
        ],
            ["LABEL", "TEXT"]
        )

        pipelineModel = TfIdf(inputCol='TEXT', outputCol='FEATURES').fit(source_df)
        actual_df = pipelineModel.transform(source_df)

        expected_df = get_spark().createDataFrame(
             [
           (0,["logistic", "regression", "models", "are", "neat"],["logistic", "regression", "models","neat"],SparseVector(3, {2: 1.0})),
            (1,["we", "should", "try", "cnn","models" ,"too"],['try', 'cnn', 'models'],SparseVector(3, {0: 1.0, 1: 1.0}))
         ],
            ["label","to_spark","filtered","FEATURES"]
        )
        actual_result = actual_df.collect()
        expected_result = expected_df.collect()
        assert(actual_result[0]["to_spark"] == expected_result[0]["to_spark"])
        assert(actual_result[1]["filtered"] == expected_result[1]["filtered"])
        actual_feature = [row['FEATURES'] for row in actual_result]  
        expected_feature = [row['FEATURES'] for row in expected_result]
        assert(len(actual_feature) == len(expected_result))
        assert(type(actual_feature[0]) == SparseVector)

