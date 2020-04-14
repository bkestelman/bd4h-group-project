import pytest
from util import get_spark
from bag_of_words import BagOfWords
from pyspark.ml.linalg import SparseVector

class TestBagOfWords(object):

    def test_bagofwords(self):

        source_df = get_spark().createDataFrame(
            [
            (0,"Logistic regression models are neat"),
            (1, "We should try CNN Models too")
        ],
            ["LABEL", "TEXT"]
        )

        pipelineModel = BagOfWords(inputCol='TEXT', outputCol='FEATURES').fit(source_df)
        actual_df = pipelineModel.transform(source_df)

        expected_df = get_spark().createDataFrame(
             [
            (0,["logistic", "regression", "models", "are", "neat"],["logistic", "regression", "models","neat"],SparseVector(3, {2: 1.0})),
            (1,["we", "should", "try", "cnn","models" ,"too"],['try', 'cnn', 'models'],SparseVector(3, {0: 1.0, 1: 1.0}))
        ],
            ["LABEL","RAW_TOKENS","TOKENS","FEATURES"]
        )
        actual_result = actual_df.select("LABEL","RAW_TOKENS","TOKENS","FEATURES").collect()
        print(actual_result)
        expected_result = expected_df.collect()
        assert(actual_result[0]["RAW_TOKENS"] == expected_result[0]["RAW_TOKENS"])
        assert(actual_result[1]["TOKENS"] == expected_result[1]["TOKENS"])
        actual_feature = [row['FEATURES'] for row in actual_result]  
        assert(len(actual_feature) == len(expected_result))
        assert(type(actual_feature[0]) == SparseVector)

