import pytest
from util import get_spark
import sparknlp
from nlp_preprocessing_tools import RawTokenizer,NoPuncTokenizer,StopWordsRemover,Stemmer,Lemmatizer,GloveWordEmbeddings,Finisher

spark = sparknlp.start()

class TestNLPProcessingTools(object):

    def test_rawpunctokenizer(self):

        source_df = get_spark().createDataFrame([
                                (1.0, "Logistic regression models are neat")
                            ], ["label", "sentence"])

        pipelineModel = RawTokenizer(inputCol='sentence', outputCol='TOKENS').fit(source_df)
        actual_df = pipelineModel.transform(source_df)
        out = actual_df.collect()
        result = [row['result'] for row in out[0]['TOKENS']]
        assert(result == ['Logistic', 'regression', 'models', 'are', 'neat'])

    def test_NoPuncTokenizer(self):
        source_df = get_spark().createDataFrame([
                                (1.0, "Logistic regression models are neat")
                            ], ["label", "sentence"])

        pipelineModel = NoPuncTokenizer(inputCol='sentence', outputCol='TOKENS')
        actual_df = pipelineModel.transform(source_df)
        assert(actual_df.collect()[0]['TOKENS'] == ['logistic', 'regression', 'models', 'are', 'neat'])

    def test_StopWordsRemover(self):
        source_df = get_spark().createDataFrame([
                                (1.0, ["Logistic", "regression", "models", "are", "neat"])
                            ], ["label", "sentence"])
        
        remover = StopWordsRemover(inputCol='sentence', outputCol='TOKENS')
        output = remover.transform(source_df)
        assert(output.collect()[0]['TOKENS'] == ['Logistic', 'regression', 'models', 'neat'])

    def test_Stemmer(self):
        pass

    def test_Lemmatizer(self):
        pass

    def test_Finisher(self):
        pass
