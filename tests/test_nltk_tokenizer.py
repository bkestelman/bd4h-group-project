import pytest
from  nltk_tokenizer import extract_words, remove_stopwords

class TestNLTK(object):
    def test_remove_stopwords(self):
        test_words = ['adam', 'is', 'the', 'president']
        expected = ['adam','president']
        assert remove_stopwords(test_words) == expected

    def test_extract_words(self):
        test_words = 'adam is the president'
        expected = ['adam', 'is', 'the', 'president']
        assert extract_words(test_words) == expected
