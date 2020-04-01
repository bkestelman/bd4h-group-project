import nltk
import string
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords

nltk.download('stopwords') # download stopwords
#nltk.download('punkt') # download tokenizer

stopwords = set(nltk_stopwords.words('english')) # this stopwords list can be expanded if needed
words_only_tokenizer = nltk.RegexpTokenizer(r'\w+') # tokenize words only (no punctuation or numbers)

def extract_words(text):
    """@return list of words in text, with punctuation removed"""
    return words_only_tokenizer.tokenize(text)
    # The method below includes "'s" as a token, which is not in the stopwords list. 
    #tokens = word_tokenize(text)
    #punc_list = string.punctuation+'0123456789'
    #return [word for word in tokens if word not in punc_list] 

def remove_stopwords(words):
    """
    @param words: list of words
    @return list of words with stopwords removed
    """
    return [word for word in words if word not in stopwords]

if __name__ == '__main__':
    text = "Hello. This is some text to test out nltk's word tokenizer. Let's see how it does, shall we?"
    words = extract_words(text)
    print(words)
    words = remove_stopwords(words)
    print(words)
