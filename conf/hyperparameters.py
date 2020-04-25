word2vec_params = {
    'vectorSize': 50, # Increasing vectorSize increases accuracy but decreases speed during word2vec fitting and during logistic regression
    'numPartitions': 1, # Increasing numPartitions increases speed but decreases accuracy during word2vec fitting (but doesn't affect logistic regression speed)
}

fit_limits = {
    'word2vec': None,
}

bag_of_words_params = {
    'vocabSize': 3000,
}

tf_idf_params = {
    'vocabSize': 3000,
}


