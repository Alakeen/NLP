# -*- coding: utf-8 -*-

import nltk
from tensorflow import keras 
from gensim.sklearn_api import W2VTransformer


def encode_dataset(dataset, max_len, enc_dim):
    # Create a model to represent each word by a enc_dim dimensional vector,
    # transform only max_len words to represent each document
        
    dataset['text'] = dataset['text'].map(lambda text:  nltk.tokenize.word_tokenize(text))
    dataset['length'] = dataset['text'].map(lambda text: len(text))
    
    model = W2VTransformer(size=enc_dim, min_count=1, seed=1)
    wordvecs = model.fit(dataset['text'].values)
    
    embedings = []
    targets = []
    for row in range(len(dataset['text'])):
        if dataset['length'][row] <= max_len:
            embedings.append(wordvecs.transform(dataset['text'][row]))
            targets.append(dataset['spam'][row])
        else:
            embedings.append(wordvecs.transform(dataset['text'][row][:max_len]))
            targets.append(dataset['spam'][row])
            
    x_lstm_sentence_seq = keras.preprocessing.sequence.pad_sequences(embedings)
    
    
    return x_lstm_sentence_seq, targets