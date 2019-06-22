# -*- coding: utf-8 -*-

import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def get_dataset(filepath):
    
    ps = nltk.PorterStemmer()
    #Read the data from .csv file
    dataset = pd.read_csv(filepath)
    
    #Setting the length column
    dataset['length'] = dataset['text'].map(lambda text: len(text))
    
    #Tokenization
    dataset['text'] = dataset['text'].map(lambda text:  nltk.tokenize.word_tokenize(text))
    
    #Removing stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    stop_words.update(['hou', 'cc', 'ect'])
    dataset['text'] = dataset['text'].map(lambda tokens: [w for w in tokens if not w in stop_words]) 
    
    #Every mail starts with 'Subject :' lets remove this from each mail 
    dataset['text'] = dataset['text'].map(lambda text: text[2:])
    dataset['text'] = dataset['text'].map(lambda text: [ps.stem(word) for word in text])
    dataset['text'] = dataset['text'].map(lambda text: ' '.join(text))
    
    #removing apecial characters from each mail 
    dataset['text'] = dataset['text'].map(lambda text: re.sub('[^A-Za-z0-9]+', ' ', text))
    
    
    dataset = dataset.sample(frac=1)
    print(dataset.columns)
    print(dataset.head())
    print(dataset.groupby('spam').count())
    
    return dataset


def word_cloud(dataset, class_label):
    
    #Wordcloud of mails with class_label
    words = ''.join(list(dataset[dataset['spam']==class_label]['text']))
    wordclod = WordCloud(width = 512,height = 512).generate(words)
    plt.figure(figsize = (10, 8), facecolor = 'k')
    plt.imshow(wordclod)
    plt.axis('off')
    plt.tight_layout(pad = 0)
    plt.show()
    
    return
    
    
