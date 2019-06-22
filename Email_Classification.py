# -*- coding: utf-8 -*-

import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from validation import validate_model
from dataset import get_dataset, word_cloud
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


## LOAD DATASET 
dataset = get_dataset('spamham.csv')
#WordCloud of spam mail words
word_cloud(dataset, 1)
#WordCloud of not-spam mail words
word_cloud(dataset, 0)

## CREATE MODEL PIPELINE
#Create pipeline
text_clf = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', stop_words=None, lowercase=True)),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
#Create grid search
parameters = {'tfidf__ngram_range': [(1, 1), (2, 2)],
              'clf__alpha': (1e-2, 1e-3, 1e-4),
             }

gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)
gs_clf = gs_clf.fit(dataset['text'].values, dataset['spam'].values)

#For grid search results inspection
results = pd.DataFrame(gs_clf.cv_results_)
#coef = gs_clf.best_estimator_.named_steps['clf'].coef_

#Save model 
with open('email_classifier', 'wb') as picklefile:  
    pickle.dump(gs_clf,picklefile)
#Load trained model   
with open('email_classifier', 'rb') as training_model:  
    model = pickle.load(training_model)

y_pred = model.predict(dataset['text'].values)
print(confusion_matrix(dataset['spam'].values,y_pred))  
print(classification_report(dataset['spam'].values,y_pred))  
print(accuracy_score(dataset['spam'].values, y_pred))  

#Model checking and training stats display
validate_model(gs_clf.best_estimator_, dataset)




