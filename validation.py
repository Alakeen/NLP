# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def validate_model(classificator, dataset):
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(classificator, dataset['text'].values, dataset['spam'].values,cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 10))
    
    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    
    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    plt.figure()
    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
    
    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    
    return