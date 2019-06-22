# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from dataset import get_dataset
from word2vec import encode_dataset  

#Get word2vec transformation of data
max_len = 250
enc_dim = 100
X, y = encode_dataset(get_dataset('spamham.csv'), max_len, enc_dim)

#Set parameters based on number of classes in the dataset
if len(set(y)) == 2:    
    num_classes = 1 
    loss = 'binary_crossentropy'
    activation = 'sigmoid'
else:
    num_classes = len(set(y))
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    y = to_categorical(y, num_classes=num_classes)
y = np.array(y, dtype=np.float32).reshape(len(y),1)


#Split the data in train/test partitions  
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


#Create the model
#Input: (None, max_len, enc_dim)
model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='sigmoid', return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation=activation)) 
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss=loss, optimizer=opt,metrics=['accuracy'], )


for layer in model.layers:
    print(layer.name + '  ||  In: ' + str(layer.input_shape) + '  ||  ' + 'Out: ' + str(layer.output_shape) )
    
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()






