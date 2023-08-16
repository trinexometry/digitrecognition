import numpy as np
import cv2
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from keras.optimizers import SGD


#train test split
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#The values will now range between 0-1(scaling)
X_train = X_train/255   
X_test = X_test/255

#setting up the layers of the neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation = 'softmax')
])

#compiling the model
model.compile(optimizer = SGD(learning_rate = 0.01, momentum=0.9), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, Y_train, epochs=10)

Y_pred = model.predict(X_test)

model.save("DigitRecognition.h5")