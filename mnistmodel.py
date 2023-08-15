import numpy as np
import cv2
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist


#train test split
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#The values will now range between 0-1(scaling)
X_train = X_train/255   
X_test = X_test/255

#setting up the layers of the neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(50, activation = 'relu'),
    keras.layers.Dense(50, activation = 'relu'),
    keras.layers.Dense(10, activation = 'sigmoid')
])

#compiling the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, Y_train, epochs=10)

Y_pred = model.predict(X_test)

model.save("DigitRecognition.h5")