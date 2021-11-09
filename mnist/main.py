import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow.tensorflow

mlflow.tensorflow.autolog()

""" Loading Dataset """
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

""" Split Dataset in new Train and Test"""
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

train_size = 0.7    # modify train size
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=1337)

""" Normailize Data to 0 - 1 """
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

""" Model Creation """
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

""" Choose Optimizer and Loss Function """
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

""" Training """
model.fit(x_train, y_train, epochs=7)

""" Evaluation """
loss, accuracy = model.evaluate(x_test, y_test)

print("Accuracy is : ", accuracy)
print("Loss is : ", loss)
