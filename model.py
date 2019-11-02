import os
import pickle
import time
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

# create tensorboard for statistics
NAME = f"bill_detection_cnn_{int(time.time())}"
tensorboard = TensorBoard(log_dir=os.path.join("logs", NAME))

# load preprocessed data
X = pickle.load(open("pickles/X.pickle", "rb"))
Y = pickle.load(open("pickles/Y.pickle", "rb"))

# Create model
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), input_shape=X.shape[1:], activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), input_shape=X.shape[1:], activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation="relu"))

model.add(Dense(8, activation="relu"))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
model.summary()

# run model with gpu
with tf.device("/gpu:0"):
    model.fit(X, Y, batch_size=1, epochs=400, callbacks=[tensorboard])
    model.save(os.path.join("model", "main.h5"))

# tensorboard --logdir=logs/
