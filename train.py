import pickle
import time

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard

# create tensorboard for statistics
NAME = f"bill_detection_cnn_{int(time.time())}"
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

# load preprocessed data
X = pickle.load("X.pickle", "wb")
Y = pickle.load("Y.pickle", "wb")

# normalize pixel values (gray scale!)
X = X / 255

# Create model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=X.shape[1:], activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), input_shape=X.shape[1:], activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation="relu"))

model.add(Dense(8, activation="relu"))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

model.fit(X, Y, batch_size=32, epochs=4000, callbacks=[tensorboard])

# tensorboard --logdir=logs/