import os

from tensorflow.keras.models import load_model


model = load_model(os.path.join("model/main.h5"))
