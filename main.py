import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle

# Path to images of bills
BILL_IMG_DIR = 'C:/Users/Filip/Desktop/Projekte/Diplomarbeit/ai/bills'
IMG_SIZE = 3000

training_data = []


# Iterate through directory with images of bills
for img in os.listdir(BILL_IMG_DIR):
    # Read image as array
    img_pixels_array = cv2.imread(os.path.join(BILL_IMG_DIR, img), cv2.IMREAD_GRAYSCALE)
    print(img_pixels_array.shape)

    resized_pixels_array = cv2.resize(img_pixels_array, (IMG_SIZE, IMG_SIZE))

    ### Print image in plot
    plt.imshow(resized_pixels_array, cmap="gray")
    plt.show()

    training_data.append([resized_pixels_array, []])

X = []
Y = []

for x, y in training_data:
    X.append(x)
    y.append(y)

X = np.array(X).reshape((-1, IMG_SIZE, IMG_SIZE, 1))

pickle.dump(X, open("X.pickle", "wb"))
pickle.dump(Y, open("Y.pickle", "wb"))



