import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle

# path to images of bills
BILL_IMG_DIR = "bills"
FULL_PATH = os.path.join(os.getcwd(), BILL_IMG_DIR)

# fixed pixel size to resize to
IMG_SIZE = 3000

training_data = []

# iterate through directory with images of bills
for img in os.listdir(FULL_PATH):
    # read image pixels in array as grayscale values
    print(os.path.join(FULL_PATH, img))
    img_pixels_array = cv2.imread(os.path.join(FULL_PATH, img), cv2.IMREAD_GRAYSCALE)
    resized_pixels_array = cv2.resize(img_pixels_array, (IMG_SIZE, IMG_SIZE))
    binary_pixels_array = cv2.adaptiveThreshold(resized_pixels_array,
                                                255,
                                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY,
                                                11,
                                                5)

    # print image in plot
    plt.imshow(binary_pixels_array, cmap="gray")
    plt.show()

    # TODO: Load labels from labelbox
    training_data.append([resized_pixels_array, []])

X = []
Y = []

# crete sperate arrays for features and labels
for x, y in training_data:
    X.append(x)
    y.append(y)

X = np.array(X).reshape((-1, IMG_SIZE, IMG_SIZE, 1))


#pickle.dump(X, open("X.pickle", "wb"))
#pickle.dump(Y, open("Y.pickle", "wb"))



