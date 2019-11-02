import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from skimage.filters import threshold_local
from tensorflow.keras.models import load_model
from preprocessing import RES_X, RES_Y
from tqdm import tqdm

# path to images of test bills
BILL_IMG_DIR = "test_bills"
FULL_PATH = os.path.join(os.getcwd(), BILL_IMG_DIR)


DEBUG = True
model = load_model(os.path.join("model/main.h5"))


for img in tqdm(os.listdir(FULL_PATH)):
    # read image pixels in 2d array as grayscale values
    img = cv2.imread(os.path.join(FULL_PATH, img), cv2.IMREAD_GRAYSCALE)

    # calculate ratios to transform labeled coordinates to resized image
    RATIO_X = RES_X / img.shape[1]
    RATIO_Y = RES_Y / img.shape[0]

    # resize image to fixed resolution
    resized_img = cv2.resize(img, (RES_X, RES_Y))

    # create binary picture (just b&w)
    # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
    # binary_img = cv2 \
    #   .adaptiveThreshold(resized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 5)
    threshold = threshold_local(resized_img, 19, offset=10, method="gaussian")
    binary_img = resized_img > threshold

    if DEBUG:
        # print image in plot
        plt.imshow(binary_img, cmap="gray")
        X = np.array(binary_img).reshape((-1, RES_X, RES_Y, 1)) / 255

        coordinates = list(model.predict(X))[0]
        chunked_list = []
        for i in range(len(coordinates)):
            if i % 2 == 0:
                coordinates[i] *= RES_X
            else:
                coordinates[i] *= RES_Y
                chunked_list.append([coordinates[i - 1], coordinates[i]])
        polygon_predicted = patches.Polygon(chunked_list, linewidth=1, edgecolor="r",
                                  facecolor="none")
        plt.gca().add_patch(polygon_predicted)
        plt.show()
