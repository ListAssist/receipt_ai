import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import imutils
import json
import pickle
from skimage.filters import threshold_local
from tqdm import tqdm

# path to images of bills
BILL_IMG_DIR = "bills"
FULL_PATH = os.path.join(os.getcwd(), BILL_IMG_DIR)

# fixed resolution which is put into model
# RATIO should be 4:3 since most mobile cams use this aspect ratio
RES_Y = 400
RES_X = 300

training_data = []


# gets 4 vertices as dictionary with x and y properties
def transformVertices(vertices, ratio):
    output_vertices = []
    for vertice in vertices:
        output_vertices.append((vertice["x"] * ratio[0]) / RES_X)
        output_vertices.append((vertice["y"] * ratio[1]) / RES_Y)
    return output_vertices

name_to_img = {}
for img in tqdm(os.listdir(FULL_PATH)):
    name_to_img[img] = cv2.imread(os.path.join(FULL_PATH, img), cv2.IMREAD_GRAYSCALE)

with open("labelbox_export.json", "r") as export_file:
    data = json.load(export_file)
    data = [label for label in data if label["Label"] != "Skip"]
    for label in data:
        IMG_NAME = label["External ID"]
        IMPORTANT_VERTICES = label["Label"]["bill"][0]["geometry"]
        BILL_VERTICES = label["Label"]["bill"][0]["geometry"]

        # read image pixels in 2d array as grayscale values
        img = name_to_img[IMG_NAME]

        # calculate ratios to transform labeled coordinates to resized image
        RATIO_X = RES_X / img.shape[1]
        RATIO_Y = RES_Y / img.shape[0]

        # transform labeled vertices to resized image
        training_output_vertices = transformVertices(BILL_VERTICES, (RATIO_X, RATIO_Y))

        # resize image to fixed resolution
        resized_img = cv2.resize(img, (RES_X, RES_Y))

        # create binary picture (just b&w)
        # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
        # binary_img = cv2 \
        #   .adaptiveThreshold(resized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 5)
        threshold = threshold_local(resized_img, 11, offset=10, method="gaussian")
        binary_img = resized_img > threshold

        # print image in plot
        # plt.imshow(binary_img, cmap="gray")
        # plt.show()

        training_data.append([resized_img, training_output_vertices])

X = []
Y = []

# crete sperate arrays for features and labels
for x, y in training_data:
    X.append(x)
    Y.append(y)

X = np.array(X).reshape((-1, RES_X, RES_Y, 1))

pickle.dump(X, open("pickles/X.pickle", "wb"))
pickle.dump(Y, open("pickles/Y.pickle", "wb"))

# Algorithm to skew image correctly
# Only works on well photographed pictures of receipts
def skewAlgorithm(image):
    # edge detection
    sigma = 0.33
    # get average grayscale value
    median = np.median(image)
    # create limit for aperture size (kernel size)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    # blur image to focus on edges and shapes
    img_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # detect edges with the Canny86 Algorithm
    img_blurred = cv2.Canny(img_blurred, lower, upper)
    # cv2.imshow("Blurred img", img_blurred)
    # cv2.waitKey(0)

    # find contours
    contours = cv2.findContours(img_blurred.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approximation) == 4:
            cv2.drawContours(image, [approximation], -1, (0, 255, 0), 2)
            cv2.imshow("Outline", image)
            cv2.waitKey(0)
            break
