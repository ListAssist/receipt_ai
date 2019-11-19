import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import imutils
import json
import pickle
from tabulate import tabulate
from matplotlib import patches
from skimage.filters import threshold_otsu
from tqdm import tqdm


# if true will print plots and information about preprocessing
DEBUG = True

# path to images of bills
BILL_IMG_DIR = "bills"
FULL_PATH = os.path.join(os.getcwd(), BILL_IMG_DIR)

# fixed resolution which is put into model
# RATIO should be 4:3 since most mobile cams use this aspect ratio
RES_X = 600
RES_Y = 900

training_data = []


# gets 4 vertices as dictionary with x and y properties
def transform_vertices(vertices, ratio):
    output_vertices = []
    # get coordinates for fixed size picture and squash them between 0 and 1
    for vertice in vertices:
        output_vertices.append(vertice["x"] * ratio[0] / (1 if DEBUG else RES_X))
        output_vertices.append(vertice["y"] * ratio[1] / (1 if DEBUG else RES_Y))
    return output_vertices


# distance between two points with pythagoras
def calculate_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def four_point_transform(points, image):
    top_left = (points[0], points[1])
    bottom_left = (points[2], points[3])
    bottom_right = (points[4], points[5])
    top_right = (points[6], points[7])

    # Take largest width as final width of cutout image
    width_top = calculate_distance(top_left, top_right)
    width_bottom = calculate_distance(bottom_left, bottom_right)
    max_width = int(max((width_top, width_bottom)))

    height_left = calculate_distance(top_left, bottom_left)
    height_right = calculate_distance(top_right, bottom_right)
    max_height = int(max((height_left, height_right)))

    dst = np.array([
        [0, 0],
        [0, max_height],
        [max_width, max_height],
        [max_width, 0]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.array((top_left, bottom_left, bottom_right, top_right), dtype="float32"), dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


# Algorithm to skew image correctly
# Only works on well photographed pictures of receipts
def skew_algorithm(image):
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


if __name__ == "__main__":
    # load every picture as grayscale and create dictionary
    name_to_img = {}
    for img in tqdm(os.listdir(FULL_PATH)):
        name_to_img[img] = cv2.imread(os.path.join(FULL_PATH, img), cv2.IMREAD_GRAYSCALE)

    # open exported json and read labels
    with open("labelbox_export.json", "r") as export_file:
        data = json.load(export_file)

        # remove skipped labels
        data = [label for label in data if label["Label"] != "Skip"]
        for label in data:
            # extra important data
            IMG_NAME = label["External ID"]
            IMPORTANT_VERTICES = label["Label"]["important"][0]["geometry"]
            #### BILL_VERTICES = label["Label"]["bill"][0]["geometry"]

            # read image pixels in 2d array as grayscale values
            img = name_to_img[IMG_NAME]

            # calculate ratios to transform labeled coordinates to resized image
            RATIO_X = RES_X / img.shape[1]
            RATIO_Y = RES_Y / img.shape[0]

            # transform labeled vertices to resized image
            training_output_vertices = transform_vertices(IMPORTANT_VERTICES, (RATIO_X, RATIO_Y))

            # resize image to fixed resolution
            resized_img = cv2.resize(img, (RES_X, RES_Y))

            ''' create binary picture (just b&w)
            https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
            binary_img = cv2 \
                .adaptiveThreshold(resized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 5)
            
            Otsu threshold for multi background picture
            Adaptive (mean, gaussian) for statistics over local threshold filter
            
            https://dsp.stackexchange.com/questions/2411/what-are-the-most-common-algorithms-for-adaptive-thresholding
            '''
            threshold = threshold_otsu(cv2.GaussianBlur(resized_img, (5, 5), 0))
            binary_img = resized_img > threshold

            # Show polygons (coordinates must not be squashed)
            if DEBUG:
                cv2.imshow("transformed", four_point_transform(training_output_vertices, resized_img))

            training_data.append([binary_img, training_output_vertices])

    X = []
    Y = []

    # crete sperate arrays for features and labels
    for x, y in training_data:
        X.append(x)
        Y.append(y)

    print(tabulate(training_data))
    X = np.array(X).reshape((-1, RES_X, RES_Y, 1))
    Y = np.array(Y)

    # normalize pixel values (gray scale!)
    X = X / 255

    pickle.dump(X, open("pickles/X.pickle", "wb"))
    pickle.dump(Y, open("pickles/Y.pickle", "wb"))
