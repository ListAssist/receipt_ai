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
import imgaug.augmenters as iaa
import imgaug as ia

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

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
    for i, vertice in enumerate(vertices):
        if i % 2 == 0:
            output_vertices.append(vertice * ratio[0])
        else:
            output_vertices.append(vertice * ratio[1])
    return output_vertices


def vertice_chunks(vertices):
    output_vertices = []

    for vertice in vertices:
        output_vertices.append((vertice["x"], vertice["y"]))
    return output_vertices


# create chunks
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# normalize vertices
def normalize_vertices(vertices):
    return [vertice / (RES_X if index % 2 == 0 else RES_Y) for index, vertice in enumerate(vertices)]


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
    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-30, 30),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
        ],
        random_order=True
    )

    with open("label_export.json", "r") as export_file:
        data = json.load(export_file)

        # remove skipped labels
        data = [label for label in data if label["Label"] != "Skip"]

        original_images = []
        polygons = []
        for label in tqdm(data):
            # CONSTANTS
            IMG_NAME = label["External ID"]
            IMPORTANT_VERTICES = label["Label"]["important"][0]["geometry"]
            # BILL_VERTICES = label["Label"]["bill"][0]["geometry"]

            raw_img = cv2.imread(os.path.join(FULL_PATH, IMG_NAME), cv2.IMREAD_GRAYSCALE)

            # load every picture as grayscale and create dictionary
            original_images.append(raw_img)
            poly = ia.Polygon(vertice_chunks(IMPORTANT_VERTICES))
            polygons.append(poly)

        images, polygons = seq(images=original_images, polygons=polygons)
        for i, img in enumerate(images):
            img_vertices = []
            for idx in range(len(polygons[i].xx)):
                img_vertices.append(polygons[i].xx[idx])
                img_vertices.append(polygons[i].yy[idx])

            # resize image to fixed resolution
            resized_img = cv2.resize(img, (RES_X, RES_Y))

            # calculate ratios to transform labeled coordinates to resized image
            RATIO_X = RES_X / raw_img.shape[1]
            RATIO_Y = RES_Y / raw_img.shape[0]
            img_vertices = transform_vertices(img_vertices, (RATIO_X, RATIO_Y))

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
            print(img.shape)
            # Show polygons (coordinates must not be squashed)
            if DEBUG:
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow("image", 600, 600)
                cv2.imshow("image", resized_img)
                cv2.waitKey(0)

                skew_algorithm(binary_img)
                # plt.imshow(four_point_transform(img_vertices, resized_img), cmap="gray")
                plt.imshow(binary_img, cmap="gray")
                plt.show()

            training_output_vertices = normalize_vertices(img_vertices)
            training_data.append([binary_img, training_output_vertices])

    X = []
    Y = []

    # crete sperate arrays for features and labels
    for x, y in training_data:
        X.append(x)
        Y.append(y)

    X = np.array(X).reshape((-1, RES_X, RES_Y, 1))
    Y = np.array(Y)

    # normalize pixel values (gray scale!)
    X = X / 255

    pickle.dump(X, open("pickles/X.pickle", "wb"))
    pickle.dump(Y, open("pickles/Y.pickle", "wb"))
