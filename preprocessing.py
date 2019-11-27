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
from scipy.spatial.distance import cdist

from HED import CropLayer

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


# https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[3] = pts[np.argmin(diff)]
    rect[1] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


# Algorithm to skew image correctly
# Only works on well photographed pictures of receipts
def edge_detection(b_w_image, type='approx'):
    # find contours
    contours = cv2.findContours(b_w_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    print(type)
    if DEBUG:
        cv2.namedWindow("Image from contour detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image from contour detection", 600, 600)
        cv2.imshow("Image from contour detection", b_w_image)
        cv2.waitKey(0)

    # calculate a min area for the bill
    res_y = len(b_w_image)
    res_x = len(b_w_image[0])
    min_area = res_x * res_y / 100

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            break

        if type == 'approx':
            peri = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.02 * peri, True)
        else:
            bbox = cv2.minAreaRect(contour)
            approximation = np.array(cv2.boxPoints(bbox))

        color = cv2.cvtColor(b_w_image, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(color, [contour], -1, (255, 0, 0), 5)
        cv2.imshow("Image from contour detection", color)
        cv2.waitKey(0)

        if len(approximation) == 4:
            # reshape approximation for better coding experience
            approximation = np.array(approximation).reshape((4, 2))

            # order points to tl, bl, br, tr
            points = order_points(approximation)
            if type == "bbox":
                for point in points:
                    if point[0] < 0:
                        point[0] = 0
                    elif point[0] > res_x:
                        point[0] = res_x

                    if point[1] < 0:
                        point[1] = 0
                    elif point[1] > res_y:
                        point[1] = res_y
            return points


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
                mode=ia.ALL  # use any of scikit-image"s warping modes (see 2nd image from the top for examples)
            )),
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),  # sometimes move parts of the image around
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
        ],
        random_order=True
    )

    with open("exports/training_data.json", "r") as export_file:
        data = json.load(export_file)

        # remove skipped labels
        data = [label for label in data if label["Label"] != "Skip"]

        images = []
        polygons = []
        for label in tqdm(data):
            # CONSTANTS
            IMG_NAME = label["External ID"]
            IMPORTANT_VERTICES = label["Label"]["important"][0]["geometry"]
            # BILL_VERTICES = label["Label"]["bill"][0]["geometry"]

            raw_img = cv2.imread(os.path.join(FULL_PATH, IMG_NAME), cv2.IMREAD_GRAYSCALE)

            # load every picture as grayscale and create dictionary
            images.append(raw_img)
            poly = ia.Polygon(vertice_chunks(IMPORTANT_VERTICES))
            polygons.append([poly])

        # images, polygons = seq(images=images, polygons=polygons)
        for i, img in enumerate(images):
            img_vertices = []
            for idx in range(len(polygons[i][0].xx)):
                img_vertices.append(polygons[i][0].xx[idx])
                img_vertices.append(polygons[i][0].yy[idx])

            # resize image to fixed resolution
            resized_img = cv2.resize(img, (RES_X, RES_Y))

            # calculate ratios to transform labeled coordinates to resized image
            RATIO_X = RES_X / raw_img.shape[1]
            RATIO_Y = RES_Y / raw_img.shape[0]
            img_vertices = transform_vertices(img_vertices, (RATIO_X, RATIO_Y))

            """ create binary picture (just b&w)
            https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
            binary_img = cv2 \
                .adaptiveThreshold(resized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 5)
            
            Otsu threshold for multi background picture
            Adaptive (mean, gaussian) for statistics over local threshold filter
                https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_thresholding.html
                https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/
            https://dsp.stackexchange.com/questions/2411/what-are-the-most-common-algorithms-for-adaptive-thresholding
            """
            ret, b_w_image = cv2.threshold(resized_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Show polygons (coordinates must not be squashed)
            if DEBUG:
                edge_detection(b_w_image)
                # plt.imshow(four_point_transform(img_vertices, resized_img), cmap="gray")
                # plt.imshow(thresh, cmap="gray")
                # plt.show()

            training_output_vertices = normalize_vertices(img_vertices)
            training_data.append([b_w_image, training_output_vertices])

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
