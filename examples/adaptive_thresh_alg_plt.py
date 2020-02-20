from time import sleep

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Load image from disk
img = cv2.imread("images/rechnung.jpeg", cv2.IMREAD_GRAYSCALE)
# transforms image into black and white
# Formulas taken from https://en.wikipedia.org/wiki/Summed-area_table
def adaptive_mean_thresh(input_img, blockSize, C):
    height, width = input_img.shape
    integral = gen_integral_img(input_img)

    # calculate boxes
    output_image = np.zeros_like(input_img)
    margin = int((blockSize - 1) / 2)
    for x in range(margin, width-margin):
        for y in range(margin, height-margin):
            pixel = input_img[y, x]
            thresh = get_threshold_for_area(input_img, integral, (x, y), blockSize)
            thresh -= C
            if pixel >= thresh:
                output_image[y, x] = 255
            else:
                output_image[y, x] = 0

    return output_image

def get_threshold_for_area(image, I, pos: tuple, b: int):
    h, w = image.shape
    x, y = pos
    db = int((b - 1) / 2)

    if 0 + db <= x <= w - db or 0 + db <= y <= h - db:
        #                  I(D)       +        I(A)     +     I(B)        -       I(C)
        pixel_sum = I[y + db, x + db] + I[y - db, x - db] - I[y - db, x + db] - I[y + db, x - db]
        return pixel_sum / b**2

# generates integral image from input image
def gen_integral_img(image):
    height, width = image.shape

    integral = np.zeros_like(image, dtype=int)
    # create first x and y row so faster algorithm can come to use
    for col in range(width):
        integral[0, col] = image[0, 0:col].sum()
    for row in range(height):
        integral[row, 0] = image[0:row, 0].sum()

    # use fast formula: I(x,y) = i(x,y) + I(x, y-1) + I(x-1, y) - I(x-1, y-1)
    for col in range(1, width):
        for row in range(1, height):
            integral[row, col] = image[row, col] + integral[row - 1, col] + integral[row, col - 1] - integral[row - 1, col - 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cols = np.arange(width)
    rows = np.arange(height)
    xx, yy = np.meshgrid(cols, rows)

    def kong(x, y):
        return integral[y,x]
    z = kong(xx, yy)
    surface = ax.plot_surface(xx, yy, z, cmap=cm.seismic, antialiased=False)
    plt.xlabel("Höhe")
    plt.ylabel("Breite")
    fig.colorbar(surface, shrink=0.5, aspect=5)

    plt.title("Integral Bild visualisiert")
    plt.show()
    return integral


ad = adaptive_mean_thresh(img, 9, 4)
plt.imshow(ad, cmap="gray")
plt.show()


