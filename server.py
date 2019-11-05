import numpy as np
from flask import Flask, request
from flask_cors import CORS
import cv2
from flask import Response
from skimage.filters import try_all_threshold, threshold_yen
from skimage.filters.rank import threshold

from preprocessing import calculate_distance, four_point_transform
import matplotlib.pyplot as plt
import json
import pytesseract

app = Flask(__name__)
CORS(app)


# creates one dim array from points dictionary
def transform_to_1d(points_as_dict):
    one_dim_coordinates = []
    for point in points_as_dict:
        one_dim_coordinates.append(point["x"])
        one_dim_coordinates.append(point["y"])
    return one_dim_coordinates


@app.route('/', methods=["POST"])
def main():
    # retrieve parameters (in this case coordinates) from request
    data = dict(request.form)
    coordinates = json.loads(data["coordinates"][0])

    # Decode image which was send from flutter with multipart form data
    img = cv2.imdecode(np.frombuffer(request.files["bill"].read(), np.uint8), cv2.IMREAD_COLOR)
    # four point transformation on the picture with the give coordinates
    cropped_img = four_point_transform(transform_to_1d(coordinates), img)
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)

    # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_thresholding.html
    # https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/
    thresh = threshold_yen(cropped_img)
    binary_img = cropped_img > thresh
    binary_img = binary_img * 255

    print(binary_img)
    # get text from tesseract ocr engine
    lines = pytesseract.image_to_string(binary_img, lang="deu")

    print(lines)
    plt.imshow(binary_img, cmap="gray")
    plt.show()
    return Response(status=201)


app.run(debug=True, port=5000)



