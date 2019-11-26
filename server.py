from flask import jsonify
import numpy as np
from flask import Flask, request
from flask_cors import CORS
import cv2
from pytesseract import Output
import matplotlib.patches as patches
from preprocessing import four_point_transform, edge_detection
import json
import pytesseract
import matplotlib.pyplot as plt


# creates one dim array from points dictionary
def transform_to_1d(points_as_dict):
    one_dim_coordinates = []
    for point in points_as_dict:
        one_dim_coordinates.append(point["x"])
        one_dim_coordinates.append(point["y"])
    return one_dim_coordinates


if __name__ == "__main__":
    app = Flask(__name__)
    CORS(app)

    @app.route("/trainable", methods=["POST"])
    def trainable():
        # retrieve parameters (in this case coordinates) from request
        data = dict(request.form)
        coordinates = json.loads(data["coordinates"])

        # Decode image which was send from flutter with multipart form data
        img = getImageFromRequest()
        # four point transformation on the picture with the give coordinates
        cropped_img = four_point_transform(transform_to_1d(coordinates), img)
        resized_img = cv2.resize(cropped_img, None, fx=2, fy=2)
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
        gray_img = cv2.medianBlur(gray_img, 3)

        # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_thresholding.html
        # https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/

        # get text from tesseract ocr engine
        lines = pytesseract.image_to_data(gray_img, lang="deu", output_type=Output.DICT)
        boxes_and_text = []

        draw_boxes(lines, gray_img)
        return jsonify(tesseract_to_json(lines))

    @app.route("/prediction", methods=["POST"])
    def prediction():
        # Decode image which was send from flutter with multipart form data
        img = getImageFromRequest()

        # four point transformation on the picture with the give coordinates
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, b_w_image = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cropped_img, points = edge_detection(b_w_image)

        # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_thresholding.html
        # https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/

        # get text from tesseract ocr engine
        lines = pytesseract.image_to_data(gray_img, lang="deu", output_type=Output.DICT)
        boxes_and_text = []

        draw_boxes(lines, gray_img)
        return jsonify(tesseract_to_json(lines))


    def getImageFromRequest():
        return cv2.imdecode(np.frombuffer(request.files["bill"].read(), np.uint8), cv2.IMREAD_COLOR)

    # Draw bound boxes which were detected by tesseract
    def draw_boxes(lines: dict, img):
        fig, ax = plt.subplots(1)
        plt.imshow(img, cmap="gray")
        for i in range(len(lines["conf"])):
            rect = patches.Rectangle((lines["left"][i], lines["top"][i]), lines["width"][i], lines["height"][i], linewidth=1, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
        plt.show()

    def tesseract_to_json(lines):
        detections = []
        for i in range(len(lines["conf"])):
            confidence = float(lines["conf"][i]) if int(lines["conf"][i]) != -1 else 0.0
            detections.append({
                "x": lines["left"][i],
                "y": lines["top"][i],
                "width": lines["width"][i],
                "height": lines["height"][i],
                "text": lines["text"][i],
                "confidence": confidence / 100
            })
        return {"detections": detections}

    app.run(debug=True, port=5000)



