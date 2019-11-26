from math import atan, atan2
from flask import jsonify, Response
import numpy as np
from flask import Flask, request
import cv2
from pytesseract import Output
import matplotlib.patches as patches
from preprocessing import four_point_transform, edge_detection, order_points
import json
import pytesseract
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
app = Flask(__name__)


@app.route("/trainable", methods=["POST"])
def trainable():
    # retrieve parameters (in this case coordinates) from request
    coordinates = json.loads(request.form["coordinates"])

    # Decode image which was send from flutter with multipart form data
    img = get_image_from_request()
    # four point transformation on the picture with the give coordinates
    cropped_img, _ = four_point_transform(transform_to_1d(coordinates), img)
    resized_img = cv2.resize(cropped_img, None, fx=2, fy=2)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.medianBlur(gray_img, 3)

    # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_thresholding.html
    # https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/

    # get text from tesseract ocr engine
    lines = pytesseract.image_to_data(gray_img, lang="deu", config="--psm 6",  output_type=Output.DICT)
    boxes_and_text = []

    draw_boxes(lines, gray_img)
    return jsonify(tesseract_to_json(lines))


@app.route("/prediction", methods=["POST"])
def prediction():
    """
    Important part is to first do otsu thresholding and THEN detect with canny since otsu optimized for such tasks
    """

    # Decode image which was send from flutter with multipart form data
    img = get_image_from_request()
    # four point transformation on the picture with the give coordinates
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, b_w_image = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # detect bill and use both approaches
    points = edge_detection(b_w_image, type="approx")
    if points is None:
        print("bbox executed")
        points = edge_detection(b_w_image, type="bbox")

    # get gray image which was cropped for tesseract
    if points is None:
        cropped_gray_img = gray_img
    else:
        print(points.flatten())
        cropped_gray_img = four_point_transform(points.flatten(), gray_img)

    cv2.namedWindow("otsu_result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("otsu_result", 600, 600)
    cv2.imshow("otsu_result", cropped_gray_img)
    cv2.waitKey(0)

    # detect horizontal billa lines
    rect_points = detect_lines(cropped_gray_img)

    if rect_points is not None:
        important_area = four_point_transform(order_points(rect_points).flatten(), cropped_gray_img)
        cv2.namedWindow("main", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("main", 600, 600)
        cv2.imshow("main", important_area)
    else:
        important_area = cropped_gray_img

    # get text from tesseract ocr engine
    tesseract_output = pytesseract.image_to_string(important_area, lang="deu", config="--psm 6")
    if not tesseract_output:
        # check if rect was empty, if not try again with the whole image
        if rect_points is not None:
            important_area = cropped_gray_img
            tesseract_output = pytesseract.image_to_string(important_area, lang="deu", config="--psm 6")
            print(tesseract_output)
            # if the whole image is not helpful, error
            if tesseract_output is None:
                return jsonify({"error": "no text detected"}), 400

            # There is text to be returned
            lines = pytesseract.image_to_data(important_area, lang="deu", config="--psm 6", output_type=Output.DICT)
            draw_boxes(lines, important_area)
            return jsonify(tesseract_to_json(lines)), 200
        else:
            # rect points was really empty and the whole image didn't have any text
            return jsonify({"error": "no text detected"}), 400
    else:
        # rect_points returned good points and text can be returned
        lines = pytesseract.image_to_data(important_area, lang="deu", config="--psm 6", output_type=Output.DICT)
        draw_boxes(lines, important_area)
        return jsonify(tesseract_to_json(lines)), 200


def detect_lines(gray_img):
    # test thresholds to see which one is fit the best
    # fig, ax = try_all_threshold(gray_img, figsize=(10, 8), verbose=False)
    # plt.show()
    b_w_edges = cv2.Canny(gray_img, 0, 255)

    # Detect points that form a line
    # threshold = how many points until it is recognized as line
    lines = cv2.HoughLinesP(b_w_edges, 1, np.pi / 180, threshold=200, minLineLength=10, maxLineGap=250)
    # Draw lines on the image
    if lines is None or len(lines) < 2:
        return

    bounding_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # calculate angle to check if its a horizontal line https://i.imgur.com/fCw3PHC.png
        # tan a = GK / AK
        x_diff = abs(x2 - x1)

        # to stay on the save side in case x_diff is 0
        if x_diff == 0:
            x_diff = 1

        # get angle to check if it can go through as a horizontal line
        angle = atan(abs(y2 - y1) / x_diff) * 180.0 / np.pi

        if abs(angle) < 5:
            # add line to array with coords
            # [(x1, y1),(x2, y2)]
            # [(x1, y1), (x2, y2)]
            avg_y = (y1 + y2) / 2
            bounding_lines.append(((x1, y1, x2, y2), avg_y))

            plt.plot((x1, x2), (y1, y2), "k-")
    # (x1, y1, x2, y2) | avg_y
    bounding_lines = sorted(bounding_lines, key=lambda line_props: line_props[1], reverse=False)[:2]

    # create structured array with the points of each line end
    rect_points = []
    xleft_sum = 0
    xright_sum = 0
    for line in bounding_lines:
        x1, y1, x2, y2 = line[0]
        if x2 > x1:
            rect_points.append([x1, y1])
            rect_points.append([x2, y2])
            xright_sum += x2
            xleft_sum += x1
        else:
            rect_points.append([x2, y2])
            rect_points.append([x1, y1])
            xright_sum += x1
            xleft_sum += x2

    # calculate average x coordinates on the right and left side
    xleft_avg = xleft_sum / 2
    xright_avg = xright_sum / 2

    # set calculated mean values
    for i, point in enumerate(rect_points):
        point[0] = xleft_avg if i % 2 == 0 else xright_avg
        plt.scatter(point[0], point[1])

    plt.imshow(gray_img, cmap="gray")
    # plt.show()
    # Show result
    # cv2.namedWindow("main", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("main", 600, 600)
    # cv2.imshow("main", gray_img)
    # cv2.waitKey(0)

    return np.array(rect_points)


def get_image_from_request():
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


# creates one dim array from points dictionary
def transform_to_1d(points_as_dict):
    one_dim_coordinates = []
    for point in points_as_dict:
        one_dim_coordinates.append(point["x"])
        one_dim_coordinates.append(point["y"])
    return one_dim_coordinates


if __name__ == "__main__":
    app.run(debug=True, port=5000)



