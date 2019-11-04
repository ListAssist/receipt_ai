import numpy as np
from flask import Flask, request
from flask_cors import CORS
import cv2
from flask import Response
from werkzeug.datastructures import ImmutableMultiDict
from preprocessing import calculate_distance, four_point_transform
import json

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
    img = cv2.imdecode(np.frombuffer(request.files["bill"].read(), np.uint8), cv2.IMREAD_UNCHANGED)

    cv2.imshow("king", four_point_transform(transform_to_1d(coordinates), img))
    cv2.waitKey(0)
    return Response(status=201)


app.run(debug=True, port=5000)



