from flask import Flask, request
from flask_cors import CORS
from werkzeug.datastructures import ImmutableMultiDict
import cv2
from flask import Response

app = Flask(__name__)
CORS(app)


@app.route('/', methods=["POST"])
def main():
    data = dict(request.form)
    cv2.imshow("king", data["bild"])
    return Response(status=201)


app.run(Debug=True, port=5000)
