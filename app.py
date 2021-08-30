#  -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation.  All rights reserved.
#  -------------------------------------------------------------
import os
import io
import re
import base64

from PIL import Image
from flask import Flask, request
from flask import jsonify
import requests

from tf_model_helper import TFModel

app = Flask(__name__)

# Path to signature.json and model file
ASSETS_PATH = os.path.join(".", "./model")
TF_MODEL = TFModel(ASSETS_PATH)

@app.errorhandler(404) 
def invalid_route(e): 
    return jsonify({'errorCode' : 404, 'message' : 'Route not found'})

@app.route('/predict', methods=["POST"])
def predict_image():
    req = request.get_json(force=True)
    image = _process_base64(req)
    return TF_MODEL.predict(image)

@app.route('/predict-from-url', methods=["POST"])
def predit_from_url():
    req = request.get_json(force=True)
    image = _process_base64_from_url(req)
    json_response = TF_MODEL.predict(image)
    return jsonify(json_response)
    #response = Response(json_response["predictions"],content_type="application/json; charset=utf-8" )
    #return response


def _process_base64(json_data):
    image_data = json_data.get("image")
    image_data = re.sub(r"^data:image/.+;base64,", "", image_data)
    image_base64 = bytearray(image_data, "utf8")
    image = base64.decodebytes(image_base64)
    return Image.open(io.BytesIO(image))

def _process_base64_from_url(json_data):
    image_url = json_data.get("url")
    image_content = requests.get(image_url, stream=True).raw
    image = Image.open(image_content)
    return image


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=PORT, debug=True)
