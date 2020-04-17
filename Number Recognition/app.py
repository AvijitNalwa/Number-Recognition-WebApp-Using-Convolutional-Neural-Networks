# importing
from tensorflow.keras.models import load_model
import flask
from flask import Flask, render_template, url_for, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import re
import io
import base64

# initializing
app = flask.Flask(__name__)


def process_input(image_url):
    img_size = 28, 28
    image_string = re.search(r'base64,(.*)', image_url).group(1)
    image_bytes = io.BytesIO(base64.b64decode(image_string))
    image = Image.open(image_bytes)
    image = image.resize(img_size, Image.ANTIALIAS)
    image = image.convert('1')
    image_array = np.asarray(image)
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array


def result(processed_img, model_path):
    model = load_model(model_path)
    graph = tf.compat.v1.get_default_graph()
    preds = model.predict(processed_img)
    guess = str(preds.argmax())
    return jsonify(guess=guess)


@app.route('/', methods=['GET', 'POST'])
def get_image():
    guess = 0
    if request.method == 'POST':
        image_url = request.values['imageBase64']
        processed_img = process_input(image_url)
        model_path = 'mnist_4-12_12(aug).h5'
        guess = result(processed_img, model_path)
        return guess

    return render_template('index.html', guess=guess)


if __name__ == '__main__':
    app.run(debug=True)
