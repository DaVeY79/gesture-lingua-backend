from flask import Flask, jsonify, request, render_template
from os.path import join
import logging
from datasets.sign_language import path_to_tensor
from keras.models import load_model
import tensorflow as tf
import numpy as np
np.random.seed(5)
tf.set_random_seed(2)
#import random
app = Flask(__name__)

# Enabling logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@app.route("/", methods=["GET"])
def upload_page():
    return render_template("linguahome.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.form["submit_button"] == "Predict the Alphabet":
        image_labels = ['A', 'B', 'C']
        image_path = join('uploaded_images', 'image.jpg')
        # create a file handler
        handler = logging.FileHandler('Log_Model_Builder.log')
        handler.setLevel(logging.DEBUG)

        # create a logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)

        file = request.files['file']
        file.save(image_path)
        model = load_model('model.h5')
        X = path_to_tensor(image_path, 50)

        y_probs = model.predict(X)
        y_preds = np.argmax(y_probs, axis=1)

        # return jsonify({'alphabet' : image_labels[y_preds[0]]})
        return render_template("linguahome.html", name=image_labels[y_preds[0]])
        # return render_template("success.html",alphabet = file)

    elif request.form["submit_button"] == "Capture Video":
        return render_template("success.html")


app.run(port=5000)
