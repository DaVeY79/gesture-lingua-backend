import numpy as np
np.random.seed(5)
import tensorflow as tf
tf.set_random_seed(2)
from keras.models import load_model
from datasets.sign_language import path_to_tensor
import logging
import random
from os.path import join
from flask import Flask, jsonify, request, render_template
app = Flask(__name__)

### Enabling logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route("/", methods = ["GET"])
def upload_page():
    return render_template("linguahome.html")

@app.route('/predict',methods = ['GET','POST'])
def predict():
    image_labels = ['A','B','C']
    image_path = join('uploaded_images','image.jpg')
    # create a file handler
    handler = logging.FileHandler('Log_Model_Builder.log')
    handler.setLevel(logging.DEBUG)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    file = request.files['file']
    file.save(image_path)
    model = load_model('model.h5')
    X = path_to_tensor(image_path,50)

    y_probs = model.predict(X)
    y_preds = np.argmax(y_probs,axis = 1)

    return jsonify({'alphabet' : image_labels[y_preds[0]]})
    #return render_template("linguahome.html",alphabet = image_labels[y_preds[0]])
    #return render_template("success.html",alphabet = file)

app.run(port = 5000)
