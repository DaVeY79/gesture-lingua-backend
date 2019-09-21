import numpy as np
np.random.seed(5)
import tensorflow as tf
tf.set_random_seed(2)
from keras.models import load_model
from datasets.sign_language import path_to_tensor, path_to_tensor_new
import logging
import random
from os.path import join
from flask import Flask, jsonify,request, render_template
from werkzeug.datastructures import FileStorage
app = Flask(__name__)

### Enabling logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route("/")
def upload_page():
    return render_template("linguahome.html")

@app.route('/predict',methods = ['GET','POST'])
def predict():
    image_labels = ['A','B','C']
    image_path = "/Users/davidabraham/gesture-lingua-backend/experiments/notebooks/"
    # create a file handler
    handler = logging.FileHandler('Log_Model_Builder.log')
    handler.setLevel(logging.DEBUG)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    #
    # # add the handlers to the logger
    logger.addHandler(handler)

    #content = request.get_json()
    #image_name = content.get("image_name")
    #filename = join(image_path,image_name)

    file = request.files['file']
    file.save(join('uploaded_images','image.jpg'))
    model = load_model('model.h5')
    X = path_to_tensor(join('uploaded_images','image.jpg'),50)

    y_probs = model.predict(X)
    y_preds = np.argmax(y_probs,axis = 1)

    #return image_labels[y_preds[0]]
    #return jsonify({"Alphabet" : image_labels[y_preds[0]]})
    #logger.info("Alphabet : {}".format(image_labels[y_preds[0]]))
    #return 'file uploaded successfully {}'.format(file.filename)
    return render_template("success.html",alphabet = image_labels[y_preds[0]])
    #return render_template("success.html",alphabet = file)

app.run(port = 5000)
