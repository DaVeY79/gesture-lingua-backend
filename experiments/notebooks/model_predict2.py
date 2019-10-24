from fastai.vision import ImageDataBunch, get_transforms, cnn_learner, normalize, imagenet_stats, open_image, models
import numpy as np
np.random.seed(5)
import torch
import pickle
from pathlib import Path
import logging
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
    if request.form["submit_button"] == "Predict the Alphabet":

        path = Path("/Users/davidabraham/gesture-lingua-backend/experiments/notebooks")

        with open(path/"class_names.pkl","rb") as pkl_file:
            classes = pickle.load(pkl_file)

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
        file.save(str(path/image_path))

        data2 = ImageDataBunch.single_from_classes(path,classes,ds_tfms= get_transforms(),size = 224).normalize(imagenet_stats)
        learn = cnn_learner(data2,models.resnet34)
        learn.load('model-after-unfreeze')

        img = open_image(path/image_path)
        label,index, pred = learn.predict(img)

        return render_template("linguavideo.html",name = label,image = image_path)

    elif request.form["submit_button"] == "Capture Video":
        return render_template("success.html")

app.run(port = 5000)
