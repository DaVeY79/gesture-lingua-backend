import numpy as np
import threading
import argparse
import datetime
import time
import logging
import pickle
import torch
import cv2
from os.path import join
from pathlib import Path
from fastai.vision import (
    ImageDataBunch,
    get_transforms,
    cnn_learner,
    imagenet_stats,
    open_image,
    models,
)
from flask import Flask, Response, jsonify, request, render_template, redirect, url_for

# from fastai.vision import *

app = Flask(__name__)

# Enabling logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

path = Path("/Users/davidabraham/gesture-lingua-backend/experiments/notebooks")

with open(path / "class_names.pkl", "rb") as pkl_file:
    classes = pickle.load(pkl_file)

data2 = ImageDataBunch.single_from_classes(
    path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = cnn_learner(data2, models.resnet34)
learn.load('model-after-unfreeze')

outputFrame = None
lock = threading.Lock()
cap = cv2.VideoCapture()

@app.route("/", methods=["GET"])
def upload_page():
    cap.release()
    return render_template("linguahome.html")


@app.route("/video", methods=["GET"])
def video():
    # return the rendered template for video
    return render_template("linguavideo2.html")

# @app.route("/release", methods=["GET"])
# def video():
#     cap.release()
#     return render_template("linguahome.html")


def recognize_gesture(cap, frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock

    ret, score = '', 0.0
    i = 0
    mem = ''
    consecutive = 0
    sequence = ''

    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        res = ''
        if ret:
            x1, y1, x2, y2 = 100, 100, 700, 700
            img_cropped = img[y1:y2, x1:x2]

            # c += 1
            cv2.imwrite('test1.jpg', img_cropped)

            a = cv2.waitKey(1)  # waits to see if `esc` is pressed

            if i == 4:
                img_ = open_image(Path('./test1.jpg'))
                label, index_, pred = learn.predict(img_)
                res = str(label)
                score = pred[0]

                i = 0
                if mem == res:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive == 2 and res not in ['nothing']:
                    if res == 'space':
                        sequence += ' '
                    elif res == 'del':
                        sequence = sequence[:-1]
                    else:
                        sequence += res
                    consecutive = 0

            i += 1
            cv2.putText(img, '%s' % (res.upper()), (100, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4)
            cv2.putText(img, '(score = {})'.format(round(float(score), 2)),
                        (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            mem = res
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # img_sequence = np.zeros((200, 1200, 3), np.uint8)
            # cv2.putText(img_sequence, '%s' % (sequence.upper()),
            #             (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.imshow('sequence', img_sequence)
        try:
            with lock:
                outputFrame = img.copy()
        except:
            pass


def generate():
        # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
                # wait until the lock is acquired
        with lock:
                        # check if the output frame is available, otherwise skip
                        # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
        # return the response generated along with the specific media
        # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.form["submit_button"] == "Predict the Alphabet":

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
        file.save(str(path / image_path))

        img = open_image(path / image_path)
        label, index, pred = learn.predict(img)

        return render_template("linguahome.html", name=label)

    elif request.form["submit_button"] == "Capture Video":
        # start a thread that will perform motion detection
        # outputFrame = None
        # lock = threading.Lock()
        global cap

        cap = cv2.VideoCapture(0)
        # c = 0
        time.sleep(2.0)

        t = threading.Thread(target=recognize_gesture, args=(cap, 32,))
        t.daemon = True
        t.start()

        return redirect(url_for('video'))

        # return render_template("linguavideo2.html")
        # return render_template("success.html")

cap.release()
app.run(port=5000)
