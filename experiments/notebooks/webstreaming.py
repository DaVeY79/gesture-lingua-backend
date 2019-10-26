
# import the necessary packages
from flask import Response, Flask, render_template
import threading
import argparse
import datetime
import time
import cv2
from fastai.vision import (
    ImageDataBunch,
    get_transforms,
    cnn_learner,
    imagenet_stats,
    open_image,
    models,
)
import numpy as np
import torch
import pickle
from pathlib import Path


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

path = Path("/Users/davidabraham/gesture-lingua-backend/experiments/notebooks")

with open(path / "class_names.pkl", "rb") as pkl_file:
    classes = pickle.load(pkl_file)


data2 = ImageDataBunch.single_from_classes(
    path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = cnn_learner(data2, models.resnet34)
learn.load('model-after-unfreeze')

# initialize the video stream and allow the camera sensor to
# warmup
cap = cv2.VideoCapture(0)
c = 0
time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("linguavideo2.html")


def recognize_gesture(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock, c

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

            c += 1
            # image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
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
            cv2.putText(img, '(score = {})'.format(round(float(score),2)),
                        (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            mem = res
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # img_sequence = np.zeros((200, 1200, 3), np.uint8)
            # cv2.putText(img_sequence, '%s' % (sequence.upper()),
            #             (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.imshow('sequence', img_sequence)

        with lock:
            outputFrame = img.copy()


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


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=recognize_gesture, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
cap.release()
