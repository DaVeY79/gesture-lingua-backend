import numpy as np
import threading
import argparse
import time
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

app = Flask(__name__)

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
frameCount = None
cam_flag = False

@app.route("/", methods=["GET"])
def upload_page():
    cap.release()
    return render_template("linguahome.html")

@app.route("/video", methods=["GET","POST"])
def video():
    global cap
    if request.method == "GET":
            # return the rendered template for video
        return render_template("linguavideo.html")
    else:
        if request.form["submit_button"] == "Return to home page":
            return redirect(url_for("upload_page"))
            # cap.release()
            # return render_template("linguahome.html")
        elif request.form["submit_button"] == "Close video":
            cap.release()
            return render_template("linguavideo.html",cam_flag = True)

@app.route("/release", methods=["POST"])
def release():
    global cap
    cap.release()
    return render_template("linguavideo.html",cam_flag = True)

def recognize_gesture(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock, cap, cam_flag

    ret, score = '', 0.0
    i = 0
    mem = ''
    consecutive = 0
    sequence = ''
    try:
        while True:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            res = ''
            if ret:
                # x1, y1, x2, y2 = 100, 100, 700, 700
                x1, y1, x2, y2 = 600, 50, 1200, 650
                img_cropped = img[y1:y2, x1:x2]

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
                # img_sequence = np.zeros((200, 1280, 3), np.uint8)
                # cv2.putText(img_sequence, '%s' % (sequence.upper()),
                #             (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # resize_img_sequence = cv2.resize(img_sequence,(img_sequence.shape[0],img.shape[1]))
                # img = np.vstack((img,img_sequence))

                try:
                    with lock:
                        outputFrame = img.copy()
                except:
                    cam_flag = True
    except:
        pass

    finally:
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

        # if cam_flag:
        #     break

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
        file = request.files['file']
        file.save(str(path / image_path))
        img = open_image(path / image_path)
        label, index, pred = learn.predict(img)
        return render_template("linguahome.html", name=label)

    elif request.form["submit_button"] == "Click an Image":
        return render_template("sucess.html")

    elif request.form["submit_button"] == "Capture Video":
        # start a thread that will perform motion detection
        global cap
        cap = cv2.VideoCapture(0)
        time.sleep(2.0)
        t = threading.Thread(target=recognize_gesture, args=(frameCount,))
        t.daemon = True
        t.start()
        return redirect(url_for('video'))


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
    frameCount = args["frame_count"]
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)
    cap.release()
